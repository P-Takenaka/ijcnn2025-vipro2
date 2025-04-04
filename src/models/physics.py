from typing import List, Optional, Dict, Any, Union
import torch.nn as nn
import torch
import copy
import logging
import ml_collections

import torch
from torch import nn

from src.lib import EngineUnrollMAE, elup1

import do_mpc
from casadi import cos, sin
import numpy as np

from src.lib import init_module

class FunctionStateVariable():
    def __init__(self, name: str, sym_size: int, latent_size: Optional[int], observability: str):
        self.name = name
        self.sym_size = sym_size
        self.latent_size = latent_size
        self.observability = observability

        self._latent_value: Optional[torch.Tensor] = None
        self._sym_value: Optional[torch.Tensor] = None

    def is_fully_observable(self):
        return self.observability == 'full'

    def is_partially_observable(self):
        return self.observability == 'partial'

    def is_unobservable(self):
        return self.observability == 'none'



class FunctionModule(nn.Module):
    def __init__(self,
                 state_keys: List[str],
                 state_sizes: List[int],
                 has_object_dim: bool,
                 observable_state_keys: List[str],
                 prepend_background: bool,
                 learned_parameters: Optional[List[str]]=None,
                 return_identity: bool=False,
                 sanity_eps=1e-3,):
        super().__init__()

        self.state_keys = state_keys
        self.observable_state_keys = observable_state_keys
        self.unobservable_state_keys = [k for k in state_keys if k not in self.observable_state_keys]

        self.state_sizes = state_sizes
        self.state_size_sum = sum(self.state_sizes)
        self.observable_state_sizes = [s for k, s in zip(self.state_keys, self.state_sizes) if k in self.observable_state_keys]
        self.observable_state_size_sum = sum(self.observable_state_sizes)

        self.unobservable_state_sizes = [s for k, s in zip(self.state_keys, self.state_sizes) if k in self.unobservable_state_keys]
        self.unobservable_state_size_sum = sum(self.unobservable_state_sizes)

        self.return_identity = return_identity
        self.has_object_dim = has_object_dim
        self.prepend_background = prepend_background
        self.sanity_eps = sanity_eps

        if learned_parameters is None:
            self.learned_parameters = None
        else:
            self.learned_parameters = nn.ParameterDict()
            for p in learned_parameters:
                self.learned_parameters[p] = nn.Parameter(nn.init.normal_(torch.empty(1)))

    def sanity_check(self, batch):
        with torch.no_grad():
            eps = self.sanity_eps

            if eps is None:
                logging.info("Skipping engine sanity check")
                return;
            # Make sure the error is below eps
            keys = self.state_keys

            initial_state = self.get_state_tensor(
                batch['unroll'] if 'unroll' in batch else batch, prepend_background=self.prepend_background)[:, 0]

            metric = EngineUnrollMAE(sync_on_compute=False).to(initial_state.get_device())

            engine_unroll_states = [initial_state]

            unroll_len = batch['unroll']['video'].shape[1] - 1 if 'unroll' in batch else batch['video'].shape[1] - 1

            for i in range(unroll_len):
                engine_unroll_states.append(self(
                    engine_unroll_states[-1], frame_idx=i))

            out_state = self.convert_tensor_to_state_dict(torch.stack(engine_unroll_states, dim=1))

            metric.update(engine_unroll_states=out_state, batch=batch)

            result = metric.compute()

            logging.info(f"Sanity Check Error: {result}")

            gt_state = {k: batch['unroll'][k] if 'unroll' in batch else batch[k] for k in out_state}


            if result > eps:
                per_frame_error = torch.sum(torch.abs(out_state['positions'] - gt_state['positions']), dim=[0,2,3])
                logging.info(f"Per Frame Error: {per_frame_error}")

                gt_state = {k: v[0] for k, v in gt_state.items()}
                out_state = {k: v[0] for k, v in out_state.items()}
                logging.info(f"GT States: {gt_state}")
                logging.info(f"Predictions: {out_state}")
                raise ValueError(f"Engine error {result} is larger than expected ({eps})!")

    def merge_observable_and_unobservable_tensor(self, observable: torch.Tensor, unobservable: torch.Tensor):
        observable_dict = self.convert_observable_tensor_to_state_dict(observable)
        unobservable_dict = self.convert_unobservable_tensor_to_state_dict(unobservable)

        res = {k: observable_dict[k] if k in observable_dict else unobservable_dict[k] for k in self.state_keys}

        return self.convert_state_dict_to_tensor(res)

    def convert_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        res = torch.cat([d[k] for k in self.state_keys], dim=-1)

        return res

    def convert_observable_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        assert(self.observable_state_keys)

        res = torch.cat([d[k] for k in self.observable_state_keys], dim=-1)

        return res

    def convert_unobservable_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        assert(self.unobservable_state_keys)

        res = torch.cat([d[k] for k in self.unobservable_state_keys], dim=-1)

        return res

    def convert_observable_tensor_to_state_dict(self, t: torch.Tensor):
        assert(self.observable_state_keys)

        t_split = torch.split(t, split_size_or_sections=t.shape[-1] // len(self.observable_state_keys), dim=-1)

        res = {k: v for k, v in zip(self.observable_state_keys, t_split)}

        return res

    def convert_unobservable_tensor_to_state_dict(self, t: torch.Tensor):
        assert(self.unobservable_state_keys)

        t_split = torch.split(t, split_size_or_sections=t.shape[-1] // len(self.unobservable_state_keys), dim=-1)

        res = {k: v for k, v in zip(self.unobservable_state_keys, t_split)}

        return res

    def convert_tensor_to_state_dict(self, z: torch.Tensor):
        z_split = torch.split(z, z.shape[-1] // len(self.state_keys), dim=-1)

        res: Dict[str, torch.Tensor] = {}
        for k, z_k in zip(self.state_keys, z_split):
            res[k] = z_k

        return res

    def convert_tensor_to_observable_tensor(self, z: torch.Tensor):
        if self.observable_state_keys:
            z_dict = self.convert_tensor_to_state_dict(z=z)

            res_dict: Dict[str, torch.Tensor] = {k: z_dict[k] for k in self.observable_state_keys}

            return self.convert_observable_state_dict_to_tensor(res_dict)
        else:
            raise ValueError()

    def convert_tensor_to_unobservable_tensor(self, z: torch.Tensor):
        if self.unobservable_state_keys:
            z_dict = self.convert_tensor_to_state_dict(z=z)

            res_dict: Dict[str, torch.Tensor] = {k: z_dict[k] for k in self.unobservable_state_keys}

            return self.convert_unobservable_state_dict_to_tensor(res_dict)
        else:
            raise ValueError()

    def get_sym_state_size(self):
        return self.state_size_sum

    def get_observable_sym_state_size(self):
        return self.observable_state_size_sum

    def get_unobservable_sym_state_size(self):
        return self.unobservable_state_size_sum

    def get_state_tensor(self, batch: Dict[str, Any], prepend_background: bool):
        assert(prepend_background == self.prepend_background)
        res = torch.cat([batch[k] for k in self.state_keys], dim=-1)
        if prepend_background:
            res = torch.cat([torch.zeros_like(res[:, :, 0:1]), res], dim=2)

        return res

    def _forward(self, previous_states: torch.Tensor, frame_idx: int, action=None) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, previous_states: torch.Tensor, frame_idx: int, action=None):
        if self.return_identity:
            return previous_states
        else:
            return self._forward(previous_states=previous_states, frame_idx=frame_idx, action=action)

class PhysicsFunctionModule(FunctionModule):
    def __init__(self,
                 step_rate: int,
                 frame_rate: int,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = {
            'step_rate': step_rate,
            'frame_rate': frame_rate,
            'simulation_steps': step_rate // frame_rate,
            'frame_rate_dt': 1.0 / frame_rate}
        assert(self.config['simulation_steps'] >= 1)
        self.config['simulation_dt'] = 1.0 / float(step_rate)

    def mpc(self, initial_state, n_horizon=100, n_steps=100):
        raise NotImplementedError()

    def calculate_force(self, state):
        raise NotImplementedError()

    def get_kinetic_energy(self, state):
        raise NotImplementedError()

    def calculate_system_energy(self, state):
        raise NotImplementedError()

class OrbitsPhysicsEngine(PhysicsFunctionModule):
    def __init__(self, G: float, only_2d: bool, obj_mass: float,
                 ghost_z: Optional[float]=None,
                 ghost_inverse_force: bool=False,
                 camera_relative_state=False,
                 camera_pos: Optional[List[float]]=None,
                 **kwargs):
        super().__init__(
            state_keys=['positions', 'velocities'],
            state_sizes=[3, 3],
            has_object_dim=True, observable_state_keys=['positions'],
            **kwargs)

        self.config.update({
            'G': G,
            'only_2d': only_2d,
            'obj_mass': obj_mass,
            'ghost_z': ghost_z})

        self.camera_relative_state = camera_relative_state
        self.camera_pos = camera_pos
        self.ghost_inverse_force = ghost_inverse_force

    def calculate_force(self, state: Dict[str, torch.Tensor], return_pot_e):
        pos = state['positions']
        if self.learned_parameters is not None:
            if 'G' in self.learned_parameters:
                G = elup1(self.learned_parameters['G'])
            else:
                G = self.config['G']

            if 'obj_mass' in self.learned_parameters:
                obj_mass = elup1(self.learned_parameters['obj_mass'])
            else:
                obj_mass = self.config['obj_mass']
        else:
            G = self.config['G']
            obj_mass = self.config['obj_mass']

        mass = torch.ones_like(pos[..., :1]) * obj_mass

        ghost_pos = torch.zeros_like(pos[:,0:1])
        ghost_z: Optional[float] = self.config['ghost_z']

        if ghost_z is not None:
            ghost_pos[..., -1] = ghost_z

        ghost_m = 2.0

        # (1, 9, 9, 3)
        p_diffs = torch.unsqueeze(pos, dim=-3) - torch.unsqueeze(pos, dim=-2)
        # (1, 9, 9, 1)
        r2 = torch.sum(torch.pow(p_diffs, 2), dim=-1, keepdim=True) + 1

        # (1, 9, 9, 1)
        norm = torch.sqrt(r2)

        # (1, 9, 9, 3)
        F_dir = p_diffs / norm

        # (1, 9, 9, 1)
        m_prod = torch.unsqueeze(mass, dim=-3) * torch.unsqueeze(mass, dim=-2)

        pot_e = G * (m_prod / (r2))

        F = torch.sum(F_dir * pot_e, dim=-2)

        if return_pot_e:
            pot_e = torch.sum(pot_e, dim=[-1, -2])

        # Add force to ghost object
        ghost_p_diffs = ghost_pos - pos

        if self.ghost_inverse_force:
            ghost_F = 0.6 * ghost_p_diffs

            if return_pot_e:
                ghost_pot_e = torch.sqrt(torch.sum(torch.square(ghost_p_diffs), dim=-1))
            else:
                ghost_pot_e = None

        else:
            ghost_m_prod = mass * ghost_m

            ghost_r2 = torch.sum(torch.pow(ghost_p_diffs, 2), dim=-1, keepdim=True) + 1

            ghost_norm = torch.sqrt(ghost_r2)

            ghost_F_dir = ghost_p_diffs / ghost_norm

            ghost_pot_e = (G * (ghost_m_prod / (ghost_r2)))

            ghost_F = ghost_F_dir * ghost_pot_e

            if return_pot_e:
                ghost_pot_e = torch.sum(ghost_pot_e, dim=-1)

        F += ghost_F
        if return_pot_e:
            pot_e += ghost_pot_e

        return F, (pot_e if return_pot_e else None)

    def get_kinetic_energy(self, state):
        if self.learned_parameters is not None:
            if 'obj_mass' in self.learned_parameters:
                obj_mass = elup1(self.learned_parameters['obj_mass'])
            else:
                obj_mass = self.config['obj_mass']
        else:
            obj_mass = self.config['obj_mass']

        return 0.5 * obj_mass * torch.sum(state['velocities']**2, dim=-1)

    def calculate_system_energy(self, state):
        ke = self.get_kinetic_energy(state)

        _, pe = self.calculate_force(state, return_pot_e=True)

        return torch.sum(ke + pe, dim=-1)

    def _forward(self, previous_states: torch.Tensor, frame_idx: int, action=None):
        previous_states_dict = self.convert_tensor_to_state_dict(
            previous_states)
        vel = previous_states_dict['velocities']
        pos = previous_states_dict['positions']

        assert(len(pos.shape) == 3)
        assert(len(vel.shape) == 3)

        if self.prepend_background:
            # Background object is ignored
            pos = pos[:, 1:]
            vel = vel[:, 1:]

        if self.learned_parameters is not None:
            if 'obj_mass' in self.learned_parameters:
                obj_mass = elup1(self.learned_parameters['obj_mass'])
            else:
                obj_mass = self.config['obj_mass']
        else:
            obj_mass = self.config['obj_mass']

        simulation_dt = self.config['simulation_dt']
        simulation_steps = self.config['simulation_steps']
        only_2d = self.config['only_2d']

        if self.camera_relative_state:
            assert(self.camera_pos is not None)
            camera_pos = torch.as_tensor(
                self.camera_pos, device=pos.device, dtype=torch.float32)[None, None, :]
            # Convert relative positions to absolute positions. We cannot use relative positions directly due to the epsilons added in the function
            pos = torch.cat([pos[..., :-1], camera_pos[..., -1:] - pos[..., -1:]], dim=-1)
        else:
            camera_pos = None

        if only_2d and pos.shape[-1] == 3:
            # We have 3D positions, but only consider x and y for force calculation
            pos_z = pos[...,-1:]
            pos = torch.cat([pos[...,:-1], torch.zeros_like(pos[...,-1:])], dim=-1)
        else:
            pos_z = None

        for sim_idx in range(simulation_steps):
            F, _ = self.calculate_force({'positions': pos}, return_pot_e=False)

            # F = ma
            a = F / obj_mass

            # Semi implicit euler
            vel = vel + simulation_dt * a
            pos = pos + simulation_dt * vel

            if only_2d and pos.shape[-1] == 3:
                pos = torch.cat([pos[...,:-1], torch.zeros_like(pos[...,-1:])], dim=-1)

        if only_2d and pos.shape[-1] == 3:
            assert(pos_z is not None)
            pos = torch.cat([pos[...,:-1], pos_z], dim=-1)

        if self.camera_relative_state:
            # Convert absolute positions to relative positions
            assert(camera_pos is not None)

            pos = torch.cat([pos[...,:-1], camera_pos[...,-1:] - pos[...,-1:]], dim=-1)

        if self.prepend_background:
            # Add background "object" again
            pos = torch.cat([previous_states_dict['positions'][:, :1], pos], dim=1)
            vel = torch.cat([previous_states_dict['velocities'][:, :1], vel], dim=1)

        res = self.convert_state_dict_to_tensor({'positions': pos, 'velocities': vel},
                                                )

        return res
