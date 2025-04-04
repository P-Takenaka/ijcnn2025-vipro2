from typing import Dict, Any, List, Optional, Union
import ml_collections
from torch import nn
import torch
from torch.nn import functional as F

from src.lib import init_module, elup1

from .physics import FunctionModule
from .misc import Identity
from .rnn import RNN

class SeparableLatentCoder(nn.Module):
    def __init__(
            self, input_sizes: List[int],
            output_sizes: List[int], model: Union[Dict[str, Any], ml_collections.ConfigDict]):
        super().__init__()

        self.input_sizes = input_sizes
        self.output_sizes = output_sizes

        models = []
        for input_size, output_size in zip(input_sizes, output_sizes):
            models.append(init_module(model, input_size=input_size, output_size=output_size))
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor):
        z = torch.split(x, split_size_or_sections=self.input_sizes, dim=-1)
        assert(len(z) == len(self.models))

        res = torch.cat([model(_z) for _z, model in zip(z, self.models)], dim=-1)

        return res


class ProcModule(nn.Module):
    def __init__(
            self,
            z_a_size:int,
            z_b_size: int,
            z_c_size: int,
            F_module: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]],
            F_out: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]]=None,
            F_in: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]]=None,
            D_module: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]]=None,

            state_keys: Optional[List[str]]=None,
            state_sizes: Optional[List[int]]=None,

            observable_state_keys: Optional[List[str]]=None,
            prepend_background: Optional[bool]=None,
        ):
        super().__init__()

        self.z_a_size = z_a_size
        self.z_b_size = z_b_size
        self.z_c_size = z_c_size


        self.F: Optional[FunctionModule] = init_module(F_module) if F_module is not None else None
        self.D = init_module(D_module) if D_module is not None else None

        if self.F is None:
            assert(state_keys is not None)
            assert(state_sizes is not None)
            assert(observable_state_keys is not None)

            self.state_keys = state_keys
            self.state_sizes = state_sizes
            self.observable_state_keys = observable_state_keys
            self.observable_state_sizes = [self.state_sizes[self.state_keys.index(k)] for k in self.observable_state_keys]
        else:
            self.state_keys = self.F.state_keys
            self.state_sizes = self.F.state_sizes
            self.observable_state_keys = self.F.observable_state_keys
            self.observable_state_sizes = self.F.observable_state_sizes

        self.unobservable_state_keys = [k for k in self.state_keys if k not in self.observable_state_keys]
        self.unobservable_state_sizes = [s for k, s in zip(self.state_keys, self.state_sizes) if k in self.unobservable_state_keys]

        self.prepend_background = self.F.prepend_background if self.F is not None else prepend_background
        assert(self.prepend_background is not None)

        num_states = len(self.state_keys)
        state_sizes = self.state_sizes

        if F_in:
            assert(not self.z_a_size % num_states)
            self.F_in = SeparableLatentCoder(
                input_sizes=[self.z_a_size // num_states] * num_states,
                output_sizes=state_sizes, model=F_in)
        else:
            self.F_in = None

        if F_out is None:
            self.F_out = None
        else:
            assert(not self.z_a_size % num_states)
            self.F_out = SeparableLatentCoder(
                input_sizes=state_sizes,
                output_sizes=[self.z_a_size // num_states] * num_states,
                model=F_out)

    @property
    def has_object_dim(self):
        return True

    def read_sym_state(self, z: torch.Tensor):
        assert(self.F_in is not None)

        if self.prepend_background:
           s: torch.Tensor = self.F_in(z[..., :self.z_a_size])
        else:
           s: torch.Tensor = self.F_in(z[..., 1:, :self.z_a_size])

        return s

    def store_sym_state(self, z: torch.Tensor, sym_state_tensor: torch.Tensor):
        assert(z.shape[-1] == self.z_a_size + self.z_b_size + self.z_c_size)
        assert(self.F_out is not None)

        inp = self.F_out(sym_state_tensor)

        if not self.prepend_background:
            # Take bg from z
            inp = torch.cat([z[..., :1, :self.z_a_size], inp], dim=-2)

        # Combine z_a and rest of tensor
        res = torch.cat([inp, z[..., self.z_a_size:]], dim=-1)

        return res

    def split_z(self, z: torch.Tensor):
        return torch.split(z, split_size_or_sections=[self.z_a_size, self.z_b_size, self.z_c_size], dim=-1)

    def remove_unobservables(self, z):
        assert(z.shape[-1] == self.z_a_size + self.z_b_size + self.z_c_size)

        z_a = self.get_z_a(z)

        z_a_obs = self.convert_tensor_to_observable_tensor(z_a)

        return torch.cat([z_a_obs, z[..., self.z_a_size:]], dim=-1)

    def remove_z_a(self, z: torch.Tensor):
        assert(self.z_a_size > 0)
        assert(self.z_b_size + self.z_c_size > 0)
        assert(z.shape[-1] == self.z_a_size + self.z_b_size + self.z_c_size)

        return z[..., self.z_a_size:]

    def get_z_a(self, z: torch.Tensor):
        assert(z.shape[-1] == self.z_a_size + self.z_b_size + self.z_c_size)
        assert(self.z_a_size > 0)

        return z[..., :self.z_a_size]

    def get_z_c(self, z: torch.Tensor):
        assert(z.shape[-1] == self.z_a_size + self.z_b_size + self.z_c_size)
        assert(self.z_c_size > 0)

        return z[..., -self.z_c_size:]

    def swap_z_b(self, z: torch.Tensor, idxs):
        assert(not self.num_objects and self.has_temporal_dim) # Has object dim
        z_a, z_b, z_c = self.split_z(z)

        z_b = z_b[:, :, idxs]

        return torch.cat([z_a, z_b, z_c], dim=-1)

    def swap_z_c(self, z: torch.Tensor, idxs):
        assert(not self.num_objects and self.has_temporal_dim) # Has object dim
        z_a, z_b, z_c = self.split_z(z)

        z_c = z_c[:, :, idxs]

        return torch.cat([z_a, z_b, z_c], dim=-1)

    def forward(self, z: torch.Tensor, z_a_sym: torch.Tensor,
                frame_idx: int, dataloader_idx: int=0,
                action: Optional[torch.Tensor]=None,
                **kwargs):
        assert((z.shape[-1] == (self.z_a_size + self.z_b_size + self.z_c_size)))
        assert(len(z.shape) == 4)
        if self.F is None:
            assert(self.D is not None)

        if self.D is not None:
            z_b = self.D(z, **kwargs)
            assert(len(z_b.shape) == 4)
            z_b = z_b[:, -1:]

            if self.F is None:
                assert(self.F_in is not None)
                z_a, z_b = torch.split(z_b, split_size_or_sections=[self.z_a_size, self.z_b_size], dim=-1)

                z_a_sym = self.F_in(z_a)[:, -1]

        else:
            assert(not self.z_b_size)
            z_b = None

        if self.F is not None:
            z_a_sym = self.F(
                z_a_sym, frame_idx=frame_idx, action=action)

        assert(z_a_sym is not None)
        assert(self.F_out is not None)
        if self.z_a_size == 0:
            z_a = None
        else:
            z_a = self.F_out(z_a_sym)

            assert(len(z_a.shape) == 3)

            z_a = torch.unsqueeze(z_a, dim=1)

        res = []
        if z_a is not None:
            assert(len(z.shape) == 4)

            if self.prepend_background:
                res.append(z_a)
            else:
                res.append(torch.cat([self.get_z_a(z[:, -1:, :1]), z_a], dim=2))

        if z_b is not None:
            res.append(z_b)

        if self.z_c_size:
            z_c = self.get_z_c(z[:, -1:])

            res.append(z_c)

        assert(res)

        z = torch.cat(res, dim=-1)

        return z, z_a_sym

    def convert_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        res = torch.cat([d[k] for k in self.state_keys], dim=-1)

        return res

    def convert_tensor_to_state_dict(self, z: torch.Tensor):
        z_split = torch.split(z, z.shape[-1] // len(self.state_keys), dim=-1)

        res: Dict[str, torch.Tensor] = {}
        for k, z_k in zip(self.state_keys, z_split):
            res[k] = z_k

        return res

    def convert_unobservable_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        res = torch.cat([d[k] for k in self.unobservable_state_keys], dim=-1)

        return res

    def convert_observable_state_dict_to_tensor(self, d: Dict[str, torch.Tensor]):
        res = torch.cat([d[k] for k in self.observable_state_keys], dim=-1)

        return res

    def convert_tensor_to_observable_tensor(self, z: torch.Tensor):
        z_dict = self.convert_tensor_to_state_dict(z=z)

        res_dict: Dict[str, torch.Tensor] = {k: z_dict[k] for k in self.observable_state_keys}

        return self.convert_observable_state_dict_to_tensor(res_dict)

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

    def merge_observable_and_unobservable_tensor(self, observable: torch.Tensor, unobservable: torch.Tensor):
        observable_dict = self.convert_observable_tensor_to_state_dict(observable)
        unobservable_dict = self.convert_unobservable_tensor_to_state_dict(unobservable)

        res = {k: observable_dict[k] if k in observable_dict else unobservable_dict[k] for k in self.state_keys}

        return self.convert_state_dict_to_tensor(res)

    def convert_tensor_to_unobservable_tensor(self, z: torch.Tensor):
        z_dict = self.convert_tensor_to_state_dict(z=z)

        res_dict: Dict[str, torch.Tensor] = {k: z_dict[k] for k in self.unobservable_state_keys}

        return self.convert_unobservable_state_dict_to_tensor(res_dict)
