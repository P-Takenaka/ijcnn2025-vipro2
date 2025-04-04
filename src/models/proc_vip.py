from typing import Optional, Dict, Union, Any, List, Tuple
import ml_collections
from scipy.optimize import linear_sum_assignment
import torch
import logging

import numpy as np
import torch
from torch import nn

from src.lib import init_module, elup1

from .vip_base import VPBaseModel, slot_decode
from .proc_mod import ProcModule
from .mlp import MLP
from .rnn import RNN

from .slot_attention import SlotAttentionV2
from .physics import PhysicsFunctionModule

def wrap(x: torch.Tensor):
    return torch.arctan2(torch.sin(x), torch.cos(x))

class ProcVIP(VPBaseModel):
    def __init__(
            self,
            state_initializer: Union[Dict[str, Any], ml_collections.ConfigDict],
            state_autoencoder_alignment_factor=0.0,
            state_fusion: Dict[str, Any]={'type': 'constant_gain', 'gain': 0.0},
            gain_predictor: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]]=None,
            burn_in_state_supervision_factor: float=0.0,
            state_observation_alignment_factor=0.0,
            **kwargs
            ):
        super().__init__(**kwargs)

        self.state_autoencoder_alignment_factor = state_autoencoder_alignment_factor
        self.burn_in_state_supervision_factor = burn_in_state_supervision_factor
        self.state_observation_alignment_factor = state_observation_alignment_factor
        self.state_fusion = state_fusion
        self.state_initializer = init_module(state_initializer)
        self.gain_predictor: Optional[RNN] = None if gain_predictor is None else init_module(gain_predictor)

        def fn(a, b):
            err = torch.pow(a - b, 2)
            return err.mean()
        self.state_loss_fn = fn

        assert(isinstance(self.predictor, ProcModule))


    # 1: Take full observed state
    # 0: Take full predicted state
    def _get_state_gain(self, obs: Optional[torch.Tensor], pred: Optional[torch.Tensor]):
        if self.state_fusion['type'] == 'constant_gain':
            return float(self.state_fusion['gain'])
        elif self.state_fusion['type'] == 'learned':
            assert(self.gain_predictor is not None)
            assert(pred is not None)
            assert(obs is not None)

            K = torch.sigmoid(
                self.gain_predictor(torch.cat([obs, pred], dim=-1))[0])

            return K
        else:
            raise ValueError("Unknown state fusion method")

    def decode(self, d):
        z = d['pred_z']
        assert(self.decoder is not None)

        assert(isinstance(self.predictor, ProcModule))

        return super().decode(z)

    def get_initial_state(self, batch_size, batch: Optional[Dict[str, Any]]=None):
        assert(isinstance(self.predictor, ProcModule))

        state_dict = self.state_initializer(batch=batch, batch_size=batch_size, device=self.device)

        if len(self.state_initializer.state_keys) == len(self.predictor.state_keys):
            state = self.predictor.convert_state_dict_to_tensor(state_dict)
        else:
            state = torch.cat([state_dict[k] for k in self.state_initializer.state_keys], dim=-1)

        return state

    def on_first_batch(self, batch, dataloader_idx=0):
        logging.info("Performing engine unroll sanity check")

        assert(isinstance(self.predictor, ProcModule))

        if self.predictor.F is not None:
            self.proc_mod_sanity_check(batch=batch, dataloader_idx=dataloader_idx)
        else:
            logging.info("Skipping sanity check")

        logging.info("Engine unroll sanity check successful")

    def log_training(self, outputs, batch):
        super().log_training(outputs, batch=batch)

        assert(isinstance(self.predictor, ProcModule))

    def get_params_for_optimizer(self, parameters, named_parameters):
        params = [{'params': list(filter(lambda p: p.requires_grad,
                                             parameters))}]

        return params

    def proc_mod_sanity_check(self, batch, dataloader_idx):
        assert(isinstance(self.predictor, ProcModule))
        if self.predictor.F is not None:
            self.predictor.F.sanity_check(batch)
        else:
            logging.info("Skipping sanity check")

    def _rollout_iteration(self, in_x: torch.Tensor, sym_state: Optional[torch.Tensor], 
                           dataloader_idx: int, frame_idx: int, action=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert(isinstance(self.predictor, ProcModule))

        z_out, sym_state = self.predictor(
            z=in_x, z_a_sym=sym_state,
            dataloader_idx=dataloader_idx, frame_idx=frame_idx, action=action)
        # feed the predicted slots autoregressively
        in_x = torch.cat([in_x[:, 1:], z_out], dim=1)

        return in_x, sym_state

    def get_loss(self, batch, outputs, dataloader_idx):
        assert(isinstance(self.predictor, ProcModule))

        loss_dict = super().get_loss(batch, outputs, dataloader_idx)

        if self.state_autoencoder_alignment_factor:
            # predicted state vs. decoded state
            keys = outputs['sym_states'].keys()
            if 'burn_in' in outputs:
                decoded_sym_states = torch.cat([
                    outputs['decoded_sym_states'][k] for k in keys], dim=-1)
                predicted_sym_states = torch.cat([
                    outputs['sym_states'][k] for k in keys], dim=-1)

                decoded_sym_states = torch.cat([
                    torch.cat([outputs['burn_in']['decoded_sym_states'][k] for k in keys],
                              dim=-1), decoded_sym_states], dim=1)
                predicted_sym_states = torch.cat([
                    torch.cat([outputs['burn_in']['sym_states'][k] for k in keys],
                              dim=-1), predicted_sym_states], dim=1)

            else:
                decoded_sym_states = torch.cat([
                    outputs['decoded_sym_states'][k] for k in keys], dim=-1)
                predicted_sym_states = torch.cat([
                    outputs['sym_states'][k] for k in keys], dim=-1)

            if self.predictor.prepend_background:
                decoded_sym_states = decoded_sym_states[..., 1:, :]
                predicted_sym_states = predicted_sym_states[..., 1:, :]

            state_autoencoder_alignment_loss = self.state_loss_fn(decoded_sym_states, predicted_sym_states)

            state_autoencoder_alignment_loss *= self.state_autoencoder_alignment_factor

            loss_dict['state_autoencoder_alignment'] = state_autoencoder_alignment_loss

        if self.state_observation_alignment_factor:
            # predicted state vs. decoded state
            keys = outputs['burn_in']['observed_sym_states'].keys() if 'burn_in' in outputs else outputs['observed_sym_states'].keys()
            if 'burn_in' in outputs:
                observed_sym_states = torch.cat([
                    outputs['burn_in']['observed_sym_states'][k] for k in keys], dim=-1)
                predicted_sym_states = torch.cat([
                    outputs['burn_in']['predicted_sym_states'][k] for k in keys], dim=-1)
            else:
                observed_sym_states = torch.cat([
                    outputs['observed_sym_states'][k] for k in keys], dim=-1)
                predicted_sym_states = torch.cat([
                    outputs['predicted_sym_states'][k] for k in keys], dim=-1)

            if self.predictor.prepend_background:
                observed_sym_states = observed_sym_states[..., 1:, :]
                predicted_sym_states = predicted_sym_states[..., 1:, :]

            state_observation_alignment_loss = self.state_loss_fn(observed_sym_states, predicted_sym_states)

            state_observation_alignment_loss *= self.state_observation_alignment_factor

            loss_dict['state_observation_alignment'] = state_observation_alignment_loss

        if self.burn_in_state_supervision_factor:
            keys = outputs['burn_in']['sym_states'].keys() if 'burn_in' in outputs else outputs['sym_states'].keys()

            if 'burn_in' in outputs:
                predicted_sym_states = torch.cat([
                    outputs['burn_in']['sym_states'][k] for k in keys], dim=-1)
                target_sym_states = torch.cat([batch['burn_in'][k] for k in keys], dim=-1)
            else:
                predicted_sym_states = torch.cat([
                    outputs['sym_states'][k] for k in keys], dim=-1)
                target_sym_states = torch.cat([batch[k] for k in keys], dim=-1)

            assert(isinstance(self.predictor, ProcModule))

            if (self.predictor.prepend_background):
                predicted_sym_states = predicted_sym_states[..., 1:, :]

            if self.predictor.has_object_dim:
                # Do hungarian matching
                errors = []
                for b in range(target_sym_states.shape[0]):
                    pairwise_error = torch.sum(torch.sum(torch.pow(
                        predicted_sym_states[b, :, None, :, :] - target_sym_states[b, :, :, None, :], 2), dim=-1), dim=0)

                    row_ind, col_ind = linear_sum_assignment(
                        pairwise_error.detach().cpu().numpy(), maximize=False)

                    errors.append(pairwise_error[row_ind, col_ind].sum())

                burn_in_state_supervision_loss = torch.stack(errors, dim=0).sum() / target_sym_states.numel()
            else:
                burn_in_state_supervision_loss = self.state_loss_fn(target_sym_states, predicted_sym_states)

            burn_in_state_supervision_loss *= self.burn_in_state_supervision_factor

            loss_dict['burn_in_state_supervision'] = burn_in_state_supervision_loss

        return loss_dict

    def _rollout(self, pred_len, batch, previous_state, frame_start_idx, dataloader_idx=0, action_sequence=None, swap_z_b_idxs=None, swap_z_c_idxs=None):
        assert(isinstance(self.predictor, ProcModule))

        in_x: torch.Tensor = previous_state['pred_z'][:, -self.num_context_frames:]

        if swap_z_b_idxs is not None:
            in_x = self.predictor.swap_z_b(in_x, swap_z_b_idxs)

        if swap_z_c_idxs is not None:
            in_x = self.predictor.swap_z_c(in_x, swap_z_c_idxs)

        sym_state = self.predictor.convert_state_dict_to_tensor(
            previous_state['sym_states'])[:, -1]

        # generate future slots autoregressively
        pred_out = []
        sym_states: List[Dict[str, torch.Tensor]] = []
        decoded_sym_states: List[Dict[str, torch.Tensor]] = []

        for i in range(frame_start_idx, frame_start_idx + pred_len):
            in_x, sym_state = self._rollout_iteration(
                in_x=in_x, sym_state=sym_state,
                dataloader_idx=dataloader_idx, frame_idx=i,
                action=action_sequence[:, i] if action_sequence is not None else None)

            assert(sym_state is not None)

            pred_out.append(in_x[:, -1])

            if self.state_autoencoder_alignment_factor:
                decoded_state = self.predictor.read_sym_state(in_x[:, -1])

                decoded_sym_states.append(self.predictor.convert_tensor_to_state_dict(decoded_state))

            sym_states.append(self.predictor.convert_tensor_to_state_dict(sym_state))

        res = {'pred_z': torch.stack(pred_out, dim=1),
               'sym_states': {k: torch.stack([v[k] for v in sym_states], dim=1) for k in sym_states[0].keys()}}

        if decoded_sym_states:
            res['decoded_sym_states'] = {k: torch.stack([v[k] for v in decoded_sym_states], dim=1) for k in decoded_sym_states[0].keys()}

        return res

class ProcSlotVIP(ProcVIP):
    def __init__(
            self,
            slot_size: int,
            num_slots: int,
            slot_attention: Union[Dict[str, Any], ml_collections.ConfigDict],
            per_slot_params=False,
            pre_decoder: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]]=None,
            **kwargs
            ):
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.per_slot_params = per_slot_params

        if 'max_instances' in kwargs:
            del kwargs['max_instances']
        super().__init__(**kwargs, max_instances=self.num_slots - 1)

        assert(isinstance(self.predictor, ProcModule))

        self.pre_decoder = init_module(pre_decoder) if pre_decoder is not None else None

        self.bg_init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, 1, self.slot_size)), requires_grad=True)

        self.fg_init_latents = nn.Parameter(
                nn.init.normal_(torch.empty(1, self.num_slots - 1 if self.per_slot_params else 1, self.slot_size - self.predictor.z_a_size)), requires_grad=True)

        self.init_latents = None

        self.slot_attention: SlotAttentionV2  = init_module(slot_attention, in_features=self.slot_size,
                                          num_slots=self.num_slots, slot_size=self.slot_size)

    def _encode(self, img: torch.Tensor, batch: Dict[str, Any], dataloader_idx: int, frame_start_idx: int,
                prev_state: Optional[Dict[str, Any]]=None):
        assert(isinstance(self.predictor, ProcModule))

        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        z = self._encode_video(img)
        # `encoder_out` has shape: [B, T, H*W, out_features]

        all_pred_z: List[torch.Tensor] = []
        sym_states: List[Dict[str, torch.Tensor]] = []
        decoded_sym_states: List[Dict[str, torch.Tensor]] = []
        observed_sym_states: List[Dict[str, torch.Tensor]] = []
        predicted_sym_states: List[Dict[str, torch.Tensor]] = []

        prev_z: Optional[torch.Tensor] = None

        # init slots
        if prev_state is None:
            assert(frame_start_idx == 0)
            sym_state = self.get_initial_state(batch=batch, batch_size=B)

            f_out_input = sym_state

            assert(batch is not None)

            assert(self.predictor.F_out is not None)

            z_a = self.predictor.F_out(f_out_input)
            assert(self.bg_init_latents is not None)
            assert(self.fg_init_latents is not None)
            bg_init_latents = self.bg_init_latents.repeat(B, 1, 1)
            fg_init_latents = self.fg_init_latents.repeat(B, 1 if self.per_slot_params else self.num_slots -1, 1)

            if self.predictor.z_a_size > 0:
                assert(self.predictor.F_out is not None)
                # Prepend encoded sym state to init latents
                fg_init_latents = torch.cat([
                    z_a, fg_init_latents], dim=-1)

            init_latents = torch.cat([bg_init_latents, fg_init_latents], dim=1)

            if self.state_autoencoder_alignment_factor:
                decoded_state = self.predictor.read_sym_state(init_latents)

                decoded_sym_states.append(self.predictor.convert_tensor_to_state_dict(decoded_state))

            if self.gain_predictor is not None:
                self.gain_predictor.reset(None, use_encoder=False)
        else:
            prev_z = prev_state['pred_z'][:, -self.num_context_frames:]

            sym_state: Optional[torch.Tensor] = self.predictor.convert_state_dict_to_tensor(prev_state['sym_states'])[:, -1]

            init_latents = None

        latents: Optional[torch.Tensor] = None
        all_masks = []
        for idx in range(T):
            # init
            if prev_z is None:
                assert(init_latents is not None)
                latents = init_latents  # [B, N, C]

            else:
                #
                # PREDICTION
                #

                # Use up to self.num_burn_in_frames of representations as reference
                x = prev_z[:, -self.num_context_frames:]

                latents, sym_state = self._rollout_iteration(
                    in_x=x, sym_state=sym_state,
                    dataloader_idx=dataloader_idx,
                    frame_idx=frame_start_idx + idx)

                if self.state_autoencoder_alignment_factor:
                    assert(sym_state is not None)
                    decoded_state = self.predictor.read_sym_state(latents[:, -1])

                    decoded_sym_states.append(self.predictor.convert_tensor_to_state_dict(decoded_state))

                latents = latents[:, -1]

            assert(latents is not None)
            assert(sym_state is not None)

            #
            # OBSERVATION
            #
            obs_z, masks = self.slot_attention.forward(z[:, idx], latents)

            obs_sym_state = self.predictor.read_sym_state(obs_z)

            if masks is not None:
                all_masks.append(masks)

            #
            # UPDATE
            #

            observed_sym_states.append(self.predictor.convert_tensor_to_state_dict(obs_sym_state))

            predicted_sym_states.append(self.predictor.convert_tensor_to_state_dict(
                sym_state))

            # 1: Take full observed state
            # 0: Take full predicted state
            if self.state_fusion['type'] == 'learned':
                K = self._get_state_gain(
                    obs=self.predictor.get_z_a(obs_z if self.predictor.prepend_background else obs_z[:, 1:]),
                    pred=self.predictor.get_z_a(latents if self.predictor.prepend_background else latents[:, 1:]))
            else:
                K = self._get_state_gain(obs=None, pred=None)

            # Update state estimate
            predicted_observable_sym_state = self.predictor.convert_tensor_to_observable_tensor(sym_state)
            predicted_unobservable_sym_state = self.predictor.convert_tensor_to_unobservable_tensor(sym_state)
            if type(K) == float and K == 0:
                # Shortcut: We do not need to reencode the state as nothing is changing
                pred_z =  obs_z
            else:
                sym_state = sym_state + K * (obs_sym_state - sym_state)

                pred_z = self.predictor.store_sym_state(
                    z=obs_z, sym_state_tensor=sym_state)

            sym_states.append(self.predictor.convert_tensor_to_state_dict(sym_state))

            all_pred_z.append(pred_z)

            # next timestep
            prev_z = torch.cat([prev_z, pred_z.unsqueeze(1)], dim=1) if prev_z is not None else pred_z.unsqueeze(1)

        pred_z = torch.stack(all_pred_z, dim=1)

        res = {
            'pred_z': pred_z,
            'sym_states': {k: torch.stack([v[k] for v in sym_states], dim=1) for k in sym_states[0].keys()},
        }

        if all_masks:
            all_masks = torch.stack(all_masks, dim=1)
            all_masks = all_masks.unflatten(-1, (H, W))
            res['masks'] = all_masks

        if decoded_sym_states:
            res['decoded_sym_states'] = {k: torch.stack([v[k] for v in decoded_sym_states], dim=1) for k in decoded_sym_states[0].keys()}
        if observed_sym_states:
            res['observed_sym_states'] = {k: torch.stack([v[k] for v in observed_sym_states], dim=1) for k in observed_sym_states[0].keys()}

        if predicted_sym_states:
            res['predicted_sym_states'] = {k: torch.stack([v[k] for v in predicted_sym_states], dim=1) for k in predicted_sym_states[0].keys()}

        return res

    def _get_decoder_out(self, z, **kwargs):
        assert(len(z.shape) == 3)
        assert(self.decoder is not None)

        bs, num_slots, slot_size = z.shape

        z = z.view(bs * num_slots, slot_size)

        if self.pre_decoder is not None:
            z = self.pre_decoder(z)

        out = self.decoder(z)

        C, H, W = out.shape[1:]

        out = out.view((bs, num_slots, C, H, W))
        assert(len(out.shape) == 5)

        return out

    def decode(self, d):
        z = d['pred_z'].flatten(0, 1)
        assert(self.decoder is not None)

        assert(isinstance(self.predictor, ProcModule))

        # Remove unobservables before decoding
        z = self.predictor.remove_unobservables(z)


        out = self._get_decoder_out(z)

        return slot_decode(z=out, reconstruct_rgb=self.reconstruct_rgb)
