from typing import Optional, List, Dict, Any, Union
import logging
import torch
import ml_collections
import torchmetrics
from src.models import BaseModel
import lpips

import torch
from torch import nn
from torch.nn import functional as F

from src.lib import init_module, concat_dict, detach_dict, elup1
from .model import BaseModel

from .slot_attention import SlotAttentionV2

def get_output_slice(out_dict: Dict[str, Any], i: int):
    result = {}
    for k, v in out_dict.items():
        if k == 'loss' or k == 'batch_idx' or k == 'loss_dict':
            continue

        if type(v) == dict:
            result[k] = get_output_slice(v, i)
        else:
            result[k] = v[:, i:i+1] if v is not None else v

    return result

def slot_decode(z: torch.Tensor, reconstruct_rgb: bool):
    masks = z[:, :, -1:, :, :]
    masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]

    channel_idx = 0

    if reconstruct_rgb:
        recons_rgb = torch.sum(z[:, :, channel_idx:channel_idx + 3, :, :] * masks, dim=1)
        channel_idx += 3
    else:
        recons_rgb = None

    assert(z.shape[2] == channel_idx + 1)

    res = {'rgb_recon': recons_rgb}

    return res

class VPBaseModel(BaseModel):
    def __init__(
            self,
            encoder: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]],
            decoder: Optional[Union[Dict[str, Any], ml_collections.ConfigDict]],
            predictor: Union[Dict[str, Any], ml_collections.ConfigDict],
            num_context_frames: int,
            reconstruct_rgb: bool=True,
            max_concurrent_frames: int=12, # During validation due to memory concerns
            num_val_frames: Optional[int]=None,
            max_instances: Optional[int]=None,
            use_lpips_loss: bool=False,
            **kwargs):
        self.max_concurrent_frames = max_concurrent_frames
        self.num_context_frames = num_context_frames
        self.reconstruct_rgb = reconstruct_rgb
        self.use_lpips_loss = use_lpips_loss

        self.reconstruct_rgb_loss = self.reconstruct_rgb

        assert(self.max_concurrent_frames >= self.num_context_frames)

        self.last_val_metrics_framewise: Optional[torch.nn.ModuleDict] = None
        self.val_metrics_framewise: Optional[torch.nn.ModuleDict] = None

        self.num_val_frames = num_val_frames
        self.max_instances = max_instances

        super().__init__(**kwargs)

        if self.use_lpips_loss:
            assert(not self.strict_checkpoint_loading)
            self.lpips_loss = lpips.LPIPS(net='vgg').eval()
            for p in self.lpips_loss.parameters():
                p.requires_grad = False
        else:
            self.lpips_loss = None


        self.predictor: nn.Module = init_module(predictor)
        self.encoder: Optional[nn.Module] = init_module(encoder) if encoder is not None else None
        self.decoder: Optional[nn.Module] = init_module(decoder) if decoder is not None else None

    def _get_decoder_out(self, z, **kwargs):
        assert(self.decoder is not None)
        assert(len(z.shape) == 2)
        out = self.decoder(z)

        return out

    def _encode_video(self, video: torch.Tensor):
        assert(self.encoder is not None)
        B, T, C, H, W = video.shape
        video = video.flatten(0, 1)

        return self.encoder(video).unflatten(0, (B, T))

    def encode(self, img: torch.Tensor, batch: Dict[str, Any], dataloader_idx: int, prev_state: Optional[Dict[str, Any]]=None):
        assert(len(img.shape) == 5)

        return self._encode(img, batch=batch, prev_state=prev_state,
                            dataloader_idx=dataloader_idx, frame_start_idx=0)

    def rollout(self, pred_len: int, batch: Dict[str, Any], frame_start_idx: int, previous_state: Dict[str, Any], dataloader_idx: int=0):
        result = self._rollout(previous_state=previous_state,
                                pred_len=pred_len, batch=batch, dataloader_idx=dataloader_idx,
                                frame_start_idx=frame_start_idx)
        B = batch['unroll']['video'].shape[0]
        T = pred_len

        out_dict = self.decode(result)
        pred_z = result.pop('pred_z')

        out_dict = {k: v.unflatten(0, (B, T)) if v is not None else v for k, v in out_dict.items()}
        out_dict.update(result)

        out_dict['pred_z'] = pred_z

        return out_dict

    def forward(self, data_dict: Dict[str, Any], dataloader_idx: int=0):
        if 'burn_in' in data_dict:
            # Video Prediction Mode
            B, T_unroll = data_dict['unroll']['video'].shape[:2]
            T_burn_in = data_dict['burn_in']['video'].shape[1]

            burn_in_video = data_dict['burn_in']['video']

            # Encode burn in frames
            burn_in_encoded = self.encode(
                img=burn_in_video,
                batch=data_dict['burn_in'],
                dataloader_idx=dataloader_idx,
                prev_state=None)

            burn_in_decoded = self.decode(burn_in_encoded)

            previous_state = burn_in_encoded
            if T_unroll <= self.max_concurrent_frames or self.training:
                out_dict = self.rollout(
                    previous_state=previous_state, pred_len=T_unroll, batch=data_dict,
                    dataloader_idx=dataloader_idx, frame_start_idx=0)
            else:
                # Split along temporal dim
                cat_dict = None
                for clip_idx in range(0, T_unroll, self.max_concurrent_frames):
                    output = self.rollout(
                        pred_len=min(self.max_concurrent_frames, T_unroll - clip_idx), batch=data_dict,
                        dataloader_idx=dataloader_idx, frame_start_idx=clip_idx,
                        previous_state=previous_state)

                    previous_state = output

                    # because this should be in test mode, we detach the outputs
                    output = detach_dict(output)
                    if cat_dict is None:
                        cat_dict = {k: [v] if v is not None else v for k, v in output.items()}
                    else:
                        for k, v in output.items():
                            if v is not None:
                                cat_dict[k].append(v)

                    torch.cuda.empty_cache()
                assert(cat_dict is not None)
                out_dict = concat_dict(cat_dict, dim=1)

            assert(burn_in_decoded is not None)
            burn_in_encoded.update(
                {k: v.unflatten(0, (B, T_burn_in)) if v is not None else v
                 for k, v in burn_in_decoded.items()})

            burn_in_encoded.pop('pred_z')
            out_dict['burn_in'] = burn_in_encoded
            torch.cuda.empty_cache()

            return out_dict

        else:
            return self._forward(prev_state=None, dataloader_idx=dataloader_idx, batch=data_dict, frame_idx_start=0)


    def _forward(self, batch, dataloader_idx, frame_idx_start, prev_state=None):
        img = batch['video'][:, frame_idx_start:frame_idx_start + self.max_concurrent_frames]
        assert(len(img.shape) == 5)

        B, T = img.shape[:2]
        out_dict = \
            self.encode(img, prev_state=prev_state, batch=batch,
                        dataloader_idx=dataloader_idx)

        post_dict = self.decode(out_dict)

        out_dict.update(
            {k: v.unflatten(0, (B, T)) if v is not None else v
             for k, v in post_dict.items()})

        return out_dict

    def decode(self, d: Dict[str, Any]):
        z = d['pred_z'].flatten(0, 1)

        out = self._get_decoder_out(z)
        assert(len(out.shape) == 4)

        channel_idx = 0

        if self.reconstruct_rgb:
            recons_rgb = out[:, channel_idx:channel_idx + 3]
            channel_idx += 3
        else:
            recons_rgb = None

        assert(out.shape[1] == channel_idx)

        res = {'rgb_recon': recons_rgb}

        return res

    def _encode(self, img: torch.Tensor, batch: Dict[str, Any], dataloader_idx: int, frame_start_idx: int, prev_state: Optional[Dict[str, Any]]=None):
        assert(len(img.shape) == 5)

        z = self._encode_video(img)

        return {'pred_z': z}

    def _rollout(self, pred_len: int, batch: Dict[str, Any], previous_state: Dict[str, Any], frame_start_idx: int, dataloader_idx: int=0):
        z = previous_state['pred_z'][:, -self.num_context_frames:]

        # generate future z autoregressively
        pred_out = []
        for _ in range(pred_len):
            assert(self.predictor is not None)
            z_next = self.predictor(z)[:, -1]  # [B, N, C]
            assert(len(z_next.shape) == 2)

            pred_out.append(z_next)

            # feed the predicted slots autoregressively
            z = torch.cat([z[:, 1:], pred_out[-1].unsqueeze(1)], dim=1)

        return {'pred_z': torch.stack(pred_out, dim=1)}

    def setup_metrics(self, metrics: Optional[Dict[str, Any]]):
        if metrics is None:
            return None, None

        # New metric format
        train_metrics = {}
        val_metrics = {}
        val_metrics_framewise = {}
        for k, metric_config in metrics.items():
            if metric_config.get('only_final', False):
                continue

            if metric_config['train_or_val'] not in ['train', 'val', 'both']:
                raise ValueError("Invalid metric config")

            # Training metrics
            if metric_config['train_or_val'] in ['train', 'both']:
                if metric_config['burn_in_or_unroll'] not in ['burn_in', 'unroll', 'both']:
                    raise ValueError("Invalid metric config")

                # Unroll metrics
                if metric_config['burn_in_or_unroll'] in ['unroll', 'both']:
                    train_metrics[f'train/unroll/{k}'] = metric_config['module'](burn_in=False)

                # burn_in metrics
                if metric_config['burn_in_or_unroll'] in ['burn_in', 'both']:
                    train_metrics[f'train/burn_in/{k}'] = metric_config['module'](burn_in=True)

            # Validation metrics
            if metric_config['train_or_val'] in ['val', 'both']:
                if metric_config['burn_in_or_unroll'] not in ['burn_in', 'unroll', 'both']:
                    raise ValueError("Invalid metric config")

                # Unroll metrics
                if metric_config['burn_in_or_unroll'] in ['unroll', 'both']:
                    val_metrics[f'val/unroll/{k}'] = metric_config['module'](burn_in=False)

                    if metric_config.get('framewise', False):
                        assert(self.num_val_frames is not None)
                        val_metrics_framewise[f'val/unroll/{k}'] = torch.nn.ModuleList([metric_config['module'](burn_in=False) for _ in range(self.num_val_frames)])


                # burn_in metrics
                if metric_config['burn_in_or_unroll'] in ['burn_in', 'both']:
                    val_metrics[f'val/burn_in/{k}'] = metric_config['module'](burn_in=True)

                    if metric_config.get('framewise', False):
                        val_metrics_framewise[f'val/burn_in/{k}'] = torch.nn.ModuleList([metric_config['module'](burn_in=True) for _ in range(self.num_context_frames)])

        self.val_metrics_framewise = torch.nn.ModuleDict(val_metrics_framewise)

        return torch.nn.ModuleDict(train_metrics), torch.nn.ModuleDict(val_metrics)

    def get_loss(self, batch: Dict[str, Any], outputs: Dict[str, Any], dataloader_idx: int):
        loss_dict = {}

        if self.reconstruct_rgb_loss:
            # L1 loss + LPIPS vs. L2 loss: Same as in slot diffusion
            if 'unroll' in batch:
                pred: torch.Tensor = outputs['rgb_recon']
                target: torch.Tensor = batch['unroll']['video']
                if self.lpips_loss:
                    loss_dict['rgb_recon'] = torch.abs(target - pred).mean()
                elif not self.lpips_loss:
                    loss_dict['rgb_recon'] = F.mse_loss(pred, target)

                if self.lpips_loss is not None:
                    loss_dict['lpips'] = self.lpips_loss(target.flatten(end_dim=1), pred.flatten(end_dim=1)).mean()

                pred: torch.Tensor = outputs['burn_in']['rgb_recon']
                target: torch.Tensor = batch['burn_in']['video']
                if self.lpips_loss:
                    loss_dict['burn_in_rgb_recon'] = torch.abs(target - pred).mean()
                elif not self.lpips_loss:
                    loss_dict['burn_in_rgb_recon'] = F.mse_loss(pred, target)

                if self.lpips_loss is not None:
                    loss_dict['burn_in_lpips'] = self.lpips_loss(target.flatten(end_dim=1), pred.flatten(end_dim=1)).mean()
            else:
                pred: torch.Tensor = outputs['rgb_recon']
                target: torch.Tensor = batch['video']
                if self.lpips_loss:
                    loss_dict['burn_in_rgb_recon'] = torch.abs(target - pred).mean()
                elif not self.lpips_loss:
                    loss_dict['burn_in_rgb_recon'] = F.mse_loss(pred, target)

                if self.lpips_loss is not None:
                    loss_dict['burn_in_lpips'] = self.lpips_loss(target.flatten(end_dim=1), pred.flatten(end_dim=1)).mean()

        return loss_dict

    def training_step(self, out_dict: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0):
        loss_dict = self.get_loss(batch, out_dict, dataloader_idx=dataloader_idx)
        loss = 0
        for v in loss_dict.values():
            loss += v

        outputs = {'loss': loss,
                   'batch_idx': batch_idx, 'loss_dict': loss_dict}

        outputs.update(out_dict)
        self.log_training(outputs, batch=batch)

        return loss

    def validation_step(self, out_dict: Dict[str, Any], batch: Dict[str, Any], batch_idx: int, dataloader_idx: int=0):
        loss_dict = self.get_loss(batch, out_dict, dataloader_idx=dataloader_idx)
        loss: float = 0
        for v in loss_dict.values():
            loss += v

        outputs = {'loss': loss,
                   'batch_idx': batch_idx, 'loss_dict': loss_dict}
        outputs.update(out_dict)

        self.log_validation(outputs, batch=batch, dataloader_idx=dataloader_idx)

        return outputs

    def setup_final_validation(self):
        assert(self.num_val_frames is not None)
        assert(self.metrics_config is not None)
        self.final_validation = True

        # Setup "best" metrics consisting of all val metrics, all metrics with any only_final, and framewise metrics
        best_metrics: Dict[str, torchmetrics.Metric] = {}
        best_metrics_framewise: Dict[str, torch.nn.ModuleList] = {}
        for k, metric_config in self.metrics_config.items():
            # Validation metrics
            if metric_config['train_or_val'] in ['val', 'both']:
                # Unroll metrics
                if metric_config['burn_in_or_unroll'] in ['unroll', 'both']:
                    best_metrics[f'best/unroll/{k}'] = metric_config['module'](burn_in=False)

                    if metric_config.get('framewise', False):
                        best_metrics_framewise[f'best/unroll/{k}'] = torch.nn.ModuleList([metric_config['module'](burn_in=False) for _ in range(self.num_val_frames)])


                # burn_in metrics
                if metric_config['burn_in_or_unroll'] in ['burn_in', 'both']:
                    best_metrics[f'best/burn_in/{k}'] = metric_config['module'](burn_in=True)

                    if metric_config.get('framewise', False):
                        best_metrics_framewise[f'best/burn_in/{k}'] = torch.nn.ModuleList([metric_config['module'](burn_in=True) for _ in range(self.num_context_frames)])

        self.last_val_metrics = torch.nn.ModuleDict(best_metrics)
        self.last_val_metrics_framewise = torch.nn.ModuleDict(best_metrics_framewise)

    def log_validation(self, outputs, batch, dataloader_idx=0):
        super().log_validation(outputs=outputs, batch=batch, dataloader_idx=dataloader_idx)

        if self.final_validation:
            assert(self.last_val_metrics_framewise is not None)

            for metric_name in self.last_val_metrics_framewise.keys():
                for i in range(len(self.last_val_metrics_framewise[metric_name])):
                    self.last_val_metrics_framewise[metric_name][i](**get_output_slice(outputs, i), batch=get_output_slice(batch, i))
                    self.log(f'{metric_name}/{i+1}', self.last_val_metrics_framewise[metric_name][i],
                             on_step=False, on_epoch=True)
        else:
            if self.val_metrics_framewise is not None:
                for metric_name in self.val_metrics_framewise.keys():
                    for i in range(len(self.val_metrics_framewise[metric_name])):
                        self.val_metrics_framewise[metric_name][i](**get_output_slice(outputs, i), batch=get_output_slice(batch, i))
                        self.log(f'{metric_name}/{i+1}', self.val_metrics_framewise[metric_name][i],
                                 on_step=False, on_epoch=True)



class SlotVIP(VPBaseModel):
    def __init__(
            self,
            slot_size: int,
            num_slots: int,
            slot_attention: Union[Dict[str, Any], ml_collections.ConfigDict],
            **kwargs
            ):
        self.slot_size = slot_size
        self.num_slots = num_slots

        super().__init__(**kwargs, max_instances=self.num_slots - 1)

        self.slot_attention: SlotAttentionV2 = init_module(slot_attention, in_features=self.slot_size,
                                          num_slots=self.num_slots, slot_size=self.slot_size)

        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)), requires_grad=True)

    def _get_decoder_out(self, z, **kwargs):
        assert(len(z.shape) == 3)
        assert(self.decoder is not None)

        bs, num_slots, slot_size = z.shape

        z = z.view(bs * num_slots, slot_size)

        out = self.decoder(z)

        C, H, W = out.shape[1:]

        out = out.view((bs, num_slots, C, H, W))
        assert(len(out.shape) == 5)

        return out

    def _encode(self, img:torch.Tensor, batch: Dict[str, Any], dataloader_idx: int, frame_start_idx: int, prev_state: Optional[Dict[str, Any]]=None):
        B, T, C, H, W = img.shape

        encoder_out: torch.Tensor = self._encode_video(img)

        # init slots
        if prev_state is None:
            prev_z = None

            assert(self.init_latents is not None)
            init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]
        else:
            init_latents = None
            prev_z = prev_state['pred_z'][:, -self.num_context_frames:]

        all_pred_z = []
        all_masks = []
        for idx in range(T):
            # init
            if prev_z is None:
                assert(init_latents is not None)
                latents = init_latents  # [B, N, C]
            else:
                # Use up to self.num_burn_in_frames of representations as reference
                x = prev_z[:, -self.num_context_frames:]

                assert(self.predictor is not None)
                latents: torch.Tensor = self.predictor(x)[:, -1]  # [B, N, C]
                assert(len(latents.shape) == 3)

            pred_z, masks = self.slot_attention.forward(encoder_out[:, idx], latents)
            all_pred_z.append(pred_z)
            if masks is not None:
                all_masks.append(masks)

            # next timestep
            prev_z = torch.cat([prev_z, pred_z.unsqueeze(1)], dim=1) if prev_z is not None else pred_z.unsqueeze(1)

        pred_z = torch.stack(all_pred_z, dim=1)
        if all_masks:
            all_masks = torch.stack(all_masks, dim=1)
            all_masks = all_masks.unflatten(-1, (H, W))


        res = {'pred_z': pred_z, 'masks': all_masks}

    def decode(self, d):
        z = d['pred_z'].flatten(0, 1)
        assert(self.decoder is not None)
        out = self._get_decoder_out(z)

        return slot_decode(z=out, 
                    reconstruct_rgb=self.reconstruct_rgb)
