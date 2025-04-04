import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
import torchvision.ops as vops
import torchmetrics

class EngineUnrollMAE(torchmetrics.Metric):
    higher_is_better = False

    def __init__(
            self, **kwargs):
        super().__init__(**kwargs)

        self.add_state('sum_error', default=torch.tensor(0.0, dtype=torch.float32),
                       dist_reduce_fx='sum')
        self.add_state('num_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, batch, engine_unroll_states, **kwargs):
        groundtruth_sym_states = {k: batch['unroll'][k] if 'unroll' in batch else batch[k] for k in engine_unroll_states}
        def get_error(preds, targets):
            # Sum over features and instances
            return torch.sum(torch.abs(preds - targets), dim=[-1, -2] if len(preds.shape) == 4 else [-1])

        # Merge feature dims
        preds = torch.cat([v for v in engine_unroll_states.values()], dim=-1)
        targets = torch.cat([v for v in groundtruth_sym_states.values()], dim=-1)

        if preds.shape[2] - 1 == targets.shape[2]:
            # There are background preds that we need to remove
            preds = preds[:, :, 1:]

        error = get_error(preds=preds, targets=targets)

        self.sum_error += torch.sum(error)
        self.num_samples += error.numel()

    def compute(self):
        return self.sum_error / self.num_samples

class LPIPS(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    higher_is_better = False

    def __init__(
            self, key, burn_in, **kwargs):
        super().__init__(net_type='vgg', normalize=False) #normalize=False -> [-1,1] range

        self.key = key
        self.burn_in = burn_in

    def update(self, batch, **kwargs):
        if self.key == 'flow':
            raise NotImplementedError()
#            preds = kwargs['post_flow_recon_combined']
#            target = kwargs['flow']
        elif self.key == 'rgb':
            if self.burn_in:
                preds = torch.clip(kwargs['burn_in']['rgb_recon'] if 'burn_in' in kwargs else kwargs['rgb_recon'], -1.0, 1.0)
                target = batch['burn_in']['video'] if 'burn_in' in batch else batch['video']
            else:
                preds = torch.clip(kwargs['rgb_recon'], -1.0, 1.0)
                target = batch['unroll']['video']
        else:
            raise NotImplementedError()

        # Merge time and batch dim
        return super().update(img1=target.view((-1,) + target.shape[2:]), img2=preds.view((-1,) + preds.shape[2:]))


class SSIM(torchmetrics.StructuralSimilarityIndexMeasure):
    higher_is_better = True

    def __init__(
            self, key, burn_in, **kwargs):
        super().__init__()

        self.key = key
        self.burn_in = burn_in

    def update(self, batch, **kwargs):
        if self.key == 'flow':
            if self.burn_in:
                preds = kwargs['burn_in']['flow_recon'] if 'burn_in' in kwargs else kwargs['flow_recon']
                target = batch['burn_in']['flow'] if 'burn_in' in batch else batch['flow']
            else:
                preds = kwargs['flow_recon']
                target = batch['unroll']['flow']

        elif self.key == 'rgb':
            if self.burn_in:
                preds = kwargs['burn_in']['rgb_recon'] if 'burn_in' in kwargs else kwargs['rgb_recon']
                target = batch['burn_in']['video'] if 'burn_in' in batch else batch['video']
            else:
                preds = kwargs['rgb_recon']
                target = batch['unroll']['video']
        else:
            raise ValueError()

        # Merge time and batch dim
        return super().update(target=target.view((-1,) + target.shape[2:]), preds=preds.view((-1,) + preds.shape[2:]))


class PSNR(torchmetrics.PeakSignalNoiseRatio):
    higher_is_better = True

    def __init__(
            self, key, burn_in, **kwargs):
        super().__init__()

        self.burn_in = burn_in
        self.key = key

    def update(self, batch, **kwargs):
        if self.key == 'flow':
            if self.burn_in:
                preds = kwargs['burn_in']['flow_recon'] if 'burn_in' in kwargs else kwargs['flow_recon']
                target = batch['burn_in']['flow'] if 'burn_in' in batch else batch['flow']
            else:
                preds = kwargs['flow_recon']
                target = batch['unroll']['flow']

        elif self.key == 'rgb':
            if self.burn_in:
                preds = kwargs['burn_in']['rgb_recon'] if 'burn_in' in kwargs else kwargs['rgb_recon']
                target = batch['burn_in']['video'] if 'burn_in' in batch else batch['video']
            else:
                preds = kwargs['rgb_recon']
                target = batch['unroll']['video']
        else:
            raise ValueError()

        # Merge time and batch dim
        return super().update(target=target.view((-1,) + target.shape[2:]), preds=preds.view((-1,) + preds.shape[2:]))
