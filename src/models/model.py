from typing import Optional, List, Dict, Any, Union
import torch.optim as optim
import torch
import logging
from collections import defaultdict
import torchmetrics
from torch.nn.parallel import DistributedDataParallel as DDP
from src.lib import init_module
from torch.nn import functional as F
import torch.distributed as dist

def get_output_slice(out_dict, i):
    result = {}
    for k, v in out_dict.items():
        if k == 'loss' or k == 'batch_idx' or k == 'loss_dict':
            continue

        if type(v) == dict:
            result[k] = get_output_slice(v, i)
        else:
            result[k] = v[:, i:i+1] if v is not None else v

    return result

class BaseDDP(DDP):
    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.module.is_distributed = True

    @property
    def current_epoch(self):
        return self.module.current_epoch

    @property
    def strict_checkpoint_loading(self):
        return self.module.strict_checkpoint_loading

    @current_epoch.setter
    def current_epoch(self, value):
        self.module.current_epoch = value

    @property
    def train_step_metrics(self):
        return self.module._train_step_metrics

    @property
    def train_epoch_metrics(self):
        return self.module._train_epoch_metrics

    @property
    def val_step_metrics(self):
        return self.module._val_step_metrics

    @property
    def val_epoch_metrics(self):
        return self.module._val_epoch_metrics

    def on_first_batch(self, batch, dataloader_idx=0):
        return self.module.on_first_batch(batch, dataloader_idx=dataloader_idx)

    def training_step(self, out_dict, batch, batch_idx, dataloader_idx=0):
        return self.module.training_step(out_dict=out_dict, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def validation_step(self, out_dict, batch, batch_idx, dataloader_idx=0):
        return self.module.validation_step(out_dict=out_dict, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)

    def on_train_epoch_end(self):
        return self.module.on_train_epoch_end()

    def on_train_step_end(self):
        return self.module.on_train_step_end()

    def on_val_epoch_start(self):
        return self.module.on_val_epoch_start()

    def on_val_step_end(self):
        return self.module.on_val_step_end()

    def on_save_checkpoint(self, checkpoint):
        return self.module.on_save_checkpoint(checkpoint=checkpoint)

    def configure_optimizers(self, parameters, named_parameters):
        return self.module.configure_optimizers(parameters, named_parameters)

class BaseModel(torch.nn.Module):
    def __init__(
            self, optimizer=None,
            scheduler=None,
            metrics=None,
            total_steps=None,
            strict_checkpoint_loading=True,
            **kwargs):
        super().__init__()

        self.is_distributed = False
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.total_steps = total_steps
        self.final_validation = False
        self.metrics_config = metrics
        self.strict_checkpoint_loading = strict_checkpoint_loading

        self.train_metrics, self.val_metrics = self.setup_metrics(metrics=metrics)
        self.last_val_metrics: Optional[torch.nn.ModuleDict] = None

        self._train_step_metrics = {}
        self._train_epoch_metrics = defaultdict(list)

        self._val_step_metrics = {}
        self._val_epoch_metrics = defaultdict(list)

        if 'num_val_frames' in kwargs:
            del kwargs['num_val_frames']
        if 'num_slots' in kwargs:
            del kwargs['num_slots']
        if kwargs:
            logging.error(f"The following parameters were not used in the model: {kwargs}")
            raise ValueError()

        self.current_epoch: Optional[int] = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def train_step_metrics(self):
        return self._train_step_metrics

    @property
    def train_epoch_metrics(self):
        return self._train_epoch_metrics

    @property
    def val_step_metrics(self):
        return self._val_step_metrics

    @property
    def val_epoch_metrics(self):
        return self._val_epoch_metrics

    def on_first_batch(self, batch, dataloader_idx=0):
        pass

    @staticmethod
    def load_from_checkpoint(checkpoint_path, config, rank, **kwargs):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % (rank if rank is not None else 0)}

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model = init_module(config, **kwargs)

        model.load_state_dict(
            checkpoint['model_state_dict'], strict=model.strict_checkpoint_loading)

        if rank is not None:
            model = BaseDDP(model)

        optimizer_state_dict = checkpoint['optimizer_state_dict']

        epoch = checkpoint['epoch']
        step = checkpoint['step']
        previous_best_epoch = checkpoint['best_epoch']
        previous_best_value = checkpoint['best_value']
        scheduler_state_dict = checkpoint['scheduler_state_dict'] if 'scheduler_state_dict' in checkpoint else None
        random_states = checkpoint['random_states'] if 'random_states' in checkpoint else None

        return model, optimizer_state_dict, epoch, step, scheduler_state_dict, random_states, previous_best_epoch, previous_best_value


    def setup_metrics(self, metrics: Optional[Dict[str, Any]]):
        raise NotImplementedError()

    def get_loss(self, batch, outputs, dataloader_idx):
        raise NotImplementedError()

    def training_step(self, out_dict, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()

    def validation_step(self, out_dict, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError()

    def on_train_epoch_end(self):
        self._train_epoch_metrics = {}

    def on_train_step_end(self):
        self._train_step_metrics = {}

    def on_val_epoch_start(self):
        self._val_epoch_metrics = {}

    def on_val_step_end(self):
        self._val_step_metrics = {}

    def on_save_checkpoint(self, checkpoint):
        keys = list(checkpoint['model_state_dict'].keys())
        for k in keys:
            if k.startswith('train_metrics') or k.startswith('val_metrics') or k.startswith('last_val_metrics') or k.startswith('lpips_loss'):
                del checkpoint['model_state_dict'][k]

    def get_params_for_optimizer(self, parameters, named_parameters):
        return [{'params': list(filter(lambda p: p.requires_grad,
                                       parameters))}]

    def get_scheduler(self, optimizer):
        if self.scheduler_config:
            scheduler = init_module(self.scheduler_config, optimizer=optimizer)
        else:
            scheduler = None

        return scheduler

    def configure_optimizers(self, parameters, named_parameters):
        assert(self.optimizer_config is not None)

        params = self.get_params_for_optimizer(parameters, named_parameters)

        if 'module' in self.optimizer_config:
            optimizer = init_module(self.optimizer_config, params=params)
        else:
            # Old
            optimizer = optim.Adam(params=params, lr=self.optimizer_config['lr'])

        scheduler = self.get_scheduler(optimizer)

        return optimizer, scheduler

    def setup_final_validation(self):
        raise NotImplementedError()

    def log(self, key, value, on_step, on_epoch):
        assert(int(on_step) + int(on_epoch) < 2)
        assert(on_step == False)

        if on_step:
            if self.training:
                self._train_step_metrics[key] = value.detach().cpu()
            else:
                self._val_step_metrics[key] = value.detach().cpu()
        elif on_epoch:
            if isinstance(value, torchmetrics.Metric):
                if self.training:
                    self._train_epoch_metrics[key] = value
                else:
                    self._val_epoch_metrics[key] = value
            else:
                if self.training:
                    if key not in self._train_epoch_metrics:
                        self._train_epoch_metrics[key] = []
                else:
                    if key not in self._val_epoch_metrics:
                        self._val_epoch_metrics[key] = []

                if self.is_distributed:
                    dist.all_reduce(value)

                if self.training:
                    self._train_epoch_metrics[key].append(value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, device='cpu'))
                else:
                    self._val_epoch_metrics[key].append(value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, device='cpu'))

    def log_training(self, outputs, batch):
        self.log('train/loss_epoch', outputs['loss'], on_step=False,
                 on_epoch=True)

        if 'loss_dict' in outputs and outputs['loss_dict'] is not None:
            for k, v in outputs['loss_dict'].items():
                self.log(f'train/loss/{k}', v, on_epoch=True, on_step=False)

        assert(self.train_metrics is not None)
        for metric_name in self.train_metrics.keys():
            self.train_metrics[metric_name](**outputs, batch=batch)
            self.log(metric_name, self.train_metrics[metric_name],
                     on_step=False, on_epoch=True)

    def log_validation(self, outputs, batch, dataloader_idx=0):
        if self.final_validation:
            assert(self.last_val_metrics is not None)

            for metric_name in self.last_val_metrics.keys():
                self.last_val_metrics[metric_name](**outputs, batch=batch)
                self.log(metric_name, self.last_val_metrics[metric_name],
                         on_step=False, on_epoch=True)

            self.log('best/val/loss', outputs['loss'], on_epoch=True, on_step=False)
            if 'loss_dict' in outputs and outputs['loss_dict'] is not None:
                for k, v in outputs['loss_dict'].items():
                    self.log(f'best/val/loss/{k}', v, on_epoch=True, on_step=False)
        else:
            self.log('val/loss', outputs['loss'], on_epoch=True, on_step=False)
            if 'loss_dict' in outputs and outputs['loss_dict'] is not None:
                for k, v in outputs['loss_dict'].items():
                    self.log(f'val/loss/{k}', v, on_epoch=True, on_step=False)

            for metric_name in self.val_metrics.keys():
                self.val_metrics[metric_name](**outputs, batch=batch)
                self.log(metric_name, self.val_metrics[metric_name],
                         on_step=False, on_epoch=True)
