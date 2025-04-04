from typing import Union
import traceback
import torch
import os
from collections import defaultdict
import torchmetrics
from tqdm.auto import tqdm
import wandb
import logging
import torch.distributed as dist
from torchsummary import summary
import random
import numpy as np

from src.models import BaseModel, BaseDDP
from src.lib import to_device

class Trainer():
    def __init__(self, max_epochs, max_steps, log_every_n_steps, online,
                 early_stopping, early_stopping_patience, monitor_metric, log_dir,
                 max_grad_norm, rank, world_size,
                 monitor_lower_better=True):
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.log_every_n_steps = log_every_n_steps
        self.online = online
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.is_distributed = rank is not None
        self.rank = rank if rank is not None else 0
        self.world_size = world_size


        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.monitor_metric = monitor_metric
        self.monitor_lower_better = monitor_lower_better

        self.best_monitor_metric_value = None
        self.best_monitor_metric_epoch = None

    def _train_step(self, model, batch, batch_idx, optimizer, scheduler):
        out_dict = model(batch)
        loss = model.training_step(batch=batch, out_dict=out_dict, batch_idx=batch_idx)

        optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            model.log('lr', optimizer.param_groups[0]['lr'], on_epoch=True, on_step=False)

        del loss

    def validate(self, model: Union[BaseDDP, BaseModel], data_module, step, move_to_device=True):
        data_module.setup()

        if move_to_device:
            model.to(f'cuda:{self.rank}')
            if self.world_size is not None:
                model = BaseDDP(model, device_ids=[self.rank])

        model.eval()

        val_dataloader = data_module.val_dataloader()
        step_idx = 0
        if self.rank == 0:
            pbar = tqdm(total=len(data_module.val_set) // data_module.batch_size, leave=False)
            pbar.set_description("Validating")
        else:
            pbar = None

        model.on_val_epoch_start()

        for batch_idx, batch in enumerate(val_dataloader):
            batch = to_device(batch, f'cuda:{self.rank}')

            if step == 0 and batch_idx == 0 and self.rank == 0:
                model.on_first_batch(batch)

            with torch.no_grad():
                out_dict = model(batch)
                model.validation_step(out_dict=out_dict, batch=batch, batch_idx=batch_idx)

            step_idx += 1
            if self.rank == 0:
                assert(pbar is not None)
                pbar.update(step_idx)

            model.on_val_step_end()

        metrics = self._log_epoch(model.val_epoch_metrics, step_idx=step)

        return metrics

    def train(self, base_model: BaseModel, data_module, run_name, optimizer_state_dict, scheduler_state_dict, start_step, start_epoch, previous_best_value, previous_best_epoch, previous_random_states, seed):
        data_module.setup()

        base_model.to(self.rank)

        if self.rank == 0:
            summary(base_model)

        if self.world_size is not None:
            model = BaseDDP(base_model, device_ids=[self.rank])
        else:
            model = base_model

        optimizer, scheduler = model.configure_optimizers(parameters=list(model.parameters()), named_parameters=list(model.named_parameters()))

        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        if scheduler_state_dict:
            assert(scheduler is not None)
            scheduler.load_state_dict(scheduler_state_dict)

        self.best_monitor_metric_value = previous_best_value
        self.best_monitor_metric_epoch = previous_best_epoch

        train_dataloader, sampler = data_module.train_dataloader(seed=seed)

        if previous_random_states:
            random.setstate(previous_random_states['python'])
            np.random.set_state(previous_random_states['numpy'])
            torch.set_rng_state(previous_random_states['torch'])
            torch.cuda.set_rng_state_all(previous_random_states['cuda'])

        step_idx = start_step

        # Initial Validation
        if step_idx == 0:
            metrics = self.validate(model, data_module, step=step_idx, move_to_device=False)

        model.train()

        if self.rank == 0:
            pbar = tqdm(range(start_step, self.max_steps))
        else:
            pbar = None

        try:
            for epoch_idx in range(start_epoch, self.max_epochs):
                if self.is_distributed:
                    sampler.set_epoch(epoch_idx)

                model.current_epoch = epoch_idx

                # Training
                for batch_idx, batch in enumerate(train_dataloader):
                    batch = to_device(batch, f'cuda:{self.rank}')

                    if epoch_idx == 0 and batch_idx == 0 and self.rank == 0:
                        model.on_first_batch(batch)

                    self._train_step(model, batch, batch_idx=batch_idx, optimizer=optimizer, scheduler=scheduler)

                    model.on_train_step_end()

                    step_idx += 1
                    if self.rank == 0:
                        assert(pbar is not None)
                        pbar.update(1)

                    del batch

                self._log_epoch(model.train_epoch_metrics, step_idx=step_idx)

                model.on_train_epoch_end()

                optimizer.zero_grad()

                # Validation
                logging.info(f"Epoch {epoch_idx} finished. Validating...")
                metrics = self.validate(model, data_module, step=step_idx, move_to_device=False)

                monitor_metric = metrics[self.monitor_metric]

                if self.rank == 0:
                    checkpoint = {'model_state_dict': model.state_dict() if self.world_size is None else model.module.state_dict(),
                                  'epoch': epoch_idx,
                                  'best_epoch': self.best_monitor_metric_epoch,
                                  'best_value': self.best_monitor_metric_value,
                                  'step': step_idx,
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'random_states': {'python': random.getstate(),
                                                    'numpy': np.random.get_state(),
                                                    'torch': torch.get_rng_state(),
                                                    'cuda': torch.cuda.get_rng_state_all()}}
                    if scheduler is not None:
                        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

                    model.on_save_checkpoint(checkpoint)
                    torch.save(checkpoint, os.path.join(self.log_dir, 'latest_checkpoint.pt'))
                else:
                    checkpoint = None

                if self.best_monitor_metric_value is None:
                    self.best_monitor_metric_value = monitor_metric
                    self.best_monitor_metric_epoch = epoch_idx
                else:
                    is_better = monitor_metric < self.best_monitor_metric_value if self.monitor_lower_better else monitor_metric > self.best_monitor_metric_value

                    if is_better:
                        self.best_monitor_metric_value = monitor_metric
                        self.best_monitor_metric_epoch = epoch_idx

                        # Save checkpoint
                        if self.rank == 0:
                            torch.save(checkpoint, os.path.join(self.log_dir, 'best_checkpoint.pt'))

                    else:
                        if self.early_stopping:
                            assert(self.best_monitor_metric_epoch is not None)
                            if (epoch_idx - self.best_monitor_metric_epoch) > self.early_stopping_patience:
                                # Early stopping triggered
                                if self.rank == 0:
                                    logging.info(f"Early Stopping in epoch {epoch_idx}")
                                break

                if self.rank == 0:
                    wandb.save(f"{self.log_dir}/*", policy='now')

                model.train()


        except Exception as e:
            logging.error("Error during training loop")
            logging.error(traceback.format_exc())

            raise e

        if self.rank == 0:
            logging.info("Training finished. Saving checkpoint")

            wandb.save(f"{str(self.log_dir)}/*", policy='now')

    def _log_epoch(self, metrics, step_idx):
        # Aggregate
        res = {}
        for k, v in metrics.items():
            if isinstance(v, torchmetrics.Metric):
                res[k] = v.compute().detach().cpu()
                v.reset()
            else:
                if self.is_distributed:
                    v = torch.stack(v, dim=0)
                    res[k] = torch.sum(v) / (v.numel() * self.world_size)
                else:
                    res[k] = torch.mean(torch.stack(v, dim=0))

        if self.rank == 0:
            wandb.log(res, step=step_idx)

        return res
