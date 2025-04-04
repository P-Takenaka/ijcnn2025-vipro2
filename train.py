import os
import sys
import pathlib
import logging
import functools
import shutil
import importlib
from absl import app
from absl import flags
import random

from dotenv import load_dotenv
from ml_collections import config_flags

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

import wandb

from src.lib import flatten_dict_with_prefixes
from src.lib import init_module, get_module
from src.models import BaseModel
from src.lib import Trainer
from src.models.model import BaseDDP

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Config file.")
flags.DEFINE_integer('random_seed', 41, "Random Seed")
flags.DEFINE_string('run_name', 'undefined', "Run Name")
flags.DEFINE_bool('online', False, "Whether to use online services such as mlflow")
flags.DEFINE_bool('debug', False, "Debugging mode")
flags.DEFINE_float('train_fraction', 1.0, "Fraction of training set to use")
flags.DEFINE_integer('batch_size', -1, "Batch Size")
flags.DEFINE_bool('deterministic', True, "")
flags.DEFINE_bool('benchmark', False, "")
flags.DEFINE_string('continue_from', '', "wandb run path to continue")
flags.DEFINE_integer('continue_for', -1, "Optional: Number of steps to continue for")

flags.mark_flags_as_required(["config"])

def init_logger(log_dir, resume=None):
    if resume:
        assert(os.path.exists(log_dir))
    else:
        os.makedirs(log_dir, exist_ok=False)

    # Logging Configuration
    formatter = logging.Formatter(
      fmt='%(asctime)s [%(levelname)s] : %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S')

    logging.getLogger().setLevel(logging.DEBUG)

    if resume:
        fh = logging.FileHandler(os.path.join(log_dir, 'output.log'), mode='a')
    else:
        fh = logging.FileHandler(os.path.join(log_dir, 'output.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger('pytorch_lightning').addHandler(fh)

    if resume:
        logging.info("-------------------------")
        logging.info("Appending to existing log")
        logging.info("-------------------------")

def train(rank, world_size, config, online, log_dir, run_id, debug, batch_size, deterministic, run_name, checkpoint, benchmark):
    if rank is not None:
        setup_distributed(rank=rank, world_size=world_size)

    if rank == 0 or rank is None:
        wandb.init(project=config.experiment_name, id=run_id, resume='must')

    torch.multiprocessing.set_sharing_strategy('file_system')

    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


    # Adjust batch size based on number of available devices
    data_config = config.data.to_dict()
    batch_size = data_config['batch_size'] if batch_size is None else batch_size

    data_config['batch_size'] = (batch_size // world_size) if world_size is not None else batch_size

    early_stopping = config.get('early_stopping', False)
    early_stopping_patience = config.get('early_stopping_patience', 50)
    early_stopping_monitor = config.get('early_stopping_monitor', None)

    if early_stopping and early_stopping_monitor is None:
        raise ValueError("Early stopping is set, but not a metric to monitor!")

    # Setup data
    data_module = init_module(data_config, train_fraction=config.train_fraction, rank=rank, world_size=world_size)

    # Round the number of steps up so that we always do full epochs
    # Infer number of epochs from num_steps
    steps_per_epoch = (len(data_module.train_data) // (data_module.batch_size * (world_size if world_size is not None else 1)))
    num_epochs = config.num_steps // steps_per_epoch
    if config.num_steps % steps_per_epoch:
        num_epochs += 1
        config.num_steps = num_epochs * steps_per_epoch

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    if checkpoint:
        model, optimizer_state_dict, epoch, step, scheduler_state_dict, random_states, previous_best_epoch, previous_best_value = BaseModel.load_from_checkpoint(
            checkpoint, config=config.model, rank=None, total_steps=config.num_steps, num_val_frames=config.data.num_val_frames,
            num_slots=config.num_slots_override if 'num_slots_override' in config else data_module.max_instances + 1)
        assert(not isinstance(model, BaseDDP))

        start_step = step + 1
        start_epoch = epoch + 1

        if 'reset_best_epoch' in config and config.reset_best_epoch:
            previous_best_epoch = None
            previous_best_value = None

        assert(random_states is not None)

    else:

        model = init_module(config.model, total_steps=config.num_steps,
                            num_val_frames=config.data.num_val_frames,
                            num_slots=config.num_slots_override if 'num_slots_override' in config else data_module.max_instances + 1)
        optimizer_state_dict = None
        scheduler_state_dict = None
        previous_best_value = None
        previous_best_epoch = None
        random_states = None
        start_step = 0
        start_epoch = 0

    trainer = Trainer(
        max_epochs=1 if debug else num_epochs, max_steps=config.num_steps,
        log_every_n_steps=50, early_stopping=early_stopping, early_stopping_patience=early_stopping_patience,
        monitor_metric=early_stopping_monitor, log_dir=log_dir, world_size=world_size,
        max_grad_norm=config.max_grad_norm if 'max_grad_norm' in config else None, online=online, rank=rank,
        )

    trainer.train(model, data_module, run_name=run_name,
                  optimizer_state_dict=optimizer_state_dict,
                  scheduler_state_dict=scheduler_state_dict,
                  start_step=start_step, start_epoch=start_epoch,
                  previous_best_value=previous_best_value,
                  previous_best_epoch=previous_best_epoch,
                  previous_random_states=random_states, seed=config.seed)

    if rank == 0 or rank is None:
        wandb.finish()

    if rank is not None:
        cleanup_distributed()

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def train_distributed(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main(argv):
    del argv
    config_param = ''
    for v in sys.argv:
        if v.startswith('--config'):
            config_param = v.split('=')[1]

    online = FLAGS.online
    run_name = FLAGS.run_name
    debug = FLAGS.debug
    batch_size = FLAGS.batch_size
    deterministic = FLAGS.deterministic
    benchmark = FLAGS.benchmark
    continue_from = FLAGS.continue_from
    continue_for = FLAGS.continue_for

    if not config_param and not continue_from:
        raise ValueError("cfg file not specified or in invalid format. it needs to be --config=CONFIG")

    if continue_from:
        run_id = continue_from.split('/')[-1]
        assert(online)
    else:
        run_id = None

    if batch_size == -1:
        batch_size = None

    log_dir = 'logs'

    if continue_from:
        assert(run_id is not None)
        api = wandb.Api()

        run = api.run(continue_from)

        # Download files into log_dir
        for f in run.files():
            if f.name.startswith('logs/'):
                f.download()
    else:
        run = None


    init_logger(log_dir=log_dir, resume=bool(continue_from))

    if not continue_from:
        # Copy config file to log dir
        config_path = os.path.join(os.getcwd(), config_param)
        shutil.copy(config_path, os.path.join(log_dir, "config.py"))

    torch.multiprocessing.set_sharing_strategy('file_system')

    if online:
        wandb.login()

    if continue_from:
        logging.info("Resuming Training")
        checkpoint_path = os.path.join(log_dir, 'latest_checkpoint.pt')
        config_file = os.path.join(log_dir, 'config.py')
        config_file = importlib.machinery.SourceFileLoader('config_file', config_file).load_module()

        config = config_file.get_config()

        with config.unlocked():
            assert(run is not None)
            config.seed = run.config['seed']
            config.train_fraction = run.config['train_fraction']

            if continue_for != -1:
                config.num_steps = config.num_steps + continue_for
    else:
        config = FLAGS.config
        with config.unlocked():
            config.seed = FLAGS.random_seed
            if 'train_fraction' not in config:
                config.train_fraction = FLAGS.train_fraction

        params_to_log = flatten_dict_with_prefixes(config.to_dict())
        params_to_log.update(flatten_dict_with_prefixes(
            vars(get_module(config.data.module).get_dataset_config(os.path.join(
                config.data.data_dir, config.data.name))['args']), 'dataset'))
        params_to_log['deterministic'] = deterministic
        params_to_log['data/batch_size'] = config.data.batch_size if batch_size is None else batch_size
        params_to_log['log_dir'] = log_dir

        # Init wandb just to get a run id
        run = wandb.init(project=config.experiment_name,
                   config=params_to_log, name=run_name, mode='online' if online else 'offline')

        logging.info(f"sys.version: {sys.version}")
        logging.info('Command: python ' + ' '.join(sys.argv))
        logging.info(f"Logging to: {log_dir}")

        assert(run is not None)
        run_id = run.id

        wandb.finish()

        checkpoint_path = None

    fn = functools.partial(train, config=config, online=online, log_dir=log_dir,
                           run_id=run_id, run_name=run_name, debug=debug, batch_size=batch_size, deterministic=deterministic,
                           checkpoint=checkpoint_path, benchmark=benchmark)

    torch.set_printoptions(sci_mode=False)

    if 'num_gpus' in config:
        if config.num_gpus > 1:
            raise NotImplementedError("Random state is currently same for each thread")
            train_distributed(fn, world_size=config.num_gpus)
        else:
            fn(rank=None, world_size=None)
    else:
        # Check how many gpus we have
        if torch.cuda.device_count() > 1:
            raise NotImplementedError("Random state is currently same for each thread")
            train_distributed(fn, world_size=torch.cuda.device_count())
        else:
            fn(rank=None, world_size=None)

    wandb.init(project=config.experiment_name,
               id=run_id, mode='online' if online else 'offline', resume='must')

    # Final validation
    data_module = init_module(config.data.to_dict(), train_fraction=config.train_fraction, rank=None, world_size=None)

    # Load model from checkpoint
    model, _, _, _, _, _, _, _ = BaseModel.load_from_checkpoint(
        os.path.join(log_dir, 'best_checkpoint.pt'),
        config=config.model, rank=None, total_steps=config.num_steps,
        num_val_frames=config.data.num_val_frames,
        num_slots=config.num_slots_override if 'num_slots_override' in config else data_module.max_instances + 1)

    assert(not isinstance(model, BaseDDP))
    model.setup_final_validation()

    trainer = Trainer(
        max_epochs=1, max_steps=config.num_steps,
        log_every_n_steps=50, early_stopping=None, early_stopping_patience=None,
        monitor_metric=None, log_dir=log_dir, world_size=None,
        max_grad_norm=config.max_grad_norm if 'max_grad_norm' in config else None, online=online, rank=None)

    trainer.validate(model, data_module, step=None)

    wandb.finish()

if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.resolve())
    load_dotenv(pathlib.Path(__file__).parent.joinpath('.env').resolve())

    app.run(main)
