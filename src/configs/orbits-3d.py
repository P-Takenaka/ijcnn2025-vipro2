import ml_collections
import os
import functools
import pickle
import torch

from src.lib import LPIPS, SSIM, PSNR

def get_config():
  config = ml_collections.ConfigDict()

  config.experiment_name = 'orbits-3d'

  config.num_steps = 500000
  config.max_grad_norm = 0.05
  config.early_stopping = False
  config.early_stopping_monitor = 'val/loss/rgb_recon'

  resolution = (64, 64)

  num_frames = 6
  num_context_frames = 6
  num_train_frames = 6
  num_val_frames = 24

  z_a_size = 64
  z_b_size = 64
  z_c_size = 64

  slot_size = z_a_size + z_b_size + z_c_size

  config.data = ml_collections.ConfigDict({
        "module": "src.datasets.VPDataModule",
        "data_dir": os.environ["DATA_DIR"],
        "name": "orbits-3d",
        "batch_size": 16, # Total
        "num_workers": 8,
        "num_train_frames": num_train_frames,
        "num_val_frames": num_val_frames,
        "num_burn_in_frames": num_context_frames,
        "load_keys": ["video", "physics", "depth", "screen_coord_depth"],
        "target_size": resolution,
        "camera_relative_state": True,
        "allow_variable_number_of_objects": False,
  })

  with open(os.path.join(config.data.data_dir, config.data.name, '0', 'metadata.pkl'), 'rb') as f:
      sample_metadata = pickle.load(f)

  with open(os.path.join(config.data.data_dir, config.data.name, 'metainfo.pkl'), 'rb') as f:
      general_metadata = pickle.load(f)

  config.model = ml_collections.ConfigDict({
        "module": "src.models.ProcSlotVIP",
        "strict_checkpoint_loading": False,
        "use_lpips_loss": True,
        "slot_size": slot_size,
        'state_fusion': {
          'type': 'learned',
        },
        "gain_predictor": ml_collections.ConfigDict({
                "module": "src.models.RNN",
                "joint_reference": True,
                "encoder": ml_collections.ConfigDict({
                        "module": "src.models.MLP",
                        "input_size": z_a_size * 4 * 2,
                        "hidden_sizes": (z_a_size,),
                        "output_size": z_a_size,
                }),
                "rnn": ml_collections.ConfigDict({
                        "module": "torch.nn.GRU",
                        "input_size": z_a_size,
                        "hidden_size": z_a_size,
                }),
                "decoder": ml_collections.ConfigDict({
                        "module": "src.models.MLP",
                        "input_size": z_a_size,
                        "hidden_sizes": (z_a_size,),
                        "output_size": 6 * 4,
                }),
          }),
        "state_observation_alignment_factor": 1.0,
        "state_autoencoder_alignment_factor": 1.0,
        "burn_in_state_supervision_factor": 0.0,
        "num_context_frames": num_context_frames,
        "max_concurrent_frames": num_context_frames,
        "state_initializer": ml_collections.ConfigDict({
                "module": "src.models.StateInitializer",
                "prepend_background": False,
                "state_init": {
                  "positions": {
                        'type': 'screen_pos_with_depth',
                        'base_module': ml_collections.ConfigDict({
                          "module": "src.models.MLP",
                          "hidden_sizes": (16,),
                          "activation_fn": torch.nn.Tanh,
                          "input_size": 3,
                          "output_size": 3,
                        }),
                        },
                  "velocities": {
                        'type': 'gt',
                  }
                }
            }),
        "predictor": ml_collections.ConfigDict({
          "module": "src.models.ProcModule",
          "z_a_size": z_a_size,
          "z_b_size": z_b_size,
          "z_c_size": z_c_size,
          "D_module": ml_collections.ConfigDict({
                "module": "src.models.TransformerDynamicsPredictor",
                "d_model": slot_size,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 512,
                "norm_first": True,
                "num_context_frames": num_frames,
                "input_dim": slot_size,
                "output_dim": z_b_size,
              }),
          "F_module": ml_collections.ConfigDict({
                  "module": "src.models.OrbitsPhysicsEngine",
                  "G": sample_metadata['flags']['gravitational_constant'],
                  "step_rate": sample_metadata['flags']['step_rate'],
                  "frame_rate": sample_metadata['flags']['frame_rate'],
                  "obj_mass": sample_metadata['flags']['fixed_mass'],
                  "only_2d": general_metadata['args'].only_2d,
                  "prepend_background": False,
                  "ghost_z": sample_metadata['flags']['ghost_z'],
                  "ghost_inverse_force": sample_metadata['flags']['ghost_inverse_force'],
                  "camera_relative_state": True,
                  "camera_pos": [0., 0., 30.],
                }),
            "F_in": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (z_a_size,),
            }),
            "F_out": ml_collections.ConfigDict({
                "module": "src.models.MLP",
                "hidden_sizes": (z_a_size,),
            }),
          }),
        "encoder": ml_collections.ConfigDict({
          "module": "src.models.CNNEncoder",
          "norm": '',
          "ks": 5,
          "channels": (3, 32, 32, 32, 32),
          "resolution": resolution,
          "out_size": slot_size,
          "add_pos_emb": True,
          "strides": (1, 1, 1, 1),
          "flatten_output": False
        }),
        "decoder": ml_collections.ConfigDict({
          "module": "src.models.SpatialBroadcastDecoder",
          "add_pos_emb": True,
          "input_size": slot_size - 32,
          "resolution": (8, 8),
          "channels": (slot_size - 32, 32, 32, 32, 32),
          "out_channels": 4,
          "ks": 5,
          "norm": '',
          "strides": (2, 2, 2, 1),
        }),
        "slot_attention": ml_collections.ConfigDict({
          "module": "src.models.SlotAttentionV2",
          "num_iterations": 5,
          "mlp_hidden_size": 128,
        }),
        "reconstruct_rgb": True,
        "optimizer": ml_collections.ConfigDict({
                "lr": 2e-4,
        }),
        "metrics": {
                'ssim_rgb': {
                        'module': functools.partial(SSIM, key='rgb'),
                        'train_or_val': 'both',
                        'framewise':  True,
                        'burn_in_or_unroll': 'both',
                        'only_final': False,
                },
                'psnr_rgb': {
                        'module': functools.partial(PSNR, key='rgb'),
                        'train_or_val': 'both',
                        'framewise':  True,
                        'burn_in_or_unroll': 'both',
                        'only_final': False,
                },
                'lpips_rgb': {
                        'module': functools.partial(LPIPS, key='rgb'),
                        'train_or_val': 'val',
                        'framewise':  True,
                        'burn_in_or_unroll': 'both',
                        'only_final': True,
                },
        },
  })

  return config
