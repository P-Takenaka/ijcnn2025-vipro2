import os
import pickle
import logging
import random

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T

class BaseDataModule():
    def __init__(self, batch_size, num_train_frames,
                 num_val_frames, target_size, train_fraction,
                 rank, world_size,
                 num_burn_in_frames=None, num_workers=None):
        super().__init__()

        self.batch_size = batch_size
        self.num_burn_in_frames = num_burn_in_frames
        self.num_train_frames = num_train_frames
        self.num_val_frames = num_val_frames
        self.target_size = target_size
        self.train_fraction = train_fraction
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        assert(self.train_fraction > 0 and self.train_fraction <= 1.0)

        self.train_set = None
        self.val_set = None
        self.test_set = None

    @staticmethod
    def get_dataset_config(data_directory):
        raise NotImplementedError()

    def setup(self, stage = None):
        if self.train_set is None:
            self.train_set = self.create_dataset('train')
            self.val_set = self.create_dataset('val')
            self.test_set = self.create_dataset('test')

    def create_dataset(self, split: str):
        raise NotImplementedError()

    def train_dataloader(self, seed):
        assert(self.train_set)
        if self.rank is None:
            sampler = None
        else:
            sampler = DistributedSampler(self.train_set, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True,
                                         seed=seed)

        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.batch_size * 2 if self.num_workers is None else self.num_workers, sampler=sampler,
            shuffle=sampler is None, drop_last=True, persistent_workers=True,
            pin_memory=True), sampler

    def val_dataloader(self):
        assert(self.val_set)
        if self.rank is None:
            sampler = None
        else:
            sampler = DistributedSampler(self.val_set, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)

        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            num_workers=self.batch_size * 2  if self.num_workers is None else self.num_workers, sampler=sampler,
            shuffle=False,
            drop_last=False, persistent_workers=True,
            pin_memory=True)

class VPDataset(Dataset):
    def __init__(
             self,
             samples,
             num_frames: int,
             max_instances: int,
             load_keys,
             target_size,
             num_burn_in_frames,
             allow_variable_number_of_objects,
             camera_relative_state: bool,
    ):
        super().__init__()

        self.samples = samples

        self.num_frames = num_frames
        self.max_instances = max_instances
        self.load_keys = load_keys
        self.target_size = target_size # H, W
        self.num_burn_in_frames = num_burn_in_frames
        self.allow_variable_number_of_objects = allow_variable_number_of_objects
        self.camera_relative_state = camera_relative_state

        self.video_transforms = T.Compose([T.ToTensor(), T.Resize(
            self.target_size, interpolation=T.InterpolationMode.BILINEAR, antialias=False)])
        self.depth_transforms = T.Compose([T.Resize(
            self.target_size, interpolation=T.InterpolationMode.BILINEAR, antialias=False)])

    def __getitem__(self, idx):
        sample = self.samples[idx]

        result = {}

        try:
            # Load additional data from metadata
            with open(os.path.join(sample, 'metadata.pkl'), 'rb') as f:
                tmp = pickle.load(f)
            metadata = tmp['metadata']
            data_ranges = tmp['data_ranges']

            num_instances = min(metadata['num_instances'], self.max_instances)
            num_frames = metadata['num_frames']
            resolution = tuple(metadata['resolution'])
            instances = tmp['instances'][:self.max_instances] if tmp['instances'] else None
            camera = tmp['camera']

            if not (resolution[0] >= self.target_size[0] and resolution[1] >= self.target_size[1]):
                raise ValueError("Resolutions do not match")

            if not (num_frames >= self.num_frames):
                raise ValueError("Number of frames do not match")

            start_idx = 0
            end_idx = start_idx + self.num_frames

            # Video
            if 'video' in self.load_keys:
                video = np.load(os.path.join(sample, 'video.npz'))['video'][start_idx:end_idx, ..., :3]
                video = torch.stack([self.video_transforms(v) for v in video], dim=0)

                result['video'] = video

            # Depth
            if 'depth' in self.load_keys:
                depth_range = (data_ranges['depth']['min'], data_ranges['depth']['max'])
                depth = np.load(os.path.join(sample, 'depth.npz'))['depth'][start_idx:end_idx]

                if depth.dtype == np.uint16:
                    # Old style depth, only allowed if max depth < 10000 since otherwise too much precision is lost
                    if depth_range[1] > 10000:
                        raise ValueError(f"uint16 depth is not allowed if there is no background (max={depth_range[1]})")

                    # Undo float to uint16 scaling
                    depth = depth.astype(np.float32) / 65535.

                    depth = depth * (depth_range[1] - depth_range[0]) + depth_range[0]

                depth = self.depth_transforms(torch.as_tensor(depth).permute(0, 3, 1, 2))

                result['depth'] = depth

            camera_pos = None
            if 'physics' in self.load_keys or 'camera' in self.load_keys:
                camera_pos = torch.as_tensor(camera['positions'], dtype=torch.float32)[start_idx:end_idx]
                result['camera_positions'] = camera_pos
                result['camera_positions'][...,0] = -result['camera_positions'][...,0]

                camera_quat = torch.as_tensor(camera['quaternions'])[start_idx:end_idx]
                result['camera_quaternions'] = camera_quat

                if 'thetas' in camera:
                    camera_thetas = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in camera['thetas']], dim=0)[1+start_idx:end_idx+1]
                    if camera_thetas.shape[1] != 2:
                        raise ValueError("Invalid camera thetas shape")

                    camera_dthetas = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in camera['dthetas']], dim=0)[1+start_idx:end_idx+1]

                    result['camera_thetas'] = camera_thetas
                    result['camera_dthetas'] = camera_dthetas

            if 'physics' in self.load_keys:
                if instances is None:
                    raise ValueError("instances is None")

                # Mass
                mass = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
                mass[:, :num_instances] = torch.stack([torch.tile(torch.as_tensor(v['mass']), (self.num_frames, 1)) for v in instances], dim=1)
                result['mass'] = mass

                # Velocities
                velocities = torch.zeros((self.num_frames, self.max_instances, 3), dtype=torch.float32)
                velocities[:, :num_instances] = torch.stack([
                    torch.as_tensor(v['velocities'][start_idx:end_idx]) for v in instances], dim=1)
                result['velocities'] = velocities

                # World Positions
                # Orbits:
                # To the right -> more negative
                # To the bottom -> more positive
                world_positions = torch.zeros((self.num_frames, self.max_instances, 3), dtype=torch.float32)
                world_positions[:, :num_instances] = torch.stack([
                    torch.as_tensor(v['positions'][start_idx:end_idx]) for v in instances], dim=1)
                result['positions'] = world_positions

                # Normal: Right-> Negative; Inverted: Right -> Positive
                result['positions'][...,0] = -result['positions'][...,0]
                result['velocities'][...,0] = -result['velocities'][...,0]

                if self.camera_relative_state:
                    result['positions'][... , 2] = result['camera_positions'][:, None, 2] - result['positions'][... ,2]

                # Screen positions [0, 1], but may contain values outside of this if out of screen
                screen_positions = torch.zeros((self.num_frames, self.max_instances, 2), dtype=torch.float32)
                screen_positions[:, :num_instances] = torch.stack([
                    torch.as_tensor(v['image_positions'][start_idx:end_idx]) for v in instances], dim=1)
                result['screen_positions'] = screen_positions

                if 'screen_coord_depth' in self.load_keys:
                    if 'depth' not in result:
                        raise ValueError("Depth not available")

                    if self.target_size[0] != self.target_size[1]:
                        raise ValueError("Invalid target size")

                    # Get depth at screen coordinates
                    # screen_coordinates = (T, N, 2)
                    # depth = (T, 1, H, W)
                    screen_coordinates = (result['screen_positions'].numpy().clip(0.0, 1.0) * (self.target_size[0] - 1)).astype(int)
                    screen_coord_depths = []
                    for t in range(screen_coordinates.shape[0]):
                        object_depths = []
                        for n in range(screen_coordinates.shape[1]):
                            object_depths.append(result['depth'][t, :, screen_coordinates[t, n, 1], screen_coordinates[t, n, 0]])
                        screen_coord_depths.append(np.stack(object_depths, axis=0))
                    result['screen_coord_depth'] = torch.as_tensor(np.stack(screen_coord_depths, axis=0))

                # Make it so that the behaviour is the same as for the world pos (right->negative,bottom->positive)
                # and scale it to [-1,1]
                result['screen_positions'] = result['screen_positions'] * 2.0 - 1.0

                # If applicable: theta and dtheta
                if 'theta' in instances[0]:
                    if num_instances != 2:
                        raise ValueError(f"Invalid number of instances: {num_instances}")

                    theta = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
                    theta[:, :num_instances] = torch.stack(
                        [torch.unsqueeze(torch.as_tensor(v['theta'][start_idx:end_idx]), dim=-1) for v in instances], dim=1)

                    result['theta'] = theta

                    dtheta = torch.zeros((self.num_frames, self.max_instances, 1), dtype=torch.float32)
                    dtheta[:, :num_instances] = torch.stack(
                        [torch.unsqueeze(torch.as_tensor(v['dtheta'][start_idx:end_idx]), dim=-1) for v in instances], dim=1)
                    result['dtheta'] = dtheta

            # We do not allow variable number of objects as that is not our goal
            if not self.allow_variable_number_of_objects and num_instances != self.max_instances:
                raise ValueError(f"Num instances {num_instances} does not match max_instances {self.max_instances}")

            result['num_instances'] = torch.tile(torch.as_tensor(num_instances).unsqueeze(0), (self.num_frames, 1))

            if self.num_burn_in_frames is not None:
                # Split all entries into burn in and unroll parts
                burn_in_dict = {}
                unroll_dict = {}

                for k, v in result.items():
                    if len(v.shape) < 1 or v.shape[0] != self.num_frames:
                        burn_in_dict[k] = v
                        unroll_dict[k] = v
                    else:
                        burn_in_dict[k] = torch.as_tensor(v[:self.num_burn_in_frames]) if v is not None else v
                        unroll_dict[k] = torch.as_tensor(v[self.num_burn_in_frames:]) if v is not None else v

                return {'burn_in': burn_in_dict, 'unroll': unroll_dict}
            else:
                return result
        except Exception as e:
            logging.info(f"Exception when loading data sample {sample}")
            raise e

    def __len__(self):
        return len(self.samples)

class VPDataModule(BaseDataModule):
    def __init__(self, data_dir, name,
                 load_keys,
                 val_name=None, max_instances=None,
                 test_settings=None,
                 allow_variable_number_of_objects=True, camera_relative_state=False, **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.base_dir = os.path.join(data_dir, name)
        self.load_keys = load_keys
        self.val_name = val_name
        self.test_settings_config = test_settings
        self.test_setting_datasets = None
        self.use_test_settings = False
        self.allow_variable_number_of_objects = allow_variable_number_of_objects
        self.camera_relative_state = camera_relative_state

        self.data_metainfo = self.get_dataset_config(self.base_dir)

        self.max_instances = vars(self.data_metainfo['args']).get('max_num_objects')
        if max_instances is not None:
            self.max_instances = max_instances

        self.train_data = [os.path.join(self.base_dir, str(i)) for i in self.data_metainfo['train_indices']]
        if self.train_fraction != 1.0:
            # Only take a fraction of training samples
            self.train_data = self.train_data[:int(len(self.train_data) * self.train_fraction)]
        if self.val_name is None:
            self.val_data = [os.path.join(self.base_dir, str(i)) for i in self.data_metainfo['val_indices']]
        else:
            with open(os.path.join(data_dir, self.val_name, 'metainfo.pkl'), 'rb') as f:
                val_data_metainfo = pickle.load(f)
            self.val_data = [os.path.join(data_dir, self.val_name, str(i)) for i in val_data_metainfo['val_indices']]

    @staticmethod
    def get_dataset_config(data_directory):
        with open(os.path.join(data_directory, 'metainfo.pkl'), 'rb') as f:
            return pickle.load(f)

    def create_dataset(self, split: str):
        assert(self.train_data is not None)
        assert(self.val_data is not None)

        if split == 'train':
            return VPDataset(
                samples=self.train_data,
                num_frames=self.num_burn_in_frames + self.num_train_frames if self.num_burn_in_frames else self.num_train_frames,
                max_instances=self.max_instances,
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
                allow_variable_number_of_objects=self.allow_variable_number_of_objects,
                camera_relative_state=self.camera_relative_state

            )
        elif split == 'val':
            return VPDataset(
                samples=self.val_data,
                num_frames=self.num_burn_in_frames + self.num_val_frames if self.num_burn_in_frames else self.num_val_frames,
                max_instances=self.max_instances,
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
                allow_variable_number_of_objects=self.allow_variable_number_of_objects,
                camera_relative_state=self.camera_relative_state
            )

        elif split == 'test':
            return VPDataset(
                samples=self.val_data,
                num_frames=self.num_burn_in_frames + self.num_val_frames if self.num_burn_in_frames else self.num_val_frames,
                max_instances=self.max_instances,
                load_keys=self.load_keys,
                target_size=self.target_size,
                num_burn_in_frames=self.num_burn_in_frames,
                allow_variable_number_of_objects=self.allow_variable_number_of_objects,
                camera_relative_state=self.camera_relative_state
            )
        else:
            raise ValueError("Invalid split name!")

