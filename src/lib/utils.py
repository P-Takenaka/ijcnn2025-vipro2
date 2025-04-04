from typing import Optional, List, Dict, Any, Union
import ml_collections
import numpy as np
import importlib
import copy
import itertools
import torch.nn.functional as F

import torch.nn as nn
import torch

def cumsum(l):
    res = []
    sum_v = 0
    for v in l:
        sum_v += v
        res.append(sum_v)

    return res


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_device(v, device))
        return res
    else:
        raise TypeError("Invalid type for to_device")

# Activation to ensure positive values
def elup1(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0

# Concat a list of dicts or dict of lists recursively
def concat_dict(
        d: Union[Dict[str, List],
                 List[Dict]],
        dim: int):
    result = {}
    if type(d) == dict:
        for k, v in d.items():
            if v is None:
                result[k] = v
                continue

            if type(v) != list:
                    raise ValueError(f"Invalid data type: {type(v)}")

            if type(v[0]) == dict:
                result[k] = concat_dict(v, dim=dim)
            else:
                result[k] = torch.cat(v, dim=dim)
    else:
        assert(type(d) == list)
        # List of dicts to be merged
        result =  {k: torch.cat([sub_d[k] for sub_d in d], dim=dim)  for k in d[0].keys()}

    return result

def detach_dict(d: Dict[str, Union[Dict, torch.Tensor]]):
    result = {}
    for k, v in d.items():
        if v is None:
            result[k] = v
        elif type(v) == dict:
            result[k] = detach_dict(v)
        else:
            assert(type(v) == torch.Tensor)
            result[k] = v.detach()

    return result

def init_module(config: Union[Dict[str, Any], ml_collections.ConfigDict], **kwargs):
    module_str: str = config['module']

    import_str, _, module_name = module_str.rpartition(".")

    py_module = importlib.import_module(import_str)

    if type(config) != dict:
        config = config.to_dict()

    config = copy.deepcopy(config)
    config.pop("module")
    config.update(kwargs)

    return getattr(py_module, module_name)(**config)

def get_module(module_str: str):
    import_str, _, module_name = module_str.rpartition(".")

    py_module = importlib.import_module(import_str)

    return getattr(py_module, module_name)

def freeze_weights(module):
    for p in module.parameters():
        p.requires_grad = False


def get_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.Conv{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

def get_deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.ConvTranspose{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')

def sigmoid_k(x, k):
    return (1. / (1. + torch.exp(-k * x)))

def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')

def flatten_dict_with_prefixes(d, prefix=""):
    flattened = {}
    for key, value in d.items():
        full_key = prefix + "/" + key if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_dict_with_prefixes(value, full_key))
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                # Unwrap the list into separate entries
                for i, v in enumerate(value):
                    list_key = f'{prefix}/{key}_{i}' if prefix else f'{key}_{i}'
                    flattened.update(flatten_dict_with_prefixes(v, list_key))
            else:
                flattened[full_key] = value
        else:
            flattened[full_key] = value

    return flattened

def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    if isinstance(in_size, int):
        return (in_size - 1) * stride - 2 * padding + dilation * (
            kernel_size - 1) + out_padding + 1
    elif isinstance(in_size, (tuple, list)):
        return type(in_size)((deconv_out_shape(s, stride, padding, kernel_size,
                                               out_padding, dilation)
                              for s in in_size))
    else:
        raise TypeError(f'Got invalid type {type(in_size)} for `in_size`')

def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)


def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """ConvTranspose - Norm - Act."""
    deconv = get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)

def get_lr(optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']


def torch_stack(tensor_list, dim):
    if len(tensor_list[0].shape) < dim:
        return torch.stack(tensor_list)
    return torch.stack(tensor_list, dim=dim)


def torch_cat(tensor_list, dim):
    if len(tensor_list[0].shape) <= dim:
        return torch.cat(tensor_list)
    return torch.cat(tensor_list, dim=dim)


def clip_tensor_norm(tensor, norm, dim=-1, eps=1e-6):
    """Clip the norm of tensor along `dim`."""
    assert norm > 0.
    tensor_norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale_factor = norm / (tensor_norm + eps)
    scale_factor = torch.clip(scale_factor, max=1.)
    clip_tensor = tensor * scale_factor
    return clip_tensor


def assert_shape(actual, expected, message=""):
    assert list(actual) == list(expected), \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution, rescaled=False):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    res = torch.cat([grid, 1.0 - grid], dim=-1)

    if rescaled:
        res = res * 2.0 - 1.0

    return res

def build_simple_grid(resolution):
    """return grid with shape [1, H, W, 3]."""
    ranges = [torch.linspace(-1.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)

    return grid


def to_rgb_from_tensor(x):
    """Reverse the Normalize operation in torchvision."""
    return (x * 0.5 + 0.5).clamp(0, 1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution, rescaled=False):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution, rescaled=rescaled))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj

def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    # ViT, BEiT etc. all use zero-init learnable pos enc
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    # in SlotFormer, we find out that sine P.E. is already good enough
    elif 'sin' in pos_enc:  # 'sin', 'sine'
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding

def get_config(config_path: str):
    config_file = importlib.machinery.SourceFileLoader('config_file', config_path).load_module()

    return config_file.get_config()
