from typing import Dict, Any, List, Optional, Tuple
import math

import torch
import torch.nn as nn

from src.lib import init_module

class StateInitializer(torch.nn.Module):
    def __init__(
            self,
            state_init: Dict[str, Dict[str, Any]],
            prepend_background: bool,
            state_keys: Optional[List[str]] = None,
            **kwargs):
        super().__init__(**kwargs)

        self.state_init = state_init
        if state_keys is None:
            self.state_keys: List[str] = list(state_init.keys())
        else:
            self.state_keys = state_keys

        self.prepend_background = prepend_background

        state_init_context = {}

        for k, v in self.state_init.items():
            if v['type'] == 'screen_pos' or v['type'] == 'screen_pos_with_depth' or v['type'] == 'boxes':
                state_init_context[k] = {}
                state_init_context[k]['base'] = init_module(v['base_module'])

                state_init_context[k] = nn.ModuleDict(state_init_context[k])
            elif v['type'] == 'screen_pos_with_z':
                state_init_context[k] = {}
                state_init_context[k]['base'] = init_module(v['base_module'])

                state_init_context[k] = nn.ModuleDict(state_init_context[k])
            elif v['type'] == 'sample_norm_learned':
                state_init_context[k] = {}
                if v['per_object']:
                    # Each slot has its own learned weights, breaking slot symmetry
                    state_init_context[k]['mu'] = nn.Parameter(torch.randn(1, v['num_objects'], v['num_features']))
                    state_init_context[k]['logsigma'] = nn.Parameter(torch.zeros(1, v['num_objects'], v['num_features']))
                    torch.nn.init.xavier_uniform_(state_init_context[k]['logsigma'])
                else:
                    state_init_context[k]['mu'] = nn.Parameter(torch.randn(1, 1, v['num_features']))
                    state_init_context[k]['logsigma'] = nn.Parameter(torch.zeros(1, 1, v['num_features']))
                    torch.nn.init.xavier_uniform_(state_init_context[k]['logsigma'])

                state_init_context[k] = nn.ParameterDict(state_init_context[k])
            elif v['type'] == 'sample_uniform_learned':
                state_init_context[k] = {}
                state_init_context[k]['min'] = nn.Parameter(torch.rand(1, 1, v['num_features']) * v['min_scale'])
                state_init_context[k]['range'] = nn.Parameter(torch.rand(1, 1, v['num_features']) * v['range_scale'])

                state_init_context[k] = nn.ParameterDict(state_init_context[k])
            elif v['type'] == 'learned':
                state_init_context[k] = {}
                if v['per_object']:
                    state_init_context[k]['base'] = nn.Parameter(torch.randn(1, v['num_objects'], v['num_features']))
                else:
                    state_init_context[k]['base'] = nn.Parameter(torch.randn(1, 1, v['num_features']))

                state_init_context[k] = nn.ParameterDict(state_init_context[k])

        self.state_init_context = nn.ModuleDict(state_init_context)

    def forward(self, batch: Optional[Dict[str, Any]], batch_size: int, device) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        B = batch_size
        res = {}


        for k, v in self.state_init.items():
            if v['type'] == 'gt':
                assert(batch is not None)
                res[k] = batch[k][:, 0]

            elif v['type'] == 'screen_pos':
                assert(batch is not None)
                screen_pos = batch['screen_positions'][:, 0]
                global_processing = v.get('global_processing', False)

                if global_processing:
                    # Merge object dim into feature dim
                    num_objects = screen_pos.shape[1]
                    screen_pos = screen_pos.view((screen_pos.shape[0], -1))
                else:
                    num_objects = None

                res[k] = self.state_init_context[k]['base'](screen_pos)

                if global_processing:
                    assert(num_objects is not None)
                    # Add object dim again
                    res[k] = res[k].view(res[k].shape[0], num_objects, -1)


            elif v['type'] == 'screen_pos_with_depth':
                assert(batch is not None)
                screen_pos = batch['screen_positions'][:, 0]
                screen_coord_depth = batch['screen_coord_depth'][:, 0]
                pos = torch.cat([screen_pos, screen_coord_depth], dim=-1)
                global_processing = v.get('global_processing', False)

                if global_processing:
                    # Merge object dim into feature dim
                    num_objects = pos.shape[1]
                    pos = pos.view((pos.shape[0], -1))
                else:
                    num_objects = None

                res[k] = self.state_init_context[k]['base'](pos)

                if global_processing:
                    assert(num_objects is not None)
                    # Add object dim again
                    res[k] = res[k].view(res[k].shape[0], num_objects, -1)

        if self.prepend_background:
            for k in res.keys():
                res[k] = torch.cat([torch.zeros_like(res[k][:, :1]), res[k]], dim=1)

        return res
