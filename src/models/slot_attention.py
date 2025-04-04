from typing import Optional, Union, Dict, List, Any
import ml_collections

import torch

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from src.lib import init_module

def build_grid(resolution):
    ranges = [torch.linspace(-1.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)

    grid = grid.unsqueeze(0)

    return grid

# From https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
class SlotAttentionV2(nn.Module):
    def __init__(self, num_slots, slot_size, num_iterations, mlp_hidden_size, eps = 1e-8, return_masks=False, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.eps = eps
        self.scale = slot_size ** -0.5
        self.return_masks = return_masks

        self.to_q = nn.Linear(slot_size, slot_size)
        self.to_k = nn.Linear(slot_size, slot_size)
        self.to_v = nn.Linear(slot_size, slot_size)

        self.gru = nn.GRUCell(slot_size, slot_size)

        if mlp_hidden_size is None:
            self.mlp = None
            self.norm_pre_ff = None
        else:
            hidden_dim = max(slot_size, mlp_hidden_size)

            self.mlp = nn.Sequential(
                nn.Linear(slot_size, hidden_dim),
                nn.ReLU(inplace = True),
                nn.Linear(hidden_dim, slot_size)
            )
            self.norm_pre_ff = nn.LayerNorm(slot_size)

        self.norm_input  = nn.LayerNorm(slot_size)
        self.norm_slots  = nn.LayerNorm(slot_size)


    def forward(self, inputs: torch.Tensor, slots: torch.Tensor,
                ):
        b, n, d = inputs.shape

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        seg_mask = None
        for attn_iter in range(self.num_iterations):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1)

            # attn_map normalized along slot-dim is treated as seg_mask
            if self.return_masks and (attn_iter == self.num_iterations - 1):
                seg_mask = attn.detach().clone()

            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            if self.mlp is not None:
                slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots, seg_mask
