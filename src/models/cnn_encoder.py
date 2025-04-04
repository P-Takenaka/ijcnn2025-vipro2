from typing import Optional
import torch
from torch import nn

from src.lib import conv_norm_act, SoftPositionEmbed, init_module

class CNNEncoder(torch.nn.Module):
    def __init__(self, norm, ks, channels, resolution, out_size, add_pos_emb, strides, flatten_output, rescaled_pos_emb=False, depth_embedding=False, fixed_depth_embedding=False, norm_depth_embedding=False):
        super().__init__()

        self.resolution = resolution
        self.channels = channels
        self.ks = ks
        self.norm = norm
        self.out_size = out_size
        self.add_pos_emb = add_pos_emb
        self.strides = strides
        self.flatten_output = flatten_output
        self.rescaled_pos_emb = rescaled_pos_emb
        self.depth_embedding = depth_embedding
        self.fixed_depth_embedding = fixed_depth_embedding
        self.norm_depth_embedding = norm_depth_embedding

        if self.depth_embedding:
            assert(self.add_pos_emb)

        # Build Encoder
        # Conv CNN --> PosEnc --> MLP
        enc_layers = len(self.channels) - 1
        self.encoder = nn.Sequential(*[
            conv_norm_act(
                self.channels[i],
                self.channels[i + 1],
                kernel_size=self.ks,
                # 2x downsampling for 128x128 image
                stride=self.strides[i],
                norm=self.norm,
                act='relu' if i != (enc_layers - 1) else '')
            for i in range(enc_layers)
        ])  # relu except for the last layer

        out_resolution = self.resolution
        for i in range(enc_layers):
            out_resolution = (int((out_resolution[0] - self.ks + 2 * (self.ks // 2)) / self.strides[i]) + 1,
                              int((out_resolution[1] - self.ks + 2 * (self.ks // 2)) / self.strides[i]) + 1)

        # Build Encoder related modules
        if self.add_pos_emb:
            if self.depth_embedding:
                self.encoder_pos_embedding = Position3DEmbed(
                    self.channels[-1],
                    out_resolution, fixed_f=self.fixed_depth_embedding,
                    normalize=self.norm_depth_embedding)
            else:
                self.encoder_pos_embedding = SoftPositionEmbed(
                    self.channels[-1],
                    out_resolution, rescaled=self.rescaled_pos_emb)
        else:
            self.encoder_pos_embedding = None

        out_layer_in_size = self.channels[-1] * out_resolution[0] * out_resolution[1] if self.flatten_output else self.channels[-1]

        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(out_layer_in_size),
            nn.Linear(out_layer_in_size, self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size, self.out_size),
        )


    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor]=None):
        out = self.encoder(x)

        if self.encoder_pos_embedding is not None:
            if self.depth_embedding:
                assert(depth is not None)
                out = self.encoder_pos_embedding(out, depth=depth)
            else:
                out = self.encoder_pos_embedding(out)

        if self.flatten_output:
            out = torch.flatten(out, start_dim=1)
        else:
            out = torch.flatten(out, start_dim=2, end_dim=3)
            out = out.permute(0, 2, 1).contiguous()

        out = self.encoder_out_layer(out)

        return out
