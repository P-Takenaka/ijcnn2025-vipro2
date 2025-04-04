import torch
import torch.nn as nn

from src.lib import build_pos_enc

class TransformerDynamicsPredictor(torch.nn.Module):
    def __init__(self,
                 d_model,
                 num_layers,
                 num_heads,
                 ffn_dim,
                 norm_first,
                 num_context_frames,
                 input_dim,
                 output_dim,
        **kwargs):
        super().__init__(**kwargs)

        self.enc_t_pe = build_pos_enc('sin', num_context_frames, d_model=d_model)

        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer, num_layers=num_layers)

        self.linear_in = torch.nn.Linear(input_dim, d_model, bias=False)
        self.linear_out = torch.nn.Linear(d_model, output_dim, bias=False)

    def forward(self, x):
        has_object_dim = len(x.shape) == 4

        assert(self.enc_t_pe is not None)
        if has_object_dim:
            B, T, N = x.shape[:3]
            enc_pe = self.enc_t_pe.unsqueeze(2).\
                        repeat(B, 1, N, 1)
        else:
            B, T = x.shape[:2]
            enc_pe = self.enc_t_pe.repeat(B, 1, 1)


        x = self.linear_in(x)

        x = x + enc_pe[:, -x.shape[1]:]

        return self.linear_out(self.transformer_encoder(x.flatten(1, 2) if has_object_dim else x).reshape(x.shape))  # [B, N, C]

    def reset(self):
        pass
