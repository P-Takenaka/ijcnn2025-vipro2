from typing import Tuple, Optional
import torch.nn as nn
import torch

from src.lib import init_module

class RNN(nn.Module):
    def __init__(self, encoder, rnn, decoder, joint_reference=False):
        super().__init__()

        self.hidden_state = None
        self.encoder = init_module(encoder)
        self.rnn = init_module(rnn)

        self.decoder = init_module(decoder)

        # RNN takes into account the whole latent state (i.e. of all objects), not just the last dim
        self.joint_reference = joint_reference

    def reset(self, z: Optional[torch.Tensor], use_encoder):
        if z is None:
            self.hidden_state = None
            return

        assert(len(z.shape) == 3)
        if self.joint_reference:
            z = z.contiguous().view(z.shape[0], 1, -1)
        else:
            z = z.contiguous().view(-1, 1, z.shape[-1])

        self.hidden_state = self.encoder(z) if use_encoder else z

        self.hidden_state = self.hidden_state.view(1, self.hidden_state.shape[0], -1)

    def forward(self, x, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert(len(x.shape) == 3)
        B, N, S = x.shape

        if hidden_state is None:
            hidden_state = self.hidden_state
            use_cls_hidden_state = True
        else:
            use_cls_hidden_state = False

        if self.joint_reference:
            x = x.contiguous().view(B, 1, -1)

        z = self.encoder(x)

        self.rnn.flatten_parameters()

        if self.joint_reference:
            out, hidden_state = self.rnn(
                z.view(1, B, -1), hidden_state)
            out = self.decoder(out[0]).view(B, N, -1)
        else:
            out, hidden_state = self.rnn(
                z.contiguous().view(1, -1, z.shape[-1]), hidden_state)
            out = self.decoder(out[0]).view(B, N, -1)

        if use_cls_hidden_state:
            self.hidden_state = hidden_state

        return out, hidden_state
