import torch

class Identity(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
