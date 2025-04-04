import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[], activation_fn=nn.ReLU, use_bias=True, activate_last=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.use_bias = use_bias
        self.activate_last = activate_last

        if hidden_sizes:
            layers = []
            prev_hidden_size = input_size
            for i in range(len(hidden_sizes)):
                layers.append(nn.Linear(prev_hidden_size, hidden_sizes[i], bias=self.use_bias))
                layers.append(activation_fn())
                prev_hidden_size = hidden_sizes[i]
            layers.append(nn.Linear(prev_hidden_size, output_size, bias=self.use_bias))
            if self.activate_last:
                layers.append(activation_fn())
        else:
            layers = []
            layers.append(nn.Linear(input_size, output_size, bias=self.use_bias))
            if self.activate_last:
                layers.append(activation_fn())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
