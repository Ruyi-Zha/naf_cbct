import torch
import torch.nn as nn


class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        
        x = self.encoder(x, self.bound)
        
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x
    