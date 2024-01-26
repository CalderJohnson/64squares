"""Model Architecture"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperparams as hp

class Input(nn.Module):
    """Initial input enters a convolution layer"""
    def __init__(self):
        """Initial convolution of the input"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=hp.N_CHANNELS,
            out_channels=hp.N_FILTERS,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.normalize = nn.BatchNorm2d(hp.N_FILTERS)
    
    def forward(self, input):
        """Forward pass"""
        x = self.conv(input)
        x = self.normalize(x)
        return F.relu(x)

class ResidualBlock(nn.Module):
    """Stacked residual blocks"""
    def __init__(self):
        """Convolution followed by normalization and residual connection"""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=hp.N_FILTERS,
            out_channels=hp.N_FILTERS,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.normalize = nn.BatchNorm2d(hp.N_FILTERS)

    def forward(self, input):
        """Forward pass"""
        x = self.conv(input)
        x = self.normalize(x)
        x = x + input # Residual connection added to improve training
        return F.relu(x)
    
class Output(nn.Module):
    """Returns output evaluation of the position"""
    def __init__(self):
        """Final convolution followed by a linear layer"""
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels=hp.N_FILTERS,
                out_channels=hp.N_FILTERS,
                kernel_size=(1, 1),
                stride=(1, 1),
                bias=False,
        )
        self.normalize = nn.BatchNorm2d(hp.N_FILTERS)
        self.compute_output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hp.N_FILTERS * 8 * 8, hp.N_FILTERS * 8 * 8),
            nn.ReLU(),
            nn.Linear(hp.N_FILTERS * 8 * 8, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        """Forward pass"""
        x = self.conv(input)
        x = self.normalize(x)
        size = x.size(0)
        x = x.view(size, -1) # Reshape for linear layer
        return self.compute_output(x)

class Model(nn.Module):
    """Chess engine"""
    def __init__(self):
        """Model architecture"""
        super().__init__()
        self.input = Input()
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(hp.N_RESIDUAL_BLOCKS)])
        self.output = Output()

    def forward(self, input, target=None):
        """Forward pass"""
        x = self.input(input)
        x = self.blocks(x)
        x = self.output(x)
        return x