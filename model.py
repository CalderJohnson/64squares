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
        """Two convolutions with a skip connection and normalization"""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=hp.N_FILTERS,
            out_channels=hp.N_FILTERS,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.normalize1 = nn.BatchNorm2d(hp.N_FILTERS)
        self.conv2 = nn.Conv2d(
            in_channels=hp.N_FILTERS,
            out_channels=hp.N_FILTERS,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.normalize2 = nn.BatchNorm2d(hp.N_FILTERS)

    def forward(self, input):
        """Forward pass"""
        x = self.conv1(input)
        x = self.normalize1(x)
        x = F.relu(x)
        x = self.conv2(input)
        x = self.normalize2(x)
        x = x + input # Residual connection added to improve training
        return F.relu(x)

class ValueHead(nn.Module):
    """Returns output evaluation of the position"""
    def __init__(self):
        """Final convolution followed by computation"""
        super().__init__()
        self.conv = nn.Conv2d(
                in_channels=hp.N_FILTERS,
                out_channels=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=False,
        )
        self.normalize = nn.BatchNorm2d(1)
        self.compute_output = nn.Sequential(
            nn.Linear(8 * 8, hp.N_FILTERS),
            nn.ReLU(),
            nn.Linear(hp.N_FILTERS, hp.OUTPUT_SHAPE[1]),
            nn.Tanh(),
        )

    def forward(self, input):
        """Forward pass"""
        x = self.conv(input)
        x = self.normalize(x)
        x = F.relu(x)
        size = x.size(0)
        x = x.view(size, -1) # Reshape for linear layer
        return self.compute_output(x)

class PolicyHead(nn.Module):
    """Returns next move probabilities"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=hp.N_FILTERS,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.normalize = nn.BatchNorm2d(2)
        self.compute_output = nn.Linear(8 * 8 * 2, hp.OUTPUT_SHAPE[0])

    def forward(self, input):
        x = self.conv(input)
        x = self.normalize(x)
        x = F.relu(x)
        size = x.size(0)
        x = x.view(size, -1) # Reshape for linear layer
        return F.sigmoid(self.compute_output(x))

class Model(nn.Module):
    """Chess engine"""
    def __init__(self):
        """Model architecture"""
        super().__init__()
        self.input = Input()
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(hp.N_RESIDUAL_BLOCKS)])
        self.p_head = PolicyHead()
        self.v_head = ValueHead()

    def forward(self, input):
        """Forward pass"""
        x = self.input(input)
        x = self.blocks(x)
        p = self.p_head(x)
        v = self.v_head(x)
        return p, v
