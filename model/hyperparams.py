"""Global configs"""
import math
import torch

# General parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training parameters
EPOCHS = 100                        # Number of training epochs
GAMES_PLAYED = 8                    # Number of games per training iteration
BATCH_SIZE = 32                     # Size of training batches
LR = 0.2                            # Initial learning rate

# Preduction parameters
DEPTH = 100                         # Simulations per move
MCTS_COEFFICIENT = math.sqrt(2)     # Trade off between exploration and exploitation in the MCTS (temperature of the model)

# Model constants
N_CHANNELS = 19                     # 19 input features
INPUT_SHAPE = (N_CHANNELS, 8, 8)    # Inputs of shape 19 channels, 8x8 boards
N_RESIDUAL_BLOCKS = 9               # Number of stacked residual blocks
N_FILTERS = 128                     # Dimensionality of outputs of convolutional layers
