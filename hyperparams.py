"""Global configs"""
import math
import torch

# General parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training parameters
EPOCHS = 40                         # Number of training epochs
GAMES_PLAYED = 8                    # Number of games per training iteration
BATCH_SIZE = 32                     # Size of training batches
LR = 0.2                            # Initial learning rate

# Preduction parameters
DEPTH = 50                          # Simulations per move
MCTS_COEFFICIENT = math.sqrt(2)     # Trade off between exploration and exploitation in the MCTS

# Model parameters
N_CHANNELS = 19                     # 19 input features
INPUT_SHAPE = (N_CHANNELS, 8, 8)    # Inputs of shape 19 channels, 8x8 boards

N_MOVES = 56 + 8 + 9                # For each square: 56 queenlike moves, 8 knight moves, 9 underpromotions
OUTPUT_SHAPE = (8*8*N_MOVES, 1)     # Output of shape 4672, 1 (policy on all possible moves, value of position)

N_RESIDUAL_BLOCKS = 19              # Number of stacked residual blocks
N_FILTERS = 256                     # Dimensionality of outputs of convolutional layers
