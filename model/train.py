import random
import torch
import chess

import hyperparams as hp
from chessutils import get_bitboards
from model import Model
from mcts import MCT

def get_batch(inputs, targets):
    """Select a batch of data"""
    index = random.randint(0, len(inputs) - hp.BATCH_SIZE - 1)
    tgt = torch.tensor(targets[index:index+hp.BATCH_SIZE])
    src = torch.tensor(inputs[index:index+hp.BATCH_SIZE])
    return src.to(hp.DEVICE), tgt.to(hp.DEVICE)

# Define model, optimizer, loss function, and scheduler for learning rate decay
model = Model()
model = model.float().to(hp.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.LR)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.EPOCHS / 4, gamma=0.1)

# Training loop
for i in range(hp.EPOCHS):
    # Each epoch, play BATCH_SIZE games with itself, generating BATCH_SIZE Monte Carlo Search Trees
    print("Self play in epoch ", i)
    positions = {} # Training mini batch being generated
    for _ in range(hp.GAMES_PLAYED):
        board = chess.Board()
        while not board.is_game_over():
            position = MCT(board.fen())
            for _ in range(hp.DEPTH):
                position.run_simulation(model)
            best_move = position.get_best_move()
            positions[position.root.state] = position.root.Q / position.root.N # Save estimated value of position
            board.push(best_move)

    # Create a tensor representation of a batch of positions
    inputs = []
    targets = []
    for position, val in positions.items():
        inputs.append(get_bitboards(position).tolist())
        targets.append(val)

    # Once training data is generated, calculate predictions of every position vs. their true value and update model accordingly
    print("Updating model, epoch ", i)
    for _ in range(hp.GAMES_PLAYED):
        src, tgt = get_batch(inputs, targets)
        preds = model(src)
        loss = loss_fn(preds, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model

torch.save(model.state_dict(), "./model.pt")
