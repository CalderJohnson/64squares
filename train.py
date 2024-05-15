"""Training loop"""
import random
import torch
import chess

import hyperparams as hp
from chessutils import get_bitboards
from model import Model
from mcts.tree import MCT

def get_batch(inputs, targets):
    """Select a batch of data"""
    if hp.BATCH_SIZE < len(inputs):
        index = random.randint(0, len(inputs) - hp.BATCH_SIZE - 1)
        tgt = torch.tensor(targets[index:index+hp.BATCH_SIZE]).unsqueeze(1)
        src = torch.tensor(inputs[index:index+hp.BATCH_SIZE])
        return src.to(hp.DEVICE), tgt.to(hp.DEVICE)
    else: # Early in training, many repeating moves, therefore more restricted batches
        tgt = torch.tensor(targets).unsqueeze(1)
        src = torch.tensor(inputs)
        return src.to(hp.DEVICE), tgt.to(hp.DEVICE)

# Define model, optimizer, loss function, and scheduler for learning rate decay
model = Model()
model = model.float().to(hp.DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.LR)
v_loss_fn = torch.nn.MSELoss()          # Mean squared error loss for the value head
p_loss_fn = torch.nn.CrossEntropyLoss() # Categorical cross entropy loss for the policy head
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.EPOCHS / 4, gamma=0.1)

# Training loop
for i in range(hp.EPOCHS):
    # Each epoch, play BATCH_SIZE games with itself, generating BATCH_SIZE Monte Carlo Search Trees
    print("Self play in epoch ", i)
    positions = {} # Training mini batch being generated
    for _ in range(hp.GAMES_PLAYED):
        board = chess.Board()
        while not board.is_game_over():
            position = MCT(board.fen(), model)
            position.run_simulations(hp.DEPTH)
            best_move = position.get_best_move()
            positions[position.root.state] = position.get_vp() # Save estimated value and policy of position
            board.push(best_move)
            print(board)

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
        p_pred, v_pred = model(src)
        p_loss = p_loss_fn(p_pred, tgt[0])
        v_loss = v_loss_fn(v_pred, tgt[1])
        loss = (0.5 * p_loss) + (0.5 * v_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
torch.save(model.state_dict(), "./model.pt")
