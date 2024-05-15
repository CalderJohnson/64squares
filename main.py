"""Play interface vs chess engine"""
import torch
import chess

import hyperparams as hp
from chessutils import get_bitboards
from MCTS.mcts import MCT
from model import Model

if __name__ == "__main__":
    model = Model()
    model = model.float().to(hp.DEVICE)
    model.load_state_dict(torch.load("./model.pt", map_location=torch.device("cuda")))

    board = chess.Board()

    while not board.is_game_over():
        movestr = input("Enter a move in standard format: ")
        board.push_san(movestr)

        position = MCT(board.fen())
        for _ in range(hp.DEPTH):
            position.run_simulation(model)
        engine_move = position.get_best_move()
        print(engine_move)
        board.push(engine_move)
        print(board)

    print("Good game!")
