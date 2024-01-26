"""Contains chess board representations and move calculations"""
import torch
import numpy as np
import chess
import hyperparams as hp

def get_bitboards(fen):
    # Matrix for current turn
    board = chess.Board(fen)
    turn = np.ones((8, 8)) if board.turn else np.zeros((8, 8))

    # 4 matrices for castling rights
    castling = np.asarray([
        np.ones((8, 8)) if board.has_queenside_castling_rights(chess.WHITE)
            else np.zeros((8, 8)),
        np.ones((8, 8)) if board.has_kingside_castling_rights(chess.WHITE)
            else np.zeros((8, 8)),
        np.ones((8, 8)) if board.has_queenside_castling_rights(chess.BLACK)
            else np.zeros((8, 8)),
        np.ones((8, 8)) if board.has_kingside_castling_rights(chess.BLACK)
            else np.zeros((8, 8)),
    ])

    # Repetition counter
    counter = np.ones((8, 8)) if board.can_claim_fifty_moves() else np.zeros((8, 8))

    # 12 matrices representing pieces of both players
    arrays = []
    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            array = np.zeros((8, 8))
            for index in list(board.pieces(piece_type, color)):
                # row calculation: 7 - index/8 because we want to count from bottom left, not top left
                array[7 - int(index/8)][index % 8] = True
            arrays.append(array)
    arrays = np.asarray(arrays)

    # Matrix for en passant squares
    en_passant = np.zeros((8, 8))
    if board.has_legal_en_passant():
        en_passant[7 - int(board.ep_square/8)][board.ep_square % 8] = True

    # Input shape is 8x8x19 (1216)
    bitboards = torch.from_numpy(np.array([turn, *castling, counter, *arrays, en_passant]).reshape(hp.INPUT_SHAPE)).float()
    return bitboards
