"""Map chess moves to model outputs"""
import chess

_MAPPER = {
    # Queenlike moves
    Q_NORTHWEST: [0, 1, 2, 3, 4, 5, 6],
    Q_NORTH: [7, 8, 9, 10, 11, 12, 13],
    Q_NORTHEAST: [14, 15, 16, 17, 18, 19, 20],
    Q_EAST: [21, 22, 23, 24, 25, 26, 27],
    Q_SOUTHEAST: [28, 29, 30, 31, 32, 33, 34],
    Q_SOUTH: [35, 36, 37, 38, 39, 40, 41],
    Q_SOUTHWEST: [42, 43, 44, 45, 46, 47, 48],
    Q_WEST: [49, 50, 51, 52, 53, 54, 55],

    # Knight moves
    K_NORTH_LEFT: 56,
    K_NORTH_RIGHT: 57,
    K_EAST_UP: 58,
    K_EAST_DOWN: 59,
    K_SOUTH_RIGHT: 60,
    K_SOUTH_LEFT: 61,
    K_WEST_DOWN: 62,
    K_WEST_UP: 63,

    # Underpromotions
    UP_KNIGHT: [64, 65, 66],
    UP_BISHOP: [67, 68, 69],
    UP_ROOK: [70, 71, 72]
}

def move_to_policy(board: chess.Board, move: chess.Move):
    """Map a chess move to the corresponding policy index (0-4671)"""
    from_square = move.from_square
    to_square = move.to_square
    piece = board.piece_at(from_square)

    diff = from_square - to_square
    move = None

    # Map underpromotions (promotions to queen are considered queenlike moves)
    if move.promotion and move.promotion != chess.QUEEN:
        # Black promotion, 1st rank
        if to_square < 8:
            direction = diff - 8
        # White promotion, 8th rank
        elif to_square > 55:
            direction = diff + 8
        if move.promotion == chess.KNIGHT:
            move = _MAPPER[UP_KNIGHT[direction + 1]]
        elif move.promotion == chess.BISHOP:
            move = _MAPPER[UP_BISHOP[direction + 1]]
        elif move.promotion == chess.ROOK:
            move = _MAPPER[UP_ROOK[direction + 1]]

    # Map knight moves
    elif piece.piece_type == chess.KNIGHT:
        if diff == -15:
            move = _MAPPER[K_NORTH_LEFT]
        elif diff == -17:
            move = _MAPPER[K_NORTH_RIGHT]
        elif diff == -6:
            move = _MAPPER[K_EAST_UP]
        elif diff == 10:
            move = _MAPPER[K_EAST_DOWN]
        elif diff == 17:
            move = _MAPPER[K_SOUTH_RIGHT]
        elif diff == 15:
            move = _MAPPER[K_SOUTH_LEFT]
        elif diff == 6:
            move = _MAPPER[K_WEST_DOWN]
        elif diff == -10:
            move = _MAPPER[K_WEST_UP]

    # Map queenlike moves
    else:
        # North / south
        if diff % 8 == 0:
            distance = abs(diff) // 8
            if diff > 0:
                move = _MAPPER[Q_NORTH[distance]]
            else:
                move = _MAPPER[Q_SOUTH[distance]]

        # Southwest / northeast
        elif diff % 9 == 0:
            distance = abs(int(diff / 8))
            if diff > 0:
                move = _MAPPER[Q_SOUTHWEST[distance]]
            else:
                move = _MAPPER[Q_NORTHEAST[distance]]

        # East / west
        elif from_square // 8 == to_square // 8:
            distance = abs(diff)
            if diff > 0:
                move = _MAPPER[Q_WEST[distance]]
            else:
                move = _MAPPER[Q_EAST[distance]]

        # Southeast / northwest
        elif diff % 7 == 0:
            distance = abs(int(diff / 8)) + 1
            if diff > 0:
                move = _MAPPER[Q_SOUTHEAST[distance]]
            else:
                move = _MAPPER[Q_NORTHWEST[distance]]

    # Move * square = index
    return (move + 1) * (from_square + 1) - 1
