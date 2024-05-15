import chess

class Node:
    """Node in the search tree, representing a position"""
    def __init__(self, state: str):
        """Initialize a new node in the MCT"""
        self.state = state                  # Position represented by the node
        self.turn = chess.Board(state).turn # Current turn
        self.children = []                  # Children of this node
        self.N = 0                          # Number of times this node has been selected during traversal
        self.Q = 0                          # Total accumulated reward of the node
    
    def is_leaf(self):
        """Return True if the node is a leaf"""
        return not self.children
