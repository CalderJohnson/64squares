from math import log, sqrt
import chess

import hyperparams as hp

class Edge:
    """Edge in the search tree, representing a move"""
    def __init__(self, input_node, output_node, move: chess.Move, policy: float):
        """Initialize a new edge in the MCT"""
        self.move = move
        self.input_node = input_node
        self.output_node = output_node

        self.N = 0 # Total number of times this edge has been traversed
        self.W = 0 # Total accumulated reward of this edge
        self.P = policy # The value of the edge as determined by the policy network

    def get_ucb(self):
        """
        Return UCB of this edge, average reward of the edge + exploration term
        """
        return (self.W / self.N + 1) + hp.MCTS_COEFFICIENT * self.P * (sqrt(self.input_node.N) / 1 + self.N)
