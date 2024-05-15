"""Monte Carlo Tree Search"""
from math import log
import torch
import chess

from model import Model
from chessutils import get_bitboards
import hyperparams as hp

from mcts.node import Node
from mcts.edge import Edge
from mcts.mapper import move_to_policy

class MCT:
    """MCT representing the current position"""
    def __init__(self, state: str, model: Model):
        self.root = Node(state)         # Current position represented by the MCT
        self.game_path = []             # Path of nodes traversed during the simulation
        self.model = model              # Model used to evaluate positions

    def get_best_move(self):
        """Return best move given current tree"""
        move = self.root.children[0]
        val = move.Q / move.N
        for node in self.root.children:
            if node.Q / node.N > val:
                move = node
                val = node.Q / node.N
        return move.move

    def get_vp(self):
        """Return the value and policy of the root node for training purposes"""
        board = chess.Board(self.root.state)
        policy_vec = [0] * hp.OUTPUT_SHAPE[0]
        probabilities = torch.nn.functional.softmax(torch.tensor([edge.get_ucb() for edge in self.root.children]), dim=0)
        for i, edge in enumerate(self.root.children):
            policy_vec[move_to_policy(board, edge.move)] = probabilities[i]
        return (self.root.Q / self.root.N), policy_vec

    def run_simulations(self, n: int):
        """Run n simulations"""
        for _ in range(n):
            node = self.selection()  # Select a leaf node
            self.expansion(node)     # Expand the node
            self.backpropagate(node) # Backpropagate the value up through the tree

    def selection(self):
        """Select a leaf node using UCB scores"""
        node = self.root
        self.game_path = []
        while not node.is_leaf():
            best_edge = None
            best_score = float('-inf')
            for edge in node.children:
                if edge.get_ucb() > best_score:
                    best_edge = edge
                    best_score = edge.get_ucb()
            node = best_edge.output_node
            self.game_path.append(best_edge)
        return node
    
    def expansion(self, node: Node):
        """Evaluate the leaf and generate weighted edges for every possible move"""
        board = chess.Board(node.state)
        bitboards = get_bitboards(board)
        p, v = self.model(bitboards) # Evaluate the position
        node.value = v               # Store the estimated value of the position

        valid_moves = board.generate_legal_moves() # For each legal move, create a new edge
        for move in valid_moves:
            policy = p[move_to_policy(board, move)]
            board.push(move)
            new_node = Node(board.fen())
            edge = Edge(node, new_node, move, policy)
            node.children.append(edge)
            board.pop()

    def backpropagate(self, value):
        """Backpropagate the value up through the edges of the tree"""
        for edge in self.game_path:
            edge.input_node.N += 1
            edge.N += 1
            edge.W += value

    def debug_tree(self, node=None, level=0):
        """FIXME: Print the current simulation"""
        if node is None:
            return
        for _ in range(level):
            print("  ", end="")
        print("Q: ", node.Q, " N: ", node.N, " Q / N: ", node.Q / node.N)
        for child in node.children:
            self.debug_tree(child, level + 1)
