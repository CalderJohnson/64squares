"""Monte Carlo Search Tree"""
from math import log
import chess

from model import Model
from chessutils import get_bitboards
import hyperparams as hp

class Node:
    """Node in the search tree"""
    def __init__(self, state, move=None, parent=None):
        """Initialize a new node in the MCT"""
        self.move = move                    # Move played to reach that position
        self.state = state                  # Position represented by the node
        self.turn = chess.Board(state).turn # Current turn
        self.Q = 0                          # Total accumulated reward of the node 
        self.N = 1                          # Number of times this node has been selected during traversal
        self.children = []                  # Children of this node
        self.parent = parent                # Parent of this node
    
    def get_UCB(self):
        """Return UCB of this node"""
        return self.Q / self.N + hp.MCTS_COEFFICIENT * (log(self.parent.N) / self.N)

class MCT:
    """MCT representing the current position"""
    def __init__(self, state: str):
        self.root = Node(state)

    def get_best_move(self):
        """Return best move given current tree"""
        move = self.root.children[0]
        val = move.Q / move.N
        for node in self.root.children:
            if node.Q / node.N > val:
                move = node
                val = node.Q / node.N
        return move.move

    def run_simulation(self, model: Model):
        """Run one simulation"""

        # Select a leaf node
        node = self.selection()

        # Expand
        self.expansion(node)

        # Evaluate
        self.evaluate(node, model)
        
        # Backpropagate
        self.backpropagate(node)

    def selection(self):
        """Select a leaf node using UCB scores"""
        node = self.root
        while node.children:
            next_node = node.children[0]
            ucb = next_node.get_UCB()
            for child in node.children:
                if child.get_UCB() > ucb:
                    next_node = child
                    ucb = next_node.get_UCB()
            node = next_node
        return node
    
    def expansion(self, node: Node):
        """Generate nodes for every possible move"""
        board = chess.Board(node.state)
        for move in board.legal_moves:
            board.push(move)
            node.children.append(Node(board.fen(), move, node))
            board.pop()
    
    def evaluate(self, node: Node, model: Model):
        """Get a score for the current node"""
        board = chess.Board(node.state)
        if board.is_checkmate():
            node.Q = 1
        elif board.is_stalemate():
            node.Q = 0
        else:
            bitboards = get_bitboards(node.state).unsqueeze(0).to(hp.DEVICE) # Add batch dimension
            node.Q = model(bitboards).item()

    def backpropagate(self, node: Node):
        """Backpropagate the value up through the tree"""
        curr = node
        while curr.parent:
            curr.parent.Q += curr.Q
            curr.parent.N += 1
            curr = curr.parent
