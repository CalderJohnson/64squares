Chess AI, trained with deep reinforcement learning

This chess engine trains itself using a modified version of the Monte Carlo Search Tree

Our MCTS works by: 
1. Selection: Traverse the tree using the computed values of the nodes picked from a normal distribution until a leaf is reached.
2. Expansion: Expand the leaf node by creating a child for every possible action.
3. Evaluation: The value of the leaf node is estimated using our model.
4. Backpropagation: Backpropagate the estimation to the root node.

This is repeated X times (X being the depth setting).

After a round of self play, the model goes back and reevaluates the position, loss being generated based off the model's prediction vs the true value of the position.

The model has the following architecture:

Input layer: 19 8x8 bitboards representing game state.

Hidden layers: 1 convolutional layer, followed by 9 residual blocks with skip connections.

Output: The value of of the given board (scalar, -1 to 1), and a weighted vector of possible moves.
