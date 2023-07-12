import numpy as np
from NNmatrix import Neural_Network

# DIMENSION FOR TRANING SET
# X: (parameters, traning set)
# Y: (hypothesis(can be vector), traning set)

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
m = len(X)
Y = np.array([[0, 1, 1, 0,]])
X = np.concatenate((np.ones((4, 1)), X), axis=1).T # 3x4

print(X)
Neural_Network(X, Y, m, [2, 4, 1], 0.75, 0, 700)
