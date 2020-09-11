import numpy as np
from model.mlp import MLP

if __name__ == '__main__':
    np.random.seed(72)
    anddata = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    xordata = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    p = MLP(anddata[:, 0:2], anddata[:, 2:3], 2)
    p.train_for_niterations(anddata[:, 0:2], anddata[:, 2:3], 0.25, 1001)
    p.confmat(anddata[:, 0:2], anddata[:, 2:3])

    q = MLP(xordata[:, 0:2], xordata[:, 2:3], 2)
    q.train_for_niterations(xordata[:, 0:2], xordata[:, 2:3], 0.25, 5001)
    q.confmat(xordata[:, 0:2], xordata[:, 2:3])
