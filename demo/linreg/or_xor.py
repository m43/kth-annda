import numpy as np

from model.linreg import linreg

if __name__ == '__main__':
    # OR and XOR function
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [1]])
    targets_xor = np.array([[0], [1], [1], [0]])

    linreg(inputs, targets)
    linreg(inputs, targets_xor)
