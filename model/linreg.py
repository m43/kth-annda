import numpy as np

from utils.util import extend_inputs_with_bias, sse


def linreg(inputs, targets, log=False) -> np.array:
    inputs = extend_inputs_with_bias(inputs)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(inputs.T, inputs)), inputs.T), targets)

    if log:
        outputs = np.dot(inputs, beta)
        print(f"Beta: {beta}")
        print(f"beta.shape:{beta.shape}")
        print(f"outputs:{outputs}")
        sse(outputs, targets)
        print()

    return beta
