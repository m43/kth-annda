import numpy as np


def normalize_vectors(data, vectors_in_rows=True):
    if vectors_in_rows:
        return (data.T / np.sqrt(np.sum(data ** 2, axis=1))).T
    else:
        return data / np.sqrt(np.sum(data ** 2, axis=0))