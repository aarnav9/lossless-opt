import numpy as np


def energy(samples):
    return np.asarray([np.sum(row * row) for row in samples])
