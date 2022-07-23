import numpy as np


def calc_G(X, v):
    U = np.diag(v ** 0.5)
    G = U @ X
    return G
