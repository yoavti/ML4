import numpy as np

from functools import partial


def calc_L(X, W):
    return np.apply_along_axis(lambda X_i: np.linalg.norm(X_i - X_i @ W), 1, X)


def optimal_beta_k(L):
    Lm = L.max()
    beta = 2 * Lm ** 2
    k = 1 / beta
    return beta, k


def calc_single_v(L_i, beta, k):
    if L_i <= np.sqrt(beta / (k + 1)):
        return 1
    if L_i <= 1 / np.sqrt(k):
        return 0
    return beta * ((1 / L_i) - k)


def update_v(L, beta, k):
    return np.vectorize(partial(calc_single_v, beta=beta, k=k))(L)


def obj_v(v, L, beta, k):
    return (v * L).sum() + ((beta ** 2) / (v + beta * k)).sum()


def calc_G(X, v):
    U = np.diag(v ** 0.5)
    G = U @ X
    return G
