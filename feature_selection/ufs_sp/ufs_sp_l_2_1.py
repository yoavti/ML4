import numpy as np
from feature_selection.ufs_sp.utils import convergence, calc_G, l_2_1_norm
from feature_selection.ufs_sp.ufs_sp import create_ufs_sp


def update_W(G, D, alpha):
    return np.linalg.inv(G.T @ G + alpha * D) @ G.T @ G


def update_D(W):
    return np.diag(np.apply_along_axis(lambda W_i: 1 / (2 * np.linalg.norm(W_i)), 0, W))


def obj_W(W, G, alpha):
    return np.linalg.norm(G - G @ W) + alpha * l_2_1_norm(W)


def solve_W(X, v, alpha, max_steps_W=20):
    n, d = X.shape
    G = calc_G(X, v)
    D = np.diag(np.random.randn(d))
    W = np.random.randn(d, d)
    obj = obj_W(W, G, alpha)
    for _ in range(max_steps_W):
        W = update_W(G, D, alpha)
        D = update_D(W)
        new_obj = obj_W(W, G, alpha)
        if convergence(new_obj, obj):
            break
        obj = new_obj
    return W


ufs_sp_l_2_1 = create_ufs_sp(solve_W)
