import numpy as np
from feature_selection.ufs_sp.utils import calc_G, convergence, calc_L, optimal_beta_k, obj_v, update_v, l_2_1_norm_vec


def update_W(X, v, alpha):
    G = calc_G(X, v)
    return np.linalg.inv(G.T @ G + 2 * alpha) @ G.T @ G


def ufs_sp_f(X, y, alpha=1e3, mu=-1.4, max_steps=20):
    n, d = X.shape
    W = np.random.randn(d, d)
    v = np.random.rand(n)
    L = calc_L(X, W)
    beta, k = optimal_beta_k(L)
    obj = obj_v(v, L, beta, k)
    for _ in range(max_steps):
        v = update_v(L, beta, k)
        W = update_W(X, v, alpha)
        k /= mu
        L = calc_L(X, W)
        new_obj = obj_v(v, L, beta, k)
        if convergence(new_obj, obj):
            break
        obj = new_obj
    w = l_2_1_norm_vec(W)
    return w
