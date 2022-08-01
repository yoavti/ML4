import numpy as np
from feature_selection.ufs_sp.utils import l_2_1_norm, calc_G, convergence, calc_L, optimal_beta_k, obj_v, update_v, l_2_1_norm_vec


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


def ufs_sp_l_2_1(X, y, alpha=1e3, mu=-1.4, max_steps=20, max_steps_W=20):
    n, d = X.shape
    W = np.random.randn(d, d)
    v = np.random.rand(n)
    L = calc_L(X, W)
    beta, k = optimal_beta_k(L)
    obj = obj_v(v, L, beta, k)
    for _ in range(max_steps):
        v = update_v(L, beta, k)
        W = solve_W(X, v, alpha, max_steps_W=max_steps_W)
        k /= mu
        L = calc_L(X, W)
        new_obj = obj_v(v, L, beta, k)
        if convergence(new_obj, obj):
            break
        obj = new_obj
    w = l_2_1_norm_vec(W)
    return w
