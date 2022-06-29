import numpy as np
from functools import partial


def l_2_1_norm_vec(X):
    return np.apply_along_axis(lambda X_i: (X_i ** 2).sum(), 0, X)


def l_2_1_norm(X):
    l_2_1_norm_vec(X).sum()


def rel_change(new, old):
    return (np.linalg.norm(new - old) ** 2) / old


def convergence(new, old, epsilon=1e-5):
    return rel_change(new, old) <= epsilon


def calc_L(X, W):
    return np.apply_along_axis(lambda X_i: np.linalg.norm(X_i - X_i @ W), 0, X)


def calc_single_v(L_i, beta, k):
    if L_i <= np.sqrt(beta / (k + 1)):
        return 1
    if L_i <= 1 / np.sqrt(k):
        return 0
    return beta * ((1 / L_i) - k)


def update_v(L, beta, k):
    return np.apply_along_axis(partial(calc_single_v, beta=beta, k=k), 0, L)


def calc_G(X, v):
    U = np.diag(v ** 0.5)
    G = U @ X
    return G


def update_W(G, D, alpha):
    return np.linalg.inv(G.T @ G + alpha * D) @ G.T @ G


def update_D(W):
    return np.diag(np.apply_along_axis(lambda W_i: 1 / (2 * np.linalg.norm(W_i)), 0, W))


def obj_W(W, G, alpha):
    return np.linalg.norm(G - G @ W) + alpha * l_2_1_norm(W)


def obj_v(v, L, beta, k):
    return (v * L).sum() + ((beta ** 2) / (v + beta * k)).sum()


def solve_W(X, v, alpha):
    n, d = X.shape
    G = calc_G(X, v)
    D = np.diag(np.random.randn(d))
    W = np.random.randn(d, d)
    obj = obj_W(W, G, alpha)
    while True:
        W = update_W(G, D, alpha)
        D = update_D(W)
        new_obj = obj_W(W, G, alpha)
        if convergence(new_obj, obj):
            break
        obj = new_obj
    return W


def ufs_sp(X, alpha, beta, k, mu):
    n, d = X.shape
    W = np.random.randn(d, d)
    v = np.random.randn(d)
    L = calc_L(X, W)
    obj = obj_v(v, L, beta, k)
    while True:
        v = update_v(L, beta, k)
        W = solve_W(X, v, alpha)
        k /= mu
        L = calc_L(X, W)
        new_obj = obj_v(v, L, beta, k)
        if convergence(new_obj, obj):
            break
        obj = new_obj
    return l_2_1_norm_vec(W)
