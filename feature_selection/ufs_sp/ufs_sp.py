import numpy as np
from functools import partial
from feature_selection.ufs_sp.utils import convergence, l_2_1_norm_vec


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


def create_ufs_sp(solve_W):
    def ufs_sp(X, y, alpha=1e3, mu=-1.4, max_steps=20, max_steps_W=20):
        n, d = X.shape
        W = np.random.randn(d, d)
        v = np.random.rand(n)
        L = calc_L(X, W)
        beta, k = optimal_beta_k(L)
        obj = obj_v(v, L, beta, k)
        for _ in range(max_steps):
            v = update_v(L, beta, k)
            W = solve_W(X, v, alpha, max_steps_W)
            k /= mu
            L = calc_L(X, W)
            new_obj = obj_v(v, L, beta, k)
            if convergence(new_obj, obj):
                break
            obj = new_obj
        w = l_2_1_norm_vec(W)
        return w
    return ufs_sp
