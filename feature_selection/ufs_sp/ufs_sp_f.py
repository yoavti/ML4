import numpy as np
from feature_selection.ufs_sp.utils import calc_G
from feature_selection.ufs_sp.ufs_sp import create_ufs_sp


def solve_W(X, v, alpha, **kwargs):
    G = calc_G(X, v)
    return np.linalg.inv(G.T @ G + 2 * alpha) @ G.T @ G


ufs_sp_f = create_ufs_sp(solve_W)
