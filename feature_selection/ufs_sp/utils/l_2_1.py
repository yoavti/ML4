import numpy as np


def l_2_1_norm_vec(X):
    return np.apply_along_axis(np.linalg.norm, 0, X)


def l_2_1_norm(X):
    return l_2_1_norm_vec(X).sum()
