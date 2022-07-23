import numpy as np


EPSILON = 1e-5


def rel_change(new, old):
    return (np.linalg.norm(new - old) ** 2) / old


def convergence(new, old, epsilon=EPSILON):
    return rel_change(new, old) <= epsilon
