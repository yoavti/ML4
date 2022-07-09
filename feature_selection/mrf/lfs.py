from sklearn.utils import check_X_y
from functools import partial
import numpy as np


def combination_sum(arr):
    return (arr.T @ arr).sum()


def markov_coefficient(j, X, X_C, gamma):
    n, p = X.shape
    first = sum(combination_sum(X_i[:, j]) / X_i.shape[0] for X_i in X_C.values()) / n
    second = -(gamma / n) * (X[:, j] ** 2).sum()
    third = ((gamma - 1) / (n ** 2)) * combination_sum(X[:, j])
    return first + second + third


def lfs(X, y=None, *, gamma):
    X, y = check_X_y(X, y)
    n, p = X.shape
    X_C = {y_val: [] for y_val in np.unique(y)}
    for X_i, y_i in zip(X, y):
        X_C[y_i].append(X_i)
    X_C = {y_val: np.array(observations) for y_val, observations in X_C.items()}
    theta = [markov_coefficient(j, X, X_C, gamma) for j in range(p)]
    theta = np.array(theta)
    theta += theta.min()
    theta /= theta.sum()
    return theta
