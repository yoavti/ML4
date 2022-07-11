import numpy as np
from skfeature.function.information_theoretical_based.MRMR import mrmr


def mrmr_score(X, y):
    F = mrmr(X, y)
    values = np.arange(F.shape[0])[::-1]
    ret = np.zeros(F.shape[0])
    ret[F] = values
    ret = ret.astype(int)
    return ret
