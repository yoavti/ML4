import math
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold


def cv_method(n):
    if n < 50:
        return LeavePOut(2)
    elif 50 <= n < 100:
        return LeaveOneOut()
    elif 100 <= n < 1000:
        return KFold(10, shuffle=True)
    return KFold(5, shuffle=True)


def cv_method_name(n):
    if n < 50:
        return 'Leave-pair-out'
    elif 50 <= n < 100:
        return 'LOOCV'
    elif 100 <= n < 1000:
        return '10-Fold'
    return '5-Fold'


def num_folds(n):
    if n < 50:
        return math.comb(n, 2)
    elif 50 <= n < 100:
        return n
    elif 100 <= n < 1000:
        return 10
    return 5
