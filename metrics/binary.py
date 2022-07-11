import numpy as np


def binary_metric(metric, y_true, y_pred):
    vals = []
    for y_val in np.unique(np.concatenate((y_true, y_pred))):
        y_true_bin = y_true == y_val
        y_pred_bin = y_pred == y_val
        vals.append(metric(y_true_bin, y_pred_bin))
    return sum(vals) / len(vals)
