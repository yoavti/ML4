import numpy as np


def binary_metric(metric):
    def averaged(y_true, y_pred):
        def label_comp(label):
            y_true_bin = y_true == label
            y_pred_bin = y_pred == label
            if len(np.unique(y_true_bin)) != 2:
                return 1
            return metric(y_true_bin, y_pred_bin)
        return np.mean(np.vectorize(label_comp)(np.unique(np.concatenate((y_true, y_pred)))))
    return averaged
