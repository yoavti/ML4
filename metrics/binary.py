import numpy as np


def binary_metric(metric):
    def averaged(y_true, y_pred):
        def label_comp(label):
            return metric(y_true == label, y_pred == label)
        return np.mean(np.vectorize(label_comp)(np.unique(np.concatenate((y_true, y_pred)))))
    return averaged
