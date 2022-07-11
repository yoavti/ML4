import numpy as np


def binary_metric(metric):
    def averaged(y_true, y_pred):
        def label_metric(label):
            y_true_bin = y_true == label
            y_pred_bin = y_pred == label
            y_true_bin = y_true_bin.astype(int)
            y_pred_bin = y_pred_bin.astype(int)
            if len(np.unique(y_true_bin)) != 2:
                return 1
            return metric(y_true_bin, y_pred_bin)
        y_total = np.concatenate((y_true, y_pred))
        labels = np.unique(y_total)
        metric_values = np.vectorize(label_metric)(labels)
        return np.mean(metric_values)
    return averaged
