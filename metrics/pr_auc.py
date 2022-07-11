from sklearn.metrics import precision_recall_curve, auc


def pr_auc_score(y_true, probas_pred, *, pos_label=None, sample_weight=None):
    precision, recall, threshold = precision_recall_curve(y_true, probas_pred,
                                                          pos_label=pos_label, sample_weight=sample_weight)
    return auc(precision, recall)
