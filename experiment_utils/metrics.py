from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, make_scorer
from metrics import pr_auc_score, binary_metric


def get_metrics(score=False):
    binary_metrics = {'ROC_AUC': roc_auc_score,
                      'PR_AUC': pr_auc_score}
    binary_metrics = {name: binary_metric(metric) for name, metric in binary_metrics.items()}
    regular_metrics = {'ACC': accuracy_score, 'MCC': matthews_corrcoef}
    all_metrics = {**binary_metrics, **regular_metrics}
    scorers = {name: make_scorer(metric) for name, metric in all_metrics.items()}
    return scorers if score else all_metrics
