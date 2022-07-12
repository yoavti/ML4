from sklearnex import patch_sklearn
patch_sklearn()

import joblib

from argparse import ArgumentParser

from model_selection import ClassifierSwitcher, TransformerSwitcher

from data import data_loader

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold

from feature_selection import lfs, ufs_sp, mrmr_score
from sklearn.feature_selection import SelectKBest, SelectFdr, RFE
from ReliefF.ReliefF import ReliefF

from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, make_scorer
from metrics import pr_auc_score, binary_metric

parser = ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

pipeline = Pipeline([('imputer', SimpleImputer()),
                     ('var_thresh', VarianceThreshold()),
                     ('transform', PowerTransformer()),
                     ('fs', TransformerSwitcher()),
                     ('clf', ClassifierSwitcher(SVC()))],
                    memory='pipeline')


ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
classifiers = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(), SVC(), RandomForestClassifier()]
parameters = [
    {
        'fs__transformer': [SelectKBest()],
        'fs__transformer__score_func': [lfs, ufs_sp, mrmr_score],
        'fs__transformer__k': ks,
        'clf__estimator': classifiers,
    },
    {
        'fs__transformer': [SelectFdr(alpha=0.1)],
        'clf__estimator': classifiers,
    },
    {
        'fs__transformer': [RFE(SVR())],
        'fs__transformer__n_features_to_select': ks,
        'clf__estimator': classifiers,
    },
    {
        'fs__transformer': [ReliefF()],
        'fs__transformer__n_features_to_keep': ks,
        'clf__estimator': classifiers,
    },
]


def my_metrics():
    binary_metrics = {'ROC_AUC': roc_auc_score,
                      'PR_AUC': pr_auc_score}
    binary_metrics = {name: binary_metric(metric) for name, metric in binary_metrics.items()}
    regular_metrics = {'ACC': accuracy_score, 'MCC': matthews_corrcoef}
    all_metrics = regular_metrics | binary_metrics
    scorers = {name: make_scorer(metric) for name, metric in all_metrics.items()}
    return scorers


def run_experiment(dataset):
    X, y = data_loader.load(dataset)
    # X, y = load_arff(dataset)
    y = LabelEncoder().fit_transform(y)

    n, d = X.shape

    if n < 50:
        cv = LeavePOut(2)
    elif 50 <= n < 100:
        cv = LeaveOneOut()
    elif 100 <= n < 1000:
        cv = KFold(10, shuffle=True)
    else:
        cv = KFold(5, shuffle=True)

    gscv = GridSearchCV(pipeline, parameters, scoring=my_metrics(), refit='ROC_AUC', cv=cv)
    gscv.fit(X, y)
    joblib.dump(gscv.cv_results_, f'{dataset}.pkl')


if __name__ == '__main__':
    run_experiment(args.dataset)