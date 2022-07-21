from sklearnex import patch_sklearn
patch_sklearn()

import json
import os

import pandas as pd

from pprint import PrettyPrinter
from time import time
from argparse import ArgumentParser

from model_selection import ClassifierSwitcher, TransformerSwitcher

from data import data_loader

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import GridSearchCV
from experiments.utils.cv import cv_method

from feature_selection import lfs, ufs_sp, mrmr_score, relief_f
from sklearn.feature_selection import SelectKBest, SelectFdr, RFE

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

pp = PrettyPrinter()

# ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
ks = [10]
classifiers = [KNeighborsClassifier(), GaussianNB(), LogisticRegression(), SVC(), RandomForestClassifier()]
parameters = [
    {
        'fs__transformer': [SelectKBest()],
        'fs__transformer__score_func': [lfs, ufs_sp, mrmr_score, relief_f],
        'fs__transformer__k': ks,
        'clf__estimator': classifiers,
    },
    {
        'fs__transformer': [SelectFdr(alpha=0.1)],
        'clf__estimator': classifiers,
    },
    # {
    #     'fs__transformer': [RFE(SVR(kernel='linear'))],
    #     'fs__transformer__n_features_to_select': ks,
    #     'clf__estimator': classifiers,
    # },
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
    y = LabelEncoder().fit_transform(y)

    n, d = X.shape

    pipeline = Pipeline([('imputer', SimpleImputer()),
                         ('var_thresh', VarianceThreshold()),
                         ('transform', PowerTransformer()),
                         ('ff', SelectKBest(k='all' if n < 1000 else 1000)),
                         ('fs', TransformerSwitcher()),
                         ('clf', ClassifierSwitcher(SVC()))],
                        memory=os.path.join('pipeline_memory', dataset))

    real_n = min(n, 1000)

    cv = cv_method(real_n)

    gscv = GridSearchCV(pipeline, parameters, scoring=my_metrics(), refit='ROC_AUC', cv=cv)
    gscv.fit(X, y)

    best = {'index': int(gscv.best_index_), 'score': gscv.best_score_}
    pp.pprint(best)
    with open(os.path.join('../results', f'{dataset}.json'), 'w+') as f:
        json.dump(best, f)
    cv_results = gscv.cv_results_
    pp.pprint(cv_results)
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(os.path.join('../results', f'{dataset}.csv'))


if __name__ == '__main__':
    start = time()
    run_experiment(args.dataset.strip())
    print(time() - start)
