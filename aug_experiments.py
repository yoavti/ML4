from sklearnex import patch_sklearn
patch_sklearn()

import os
import json

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pprint import PrettyPrinter

from data import data_loader
from experiment_utils.cv import cv_method, num_rows
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import named_score_funcs, named_classifiers, ks
from experiment_utils.preprocess import preprocess_steps

from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

pp = PrettyPrinter()

fss = {name: SelectKBest(score_func) for name, score_func in named_score_funcs.items()}
fss['select_fdr'] = SelectFdr(alpha=0.1)

parser = ArgumentParser(description='Data augmentation experiments.')
parser.add_argument('-d', '--dataset', type=str, help='dataset')
parser.add_argument('-fs', '--feature_selection', type=str, choices=list(fss), help='feature selection method')
parser.add_argument('-clf', '--classifier', type=str, choices=list(named_classifiers), help='classifier')
parser.add_argument('-k', '--n_features_to_select', default=10, type=int, choices=ks,
                    help='number of features to select')
args = parser.parse_args()


def run_aug(ds, fs, clf):
    fs_orig = clone(fs)
    clf_orig = clone(clf)

    X, y = data_loader.load(ds)

    y = LabelEncoder().fit_transform(y)
    n, d = X.shape
    for _, transformer in preprocess_steps(n):
        X = transformer.fit_transform(X, y)

    n, d = X.shape

    _num_rows = num_rows(n)
    X = X[:_num_rows]
    y = y[:_num_rows]

    n, d = X.shape

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, index=None, columns=None)

    metrics = get_metrics()
    metric_values = {name: [] for name in metrics}

    for train_index, test_index in cv_method(n).split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fs = clone(fs_orig)
        clf = clone(clf_orig)

        fs.fit(X_train, y_train)
        X_train = fs.transform(X_train)
        X_test = fs.transform(X_test)

        kernels = ['linear', 'rbf']
        pcas = [KernelPCA(kernel=kernel) for kernel in kernels]
        reduced_X_trains = [pca.fit_transform(X_train) for pca in pcas]
        reduced_X_tests = [pca.transform(X_test) for pca in pcas]
        X_train = np.hstack([X_train] + [X for X in reduced_X_trains])
        X_test = np.hstack([X_test] + [X for X in reduced_X_tests])

        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        for metric_name, metric in metrics.items():
            metric_values[metric_name].append(metric(y_test, y_pred))

    return metric_values


def main():
    metric_values = run_aug(args.dataset, fss[args.feature_selection], named_classifiers[args.classifier])
    pp.pprint(metric_values)
    results_path = os.path.join('results', 'aug', args.dataset)
    with open(os.path.join(results_path, 'res.json'), 'w+') as f:
        json.dump(metric_values, f)


if __name__ == '__main__':
    main()
