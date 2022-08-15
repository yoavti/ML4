from sklearnex import patch_sklearn
patch_sklearn()

import os
import json

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from time import time

from data import data_loader
from experiment_utils.cv import cv_method, num_rows
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import named_score_funcs, named_classifiers, ks
from experiment_utils.preprocess import preprocess_steps

from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE, RandomOverSampler

parser = ArgumentParser(description='Data augmentation experiments.')
parser.add_argument('-d', '--dataset', type=str, help='dataset')
parser.add_argument('-fs', '--feature_selection', type=str, choices=list(named_score_funcs) + ['SelectFdr'],
                    help='feature selection method')
parser.add_argument('-clf', '--classifier', type=str, choices=list(named_classifiers), help='classifier')
parser.add_argument('-k', '--n_features_to_select', default=10, type=int, choices=ks,
                    help='number of features to select')
args = parser.parse_args()

fss = {name: SelectKBest(score_func, k=args.n_features_to_select) for name, score_func in named_score_funcs.items()}
fss['SelectFdr'] = SelectFdr(alpha=0.1)


def run_aug(ds, fs, clf):
    fs_orig = clone(fs)
    clf_orig = clone(clf)

    X, y = data_loader.load(ds)

    n, d = X.shape

    y = LabelEncoder().fit_transform(y)
    for _, transformer in preprocess_steps(d):
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
    metric_values['time'] = []

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

        if np.unique(y).size > 1:
            over_sampler = RandomOverSampler()
            X_train, y_train = over_sampler.fit_resample(X_train, y_train)

            sm = SMOTE()
            X_train, y_train = sm.fit_resample(X_train, y_train)

        start = time()
        clf.fit(X_train, y_train)
        fit_time = time() - start

        y_pred = clf.predict(X_test)

        for metric_name, metric in metrics.items():
            metric_values[metric_name].append(metric(y_test, y_pred))
        metric_values['time'].append(fit_time)

    return metric_values


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run_and_save_aug(dataset, feature_selection, classifier, k):
    fs = fss[feature_selection]
    clf = named_classifiers[classifier]
    parameters = dict(k=k, fs=feature_selection, clf=classifier)
    metric_values = run_aug(dataset, fs, clf)
    metric_values = pd.DataFrame(metric_values)
    aug_path = 'results'
    create_if_not_exists(aug_path)
    aug_path = os.path.join(aug_path, dataset)
    create_if_not_exists(aug_path)
    aug_path = os.path.join(aug_path, 'aug')
    create_if_not_exists(aug_path)
    results_path = os.path.join(aug_path, 'results.csv')
    metric_values.to_csv(results_path, index=False)
    parameters_path = os.path.join(aug_path, 'parameters.json')
    with open(parameters_path, 'w+') as f:
        json.dump(parameters, f)


def run_args():
    run_and_save_aug(args.dataset, args.feature_selection, args.classifier, args.n_features_to_select)


def run_best():
    df = pd.read_csv('best.csv')
    for _, row in df.iterrows():
        print(row)
        run_and_save_aug(row['dataset'], row['fs'], row['clf'], row['k'])


if __name__ == '__main__':
    run_best()
