from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd

from data import data_loader
from experiment_utils.cv import cv_method
from experiment_utils.metrics import get_metrics
from experiment_utils.preprocess import preprocess

from sklearn.decomposition import KernelPCA

from imblearn.over_sampling import SMOTE


def insert_pca_columns(df, data, kernel):
    n, d = data.shape
    df[[f'{kernel}_{i}' for i in range(d)]] = data


def run(ds, fs, clf):
    X, y = data_loader.load(ds)
    X, y = preprocess(X, y)
    n, d = X.shape
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, index=None, columns=None)

    metrics = get_metrics()
    metric_values = {name: [] for name in metrics}

    for train_index, test_index in cv_method(n).split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fs.fit(X_train, y_train)
        X_train = fs.transform(X_train, y_train)
        X_test = fs.transform(X_test, y_test)

        kernels = ['linear', 'rbf']
        pcas = {kernel: KernelPCA(kernel=kernel) for kernel in kernels}
        reduced_X_trains = {kernel: pca.fit_transform(X_train) for kernel, pca in pcas.items()}
        reduced_X_tests = {kernel: pca.transform(X_test) for kernel, pca in pcas.items()}
        for kernel in kernels:
            insert_pca_columns(X_train, reduced_X_trains[kernel], kernel)
            insert_pca_columns(X_test, reduced_X_tests[kernel], kernel)

        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        for metric_name, metric in metrics.items():
            metric_values[metric_name].append(metric(y_test, y_pred))

    return metric_values
