from sklearnex import patch_sklearn
patch_sklearn()

import json
import os

from time import time
from argparse import ArgumentParser
from pprint import PrettyPrinter

import pandas as pd
import numpy as np

from data import data_loader
from feature_selection import lfs, ufs_sp, mrmr_score, relief_f

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFdr
from sklearn.model_selection import LeaveOneOut, LeavePOut, KFold

pp = PrettyPrinter()

parser = ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()


def cv_method(n):
    if n < 50:
        return LeavePOut(2)
    elif 50 <= n < 100:
        return LeaveOneOut()
    elif 100 <= n < 1000:
        return KFold(10, shuffle=True)
    return KFold(5, shuffle=True)


def preprocess(X, y):
    y = LabelEncoder().fit_transform(y)
    n, d = X.shape
    transformers = [SimpleImputer(),
                    VarianceThreshold(),
                    PowerTransformer(),
                    SelectKBest(k='all' if n < 1000 else 1000)]
    for transformer in transformers:
        X = transformer.fit_transform(X, y)
    return X, y


ks = [10]


def run_experiment(dataset):
    times = []
    all_selected_features = []
    all_selected_features_scores = []

    X, y = data_loader.load(dataset)

    X, y = preprocess(X, y)

    n, d = X.shape

    columns = range(n)
    if isinstance(X, pd.DataFrame):
        columns = X.columns
    columns = np.array(columns)

    score_funcs = {'lfs': lfs, 'ufs_sp': ufs_sp, 'mrmr_score': mrmr_score, 'relief_f': relief_f}

    cv = cv_method(n)

    for i, (train_index, test_index) in enumerate(cv.split(X, y)):
        times.append({})
        all_selected_features.append({})
        all_selected_features_scores.append({})

        X_ = np.array(X)
        X_train, X_test = X_[train_index], X_[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for name, score_func in score_funcs.items():
            all_selected_features[i][name] = {}
            all_selected_features_scores[i][name] = {}

            start = time()
            score = score_func(X_train, y_train)
            times[i][name] = time() - start

            idxs = np.argsort(score)
            selected_features = columns[idxs]
            selected_features_scores = score[idxs]
            for k in ks:
                all_selected_features[i][name][k] = selected_features[:k]
                all_selected_features_scores[i][name][k] = selected_features_scores[:k]

        select_fdr = SelectFdr(alpha=0.1)

        start = time()
        select_fdr.fit(X_train, y_train)
        times[i]['SelectFdr'] = time() - start

        all_selected_features[i]['SelectFdr'] = select_fdr.get_feature_names_out(columns)
        all_selected_features_scores[i]['SelectFdr'] = select_fdr.scores_[select_fdr.get_support()]

    return times, all_selected_features, all_selected_features_scores


def save_dict(dataset, name, d):
    print(name)
    pp.pprint(d)
    with open(os.path.join('../results', f'{dataset}_{name}.json'), 'w+') as f:
        json.dump(d, f)


def main(dataset):
    times, selected_features, selected_features_scores = run_experiment(dataset)
    save_dict(dataset, 'time', time)
    save_dict(dataset, 'selected_features', selected_features)
    save_dict(dataset, 'selected_features_scores', selected_features_scores)


if __name__ == '__main__':
    main(args.dataset.strip())
