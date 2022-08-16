import json
import os

import pandas as pd

from pprint import PrettyPrinter
from time import time
from argparse import ArgumentParser

from model_selection import ClassifierSwitcher, FSSwitcher, fs_results
from data import data_loader

from experiment_utils.preprocess import preprocess_steps
from experiment_utils.cv import cv_method
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import score_funcs, ks, classifiers

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.svm import SVC

parser = ArgumentParser()
parser.add_argument('dataset', type=str, help='dataset to run experiments on')
parser.add_argument('-k', default=10, type=int, choices=ks, required=False, help='number of selected features')
args = parser.parse_args()
dataset = args.dataset.strip()
k = args.k

pp = PrettyPrinter()

parameters = [
    {
        'fs__transformer': [SelectKBest(k=k)],
        'fs__transformer__score_func': score_funcs,
        'clf__estimator': classifiers,
    },
    {
        'fs__transformer': [SelectFdr(alpha=0.1)],
        'clf__estimator': classifiers,
    },
]


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run_experiment(ds):
    X, y = data_loader.load(ds)
    y = LabelEncoder().fit_transform(y)

    n, d = X.shape

    results_path = 'results'
    create_if_not_exists(results_path)
    results_path = os.path.join(results_path, ds)
    create_if_not_exists(results_path)
    results_path = os.path.join(results_path, str(k))
    create_if_not_exists(results_path)

    additional_preprocess_steps = [('fs', FSSwitcher(results_path=results_path)),
                                   ('clf', ClassifierSwitcher(SVC()))]
    pipeline = Pipeline(preprocess_steps(d) + additional_preprocess_steps,
                        memory=os.path.join('pipeline_memory', ds))

    gscv = GridSearchCV(pipeline, parameters, scoring=get_metrics(True), refit='ROC_AUC', cv=cv_method(n))
    gscv.fit(X, y)

    best = {'index': int(gscv.best_index_), 'score': gscv.best_score_}
    pp.pprint(best)
    with open(os.path.join(results_path, f'best.json'), 'w+') as f:
        json.dump(best, f)

    cv_results = gscv.cv_results_
    pp.pprint(cv_results)
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(os.path.join(results_path, f'cv_results.csv'))

    pp.pprint(fs_results)
    df = pd.DataFrame(fs_results)
    df.to_csv(os.path.join(results_path, 'fs.csv'))


if __name__ == '__main__':
    start = time()
    run_experiment(dataset)
    print(time() - start)
