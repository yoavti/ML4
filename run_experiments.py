from sklearnex import patch_sklearn
patch_sklearn()

import json
import os

import pandas as pd

from pprint import PrettyPrinter
from time import time

from model_selection import ClassifierSwitcher, TransformerSwitcher
from data import data_loader

from experiment_utils.preprocess import preprocess_steps
from experiment_utils.cv import cv_method
from experiment_utils.metrics import get_metrics
from experiment_utils.parameters import score_funcs, ks, classifiers
from experiment_utils.argument_parser import dataset

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFdr
# from sklearn.feature_selection import RFE
from sklearn.svm import SVC
# from sklearn.svm import SVR

pp = PrettyPrinter()

parameters = [
    {
        'fs__transformer': [SelectKBest()],
        'fs__transformer__score_func': score_funcs,
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


def run_experiment(ds):
    X, y = data_loader.load(ds)
    y = LabelEncoder().fit_transform(y)

    n, d = X.shape

    pipeline = Pipeline(preprocess_steps(n) + [('fs', TransformerSwitcher()), ('clf', ClassifierSwitcher(SVC()))],
                        memory=os.path.join('pipeline_memory', ds))

    gscv = GridSearchCV(pipeline, parameters, scoring=get_metrics(), refit='ROC_AUC', cv=cv_method(min(n, 1000)))
    gscv.fit(X, y)

    best = {'index': int(gscv.best_index_), 'score': gscv.best_score_}
    pp.pprint(best)
    with open(os.path.join('results', f'{ds}.json'), 'w+') as f:
        json.dump(best, f)
    cv_results = gscv.cv_results_
    pp.pprint(cv_results)
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(os.path.join('results', f'{ds}.csv'))


if __name__ == '__main__':
    start = time()
    run_experiment(dataset)
    print(time() - start)
