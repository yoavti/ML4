from sklearnex import patch_sklearn
patch_sklearn()

import json
import os

import pandas as pd

from pprint import PrettyPrinter
from time import time

from experiments.utils.argument_parser import dataset

from model_selection import ClassifierSwitcher, TransformerSwitcher

from data import data_loader

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import GridSearchCV
from experiments.utils.cv import cv_method

from sklearn.feature_selection import SelectKBest, SelectFdr
# from sklearn.feature_selection import RFE

from sklearn.svm import SVC
# from sklearn.svm import SVR

from experiments.utils.metrics import get_metrics

from experiments.utils.parameters import score_funcs, ks, classifiers

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

    pipeline = Pipeline([('imputer', SimpleImputer()),
                         ('var_thresh', VarianceThreshold()),
                         ('transform', PowerTransformer()),
                         ('ff', SelectKBest(k='all' if n < 1000 else 1000)),
                         ('fs', TransformerSwitcher()),
                         ('clf', ClassifierSwitcher(SVC()))],
                        memory=os.path.join('pipeline_memory', ds))

    real_n = min(n, 1000)

    gscv = GridSearchCV(pipeline, parameters, scoring=get_metrics(), refit='ROC_AUC', cv=cv_method(real_n))
    gscv.fit(X, y)

    best = {'index': int(gscv.best_index_), 'score': gscv.best_score_}
    pp.pprint(best)
    with open(os.path.join('../results', f'{ds}.json'), 'w+') as f:
        json.dump(best, f)
    cv_results = gscv.cv_results_
    pp.pprint(cv_results)
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(os.path.join('../results', f'{ds}.csv'))


if __name__ == '__main__':
    start = time()
    run_experiment(dataset)
    print(time() - start)
