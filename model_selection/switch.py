import os

import pandas as pd

from time import time

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest


class ClassifierSwitcher(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=SGDClassifier()):
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


fs_results = dict(times=[], scores=[], features=[])


class FSSwitcher(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=SelectKBest(), results_path=None):
        self.transformer = transformer
        self.results_path = results_path

    def fit(self, X, y=None, **kwargs):
        start = time()
        self.transformer.fit(X, y)
        elapsed = time() - start
        mask = self.transformer.get_support()
        scores = self.transformer.scores_
        selected_scores = scores[mask]
        feature_features = self.transformer.get_feature_names_out()
        fs_results['times'].append(elapsed)
        fs_results['scores'].append(selected_scores)
        fs_results['features'].append(feature_features)
        if self.results_path:
            fname = self.transformer.__class__.__name__
            if isinstance(self.transformer, SelectKBest):
                fname = self.transformer.score_func.__name__
            path = os.path.join(self.results_path, fname + '.csv')
            if not os.path.exists(path):
                processed = self.transformer.transform(X)
                df = pd.DataFrame(processed)
                df.to_csv(path)
        return self

    def transform(self, X):
        return self.transformer.transform(X)
