from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import SGDClassifier


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


class TransformerSwitcher(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None, **kwargs):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)
