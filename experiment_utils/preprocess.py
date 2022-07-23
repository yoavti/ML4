from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.preprocessing import LabelEncoder


def preprocess_steps(n):
    return [('imputer', SimpleImputer()),
            ('var_thresh', VarianceThreshold()),
            ('transform', PowerTransformer()),
            ('ff', SelectKBest(k='all' if n < 1000 else 1000))]


def preprocess(X, y):
    y = LabelEncoder().fit_transform(y)
    n, d = X.shape
    for _, transformer in preprocess_steps(n):
        X = transformer.fit_transform(X, y)
    return X, y
