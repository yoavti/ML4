from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.preprocessing import LabelEncoder


def preprocess_steps(n):
    ret = [('imputer', SimpleImputer()),
           ('var_thresh', VarianceThreshold()),
           ('transform', PowerTransformer())]
    if n >= 1000:
        ret.append(('ff', SelectKBest(k=n)))
    return ret


def preprocess(X, y):
    y = LabelEncoder().fit_transform(y)
    n, d = X.shape
    for _, transformer in preprocess_steps(n):
        X = transformer.fit_transform(X, y)
    return X, y
