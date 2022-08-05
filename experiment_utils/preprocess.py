from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest


def preprocess_steps(n):
    ret = [('imputer', SimpleImputer()),
           ('var_thresh', VarianceThreshold()),
           ('transform', PowerTransformer())]
    if n >= 1000:
        ret.append(('ff', SelectKBest(k=n)))
    return ret
