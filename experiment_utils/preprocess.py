from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest


def preprocess_steps(d):
    ret = [('imputer', SimpleImputer()),
           ('var_thresh', VarianceThreshold()),
           ('transform', PowerTransformer())]
    if d >= 1000:
        ret.append(('ff', SelectKBest(k=d)))
    return ret
