from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest


def preprocess_steps(n):
    return [('imputer', SimpleImputer()),
            ('var_thresh', VarianceThreshold()),
            ('transform', PowerTransformer()),
            ('ff', SelectKBest(k='all' if n < 1000 else 1000))]
