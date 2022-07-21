from time import time
from argparse import ArgumentParser
from pprint import PrettyPrinter

from data import data_loader
from feature_selection import lfs, ufs_sp, mrmr_score, relief_f

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFdr

parser = ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

pp = PrettyPrinter()


def preprocess(X, y):
    y = LabelEncoder().fit_transform(y)
    n, d = X.shape
    transformers = [SimpleImputer(),
                    VarianceThreshold(),
                    PowerTransformer(),
                    SelectKBest(k='all' if n < 1000 else 1000)]
    for transformer in transformers:
        X = transformer.fit_transform(X, y)
    return X, y


def run(dataset):
    X, y = data_loader.load(dataset)
    X, y = preprocess(X, y)
    score_funcs = {'lfs': lfs,
                   'ufs_sp': ufs_sp,
                   'mrmr_score': mrmr_score,
                   'relief_f': relief_f}
    times = {}
    for name, score_func in score_funcs.items():
        start = time()
        score_func(X, y)
        times[name] = time() - start
    start = time()
    SelectFdr(alpha=0.1).fit_transform(X, y)
    times['SelectFdr'] = time() - start
    pp.pprint(times)


if __name__ == '__main__':
    run(args.dataset.strip())
