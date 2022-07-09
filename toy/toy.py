import pandas as pd
from feature_selection import lfs, ufs_sp
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_selection import SelectKBest


FILENAME = 'SPECTF.train'


def split_X_y(df, column):
    y = df[column]
    X = df.drop(column, axis=1)
    return X, y


def load_toy(path=FILENAME):
    return split_X_y(pd.read_csv(path), '1')


if __name__ == '__main__':
    X, y = load_toy()
    alpha = lfs(X, y, gamma=-1)
    w = ufs_sp(X.to_numpy(), 1, 2, 100, 100)
