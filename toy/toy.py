import pandas as pd


FILENAME = 'SPECTF.train'


def split_X_y(df, column):
    y = df[column]
    X = df.drop(column, axis=1)
    return X, y


def load_toy(path=FILENAME):
    return split_X_y(pd.read_csv(path), '1')
