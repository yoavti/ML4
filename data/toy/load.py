import pandas as pd
from data.utils import split_X_y


FILENAME = 'SPECTF.train'


def load_toy(name=FILENAME):
    df = pd.read_csv(name)
    X, y = split_X_y(df, '1')
    return X, y
