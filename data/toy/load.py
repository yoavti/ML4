import pandas as pd
import os
from data.utils import split_X_y


DIR = os.path.join('data', 'toy')
FILENAME = 'SPECTF.train'


def load_toy(name=FILENAME):
    path = os.path.join(DIR, name)
    df = pd.read_csv(path, header=None)
    X, y = split_X_y(df, 0)
    return X, y
