import pandas as pd
import os
from data.utils import Loader, split_X_y


class ToyLoader(Loader):
    def __init__(self, filename):
        self._filename = filename

    def _load(self, name, parent=''):
        path = os.path.join(parent, 'toy', self._filename)
        df = pd.read_csv(path, header=None)
        X, y = split_X_y(df, 0)
        return X, y

    def is_dataset_available(self, name):
        return name == 'toy'

    def available_datasets(self):
        return ['toy']

    def dataset_size(self, name):
        return 44, 80


toy_loader = ToyLoader('SPECTF.train')


if __name__ == '__main__':
    df = pd.read_csv('SPECTF.train', header=None)
    X, y = split_X_y(df, 0)
    print(X.shape)
