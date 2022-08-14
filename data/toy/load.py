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

    def is_dataset_available(self, name, parent=''):
        return name == 'toy'

    def available_datasets(self):
        return ['toy']


toy_loader = ToyLoader('SPECTF.train')
