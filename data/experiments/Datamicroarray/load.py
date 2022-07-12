import pandas as pd
import os
from data.utils import FileLoader


class DatamicroarrayLoader(FileLoader):
    def _load(self, name, parent=''):
        purposes = ['inputs', 'outputs']
        paths = [os.path.join(parent, 'Datamicroarray', name, f'{purpose}.csv') for purpose in purposes]
        dfs = [pd.read_csv(path, header=None) for path in paths]
        return dfs


datamicroarray_loader = DatamicroarrayLoader({name: None
                                              for name in ['alon', 'borovecki', 'burczynski', 'chiaretti', 'chin']})
