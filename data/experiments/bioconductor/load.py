import pandas as pd
import numpy as np
import os
from data.utils import split_X_y, FileLoader


label_columns = {'ayeastCC': 'Class',
                 'CLL': 'Class',
                 'DLBCL': 'IPIClass',
                 'curatedOvarianData': 'GradeClass',
                 'leukemiasEset': 'LeukemiaTypeClass'}


class BioconductorLoader(FileLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'bioconductor', f'{name}.csv')
        df = pd.read_csv(path, index_col=0, header=None, low_memory=False).T
        df = df.drop(np.nan, axis=1)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


bioconductor_loader = BioconductorLoader({'ayeastCC': 'Class',
                                          'CLL': 'Class',
                                          'DLBCL': 'IPIClass',
                                          'curatedOvarianData': 'GradeClass',
                                          'leukemiasEset': 'LeukemiaTypeClass'})
