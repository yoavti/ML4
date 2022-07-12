import pandas as pd
import os
from scipy.io import arff
from data.utils import split_X_y, FileLoader


class ARFFLoader(FileLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'ARFF', f'{name}.arff')
        data = arff.loadarff(path)[0]
        df = pd.DataFrame(data)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


arff_loader = ARFFLoader({'Lymphoma': 'class',
                          'CNS': 'CLASS',
                          'SRBCT': 'CLASS',
                          'MLL': 'class',
                          'Breast': 'Class'})
