import pandas as pd
import os
from scipy.io import arff
from data.utils import split_X_y, LabelColumnLoader


datasets = ['Breast', 'CNS', 'Lung', 'Lymphoma', 'MLL', 'Ovarian', 'SRBCT']
label_columns = {'Breast': 'Class',
                 'CNS': 'CLASS',
                 'Lung': 'type',
                 'Lymphoma': 'class',
                 'MLL': 'class',
                 'Ovarian': 'Class',
                 'SRBCT': 'CLASS'}
dataset_sizes = {'Breast': (24481, 97),
                 'CNS': (7129, 60),
                 'Lung': (12600, 203),
                 'Lymphoma': (4026, 66),
                 'MLL': (12582, 72),
                 'Ovarian': (15154, 253),
                 'SRBCT': (2308, 83)}


class ARFFLoader(LabelColumnLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'ARFF', f'{name}.arff')
        data = arff.loadarff(path)[0]
        df = pd.DataFrame(data)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


arff_loader = ARFFLoader(label_columns)
