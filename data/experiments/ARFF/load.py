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
dataset_sizes = {'Breast': (97, 24481),
                 'CNS': (60, 7129),
                 'Lung': (203, 12600),
                 'Lymphoma': (66, 4026),
                 'MLL': (72, 12582),
                 'Ovarian': (253, 15154),
                 'SRBCT': (83, 2308)}


class ARFFLoader(LabelColumnLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'ARFF', f'{name}.arff')
        data = arff.loadarff(path)[0]
        df = pd.DataFrame(data)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


arff_loader = ARFFLoader(label_columns)
