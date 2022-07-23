import pandas as pd
import numpy as np
import os
from data.utils import split_X_y, LabelColumnLoader


datasets = ['ALL', 'ayeastCC', 'bcellViper', 'bladderbatch', 'breastCancerVDX', 'CLL', 'COPDSexualDimorphism.data',
            'curatedOvarianData', 'DLBCL', 'leukemiasEset']
label_columns = {'ALL': 'MDRClass',
                 'ayeastCC': 'Class',
                 'bcellViper': 'TypeClass',
                 'bladderbatch': 'CancerClass',
                 'breastCancerVDX': 'oestrogenreceptorsClass',
                 'CLL': 'Class',
                 'COPDSexualDimorphism.data': 'DiagClass',
                 'curatedOvarianData': 'GradeClass',
                 'DLBCL': 'IPIClass',
                 'leukemiasEset': 'LeukemiaTypeClass'}
dataset_sizes = {'ALL': (128, 12625),
                 'ayeastCC': (50, 6228),
                 'bcellViper': (211, 6249),
                 'bladderbatch': (57, 22283),
                 'breastCancerVDX': (344, 22284),
                 'CLL': (22, 12625),
                 'COPDSexualDimorphism.data': (229, 14497),
                 'curatedOvarianData': (194, 3584),
                 'DLBCL': (194, 3583),
                 'leukemiasEset': (60, 20172)}


class BioconductorLoader(LabelColumnLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'bioconductor', f'{name}.csv')
        df = pd.read_csv(path, index_col=0, header=None, low_memory=False).T
        df = df.drop(np.nan, axis=1)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


bioconductor_loader = BioconductorLoader(label_columns)
