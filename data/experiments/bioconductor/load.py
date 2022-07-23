import pandas as pd
import numpy as np
import os
from data.utils import split_X_y, LabelColumnLoader


datasets = ['DLBCL', 'curatedOvarianData', 'ayeastCC', 'bcellViper', 'CLL', 'ALL', 'COPDSexualDimorphism.data',
            'leukemiasEset', 'bladderbatch', 'breastCancerVDX']  # sorted in increasing size
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
dataset_sizes = {'ALL': (12625, 128),
                 'ayeastCC': (6228, 50),
                 'bcellViper': (6249, 211),
                 'bladderbatch': (22283, 57),
                 'breastCancerVDX': (22284, 344),
                 'CLL': (12625, 22),
                 'COPDSexualDimorphism.data': (14497, 229),
                 'curatedOvarianData': (3584, 194),
                 'DLBCL': (3583, 194),
                 'leukemiasEset': (20172, 60)}


class BioconductorLoader(LabelColumnLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'bioconductor', f'{name}.csv')
        df = pd.read_csv(path, index_col=0, header=None, low_memory=False).T
        df = df.drop(np.nan, axis=1)
        X, y = split_X_y(df, self._label_columns[name])
        return X, y


bioconductor_loader = BioconductorLoader(label_columns)
