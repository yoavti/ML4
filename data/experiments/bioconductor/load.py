import pandas as pd
import numpy as np
import os
from data.utils import split_X_y


DIR = os.path.join('data', 'experiments', 'bioconductor')
label_columns = {'ayeastCC': 'Class',
                 'CLL': 'Class',
                 'DLBCL': 'IPIClass',
                 'curatedOvarianData': 'GradeClass',
                 'leukemiasEset': 'LeukemiaTypeClass'}


def load_bioconductor(name):
    path = os.path.join(DIR, f'{name}.csv')
    df = pd.read_csv(path, index_col=0, header=None, low_memory=False).T
    df = df.drop(np.nan, axis=1)
    X, y = split_X_y(df, label_columns[name])
    return X, y
