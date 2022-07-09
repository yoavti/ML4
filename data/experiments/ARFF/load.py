import pandas as pd
import os
from scipy.io import arff
from data.utils import split_X_y


DIR = os.path.join('data', 'experiments', 'ARFF')

label_columns = {'Lymphoma': 'class', 'CNS': 'CLASS', 'SRBCT': 'CLASS', 'MLL': 'class', 'Breast': 'Class'}


def load_arff(name):
    path = os.path.join(DIR, f'{name}.arff')
    data = arff.loadarff(path)[0]
    df = pd.DataFrame(data)
    X, y = split_X_y(df, label_columns[name])
    return X, y
