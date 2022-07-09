import pandas as pd
import os


DIR = os.path.join('data', 'experiments', 'Datamicroarray')
selected_datasets = {'alon', 'borovecki', 'burczynski', 'chiaretti', 'chin'}


def load_datamicroarray(name):
    return [pd.read_csv(os.path.join(DIR, name, f'{purpose}.csv'), header=None) for purpose in ['inputs', 'outputs']]
