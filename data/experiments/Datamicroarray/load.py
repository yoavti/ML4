import pandas as pd
import os
from data.utils import LabelFileLoader


purposes = ['inputs', 'outputs']
datasets = ['alon', 'borovecki', 'burczynski', 'chiaretti', 'chin', 'chowdary', 'christensen', 'golub', 'gordon',
            'gravier', 'khan', 'nakayama', 'pomeroy', 'shipp', 'singh', 'sorlie', 'su', 'subramanian', 'sun', 'tian',
            'west', 'yeoh']
dataset_sizes = {'alon': (62, 2000),
                 'borovecki': (31, 22283),
                 'burczynski': (127, 22283),
                 'chiaretti': (128, 12625),
                 'chin': (118, 22215),
                 'chowdary': (104, 22283),
                 'christensen': (217, 1413),
                 'golub': (72, 7129),
                 'gordon': (181, 12533),
                 'gravier': (168, 2905),
                 'khan': (63, 2308),
                 'nakayama': (105, 22283),
                 'pomeroy': (60, 7128),
                 'shipp': (77, 7129),
                 'singh': (102, 12600),
                 'sorlie': (85, 456),
                 'su': (102, 5565),
                 'subramanian': (50, 10100),
                 'sun': (180, 54613),
                 'tian': (173, 12625),
                 'west': (49, 7129),
                 'yeoh': (248, 12625)}


class DatamicroarrayLoader(LabelFileLoader):
    def _load(self, name, parent=''):
        paths = [os.path.join(parent, 'Datamicroarray', name, f'{purpose}.csv') for purpose in purposes]
        dfs = [pd.read_csv(path, header=None) for path in paths]
        return dfs


datamicroarray_loader = DatamicroarrayLoader(datasets)
