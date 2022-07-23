import pandas as pd
import os
from data.utils import LabelFileLoader


purposes = ['inputs', 'outputs']
datasets = ['sorlie', 'christensen', 'alon', 'khan', 'gravier', 'su', 'pomeroy', 'west', 'golub', 'shipp',
            'subramanian', 'gordon', 'singh', 'chiaretti', 'tian', 'yeoh', 'chin', 'borovecki', 'chowdary', 'nakayama',
            'burczynski', 'sun']  # sorted in increasing size
dataset_sizes = {'alon': (2000, 62),
                 'borovecki': (22283, 31),
                 'burczynski': (22283, 127),
                 'chiaretti': (12625, 128),
                 'chin': (22215, 118),
                 'chowdary': (22283, 104),
                 'christensen': (1413, 217),
                 'golub': (7129, 72),
                 'gordon': (12533, 181),
                 'gravier': (2905, 168),
                 'khan': (2308, 63),
                 'nakayama': (22283, 105),
                 'pomeroy': (7128, 60),
                 'shipp': (7129, 77),
                 'singh': (12600, 102),
                 'sorlie': (456, 85),
                 'su': (5565, 102),
                 'subramanian': (10100, 50),
                 'sun': (54613, 180),
                 'tian': (12625, 173),
                 'west': (7129, 49),
                 'yeoh': (12625, 248)}


class DatamicroarrayLoader(LabelFileLoader):
    def _load(self, name, parent=''):
        paths = [os.path.join(parent, 'Datamicroarray', name, f'{purpose}.csv') for purpose in purposes]
        dfs = [pd.read_csv(path, header=None) for path in paths]
        return dfs


datamicroarray_loader = DatamicroarrayLoader(datasets)
