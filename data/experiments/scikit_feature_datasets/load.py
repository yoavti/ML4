from scipy.io import loadmat
import os
from data.utils import LabelFileLoader


datasets = ['ALLAML', 'arcene', 'BASEHOCK', 'Carcinom', 'CLL-SUB-111', 'COIL20', 'colon', 'gisette', 'GLI-85', 'GLIOMA',
            'Isolet', 'leukemia', 'lung', 'lung_small', 'lymphoma', 'madelon', 'nci9', 'ORL', 'orlraws10P', 'PCMAC',
            'pixraw10P', 'Prostate-GE', 'RELATHE', 'SMK-CAN-187', 'TOX-171', 'USPS', 'warpAR10P', 'warpPIE10P', 'Yale']
dataset_sizes = {'ALLAML': (72, 7129),
                 'arcene': (200, 10000),
                 'BASEHOCK': (1993, 4862),
                 'Carcinom': (174, 9182),
                 'CLL-SUB-111': (111, 11340),
                 'COIL20': (1440, 1024),
                 'colon': (62, 2000),
                 'gisette': (7000, 5000),
                 'GLI-85': (85, 22283),
                 'GLIOMA': (50, 4434),
                 'Isolet': (1560, 617),
                 'leukemia': (72, 7070),
                 'lung': (203, 3312),
                 'lung_small': (73, 325),
                 'lymphoma': (96, 4026),
                 'madelon': (2600, 500),
                 'nci9': (60, 9712),
                 'ORL': (400, 1024),
                 'orlraws10P': (100, 10304),
                 'PCMAC': (1943, 3289),
                 'pixraw10P': (100, 10000),
                 'Prostate-GE': (102, 5966),
                 'RELATHE': (1427, 4322),
                 'SMK-CAN-187': (187, 19993),
                 'TOX-171': (171, 5748),
                 'USPS': (9298, 256),
                 'warpAR10P': (130, 2400),
                 'warpPIE10P': (210, 2420),
                 'Yale': (165, 1024)}


class ScikitFeatureLoader(LabelFileLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'scikit_feature_datasets', f'{name}.mat')
        mat = loadmat(path)
        X = mat['X']
        Y = mat['Y']
        Y = Y.squeeze()
        return X, Y


scikit_feature_loader = ScikitFeatureLoader(datasets)
