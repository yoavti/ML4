from scipy.io import loadmat
import os
from data.utils import LabelFileLoader


datasets = ['ALLAML', 'arcene', 'BASEHOCK', 'Carcinom', 'CLL-SUB-111', 'colon', 'GLI-85', 'GLIOMA', 'leukemia', 'lung',
            'lung_small', 'lymphoma', 'nci9', 'ORL', 'orlraws10P', 'PCMAC', 'pixraw10P', 'Prostate-GE', 'RELATHE',
            'SMK-CAN-187', 'TOX-171', 'warpAR10P', 'warpPIE10P', 'Yale']
dataset_sizes = {'ALLAML': (7129, 72),
                 'arcene': (10000, 200),
                 'BASEHOCK': (4862, 1993),
                 'Carcinom': (9182, 174),
                 'CLL-SUB-111': (11340, 111),
                 'colon': (2000, 62),
                 'GLI-85': (22283, 85),
                 'GLIOMA': (4434, 50),
                 'leukemia': (7070, 72),
                 'lung': (3312, 203),
                 'lung_small': (325, 73),
                 'lymphoma': (4026, 96),
                 'nci9': (9712, 60),
                 'ORL': (1024, 400),
                 'orlraws10P': (10304, 100),
                 'PCMAC': (3289, 1943),
                 'pixraw10P': (10000, 100),
                 'Prostate-GE': (5966, 102),
                 'RELATHE': (4322, 1427),
                 'SMK-CAN-187': (19993, 187),
                 'TOX-171': (5748, 171),
                 'warpAR10P': (2400, 130),
                 'warpPIE10P': (2420, 210),
                 'Yale': (1024, 165)}


class ScikitFeatureLoader(LabelFileLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'scikit_feature_datasets', f'{name}.mat')
        mat = loadmat(path)
        X = mat['X']
        Y = mat['Y']
        Y = Y.squeeze()
        return X, Y


scikit_feature_loader = ScikitFeatureLoader(datasets)
