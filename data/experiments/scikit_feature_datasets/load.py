from scipy.io import loadmat
import os
from data.utils import FileLoader


class ScikitFeatureLoader(FileLoader):
    def _load(self, name, parent=''):
        path = os.path.join(parent, 'scikit_feature_datasets', f'{name}.mat')
        mat = loadmat(path)
        X = mat['X']
        Y = mat['Y']
        Y = Y.squeeze()
        return X, Y


scikit_feature_loader = ScikitFeatureLoader({name: None for name in ['lung_small', 'colon', 'Yale', 'nci9', 'PCMAC']})
