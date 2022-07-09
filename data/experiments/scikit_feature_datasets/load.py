from scipy.io import loadmat
import os

DIR = os.path.join('data', 'experiments', 'scikit_feature_datasets')
selected_datasets = {'lung_small', 'colon', 'Yale', 'nci9', 'PCMAC'}


def load_scikit_feature(name):
    path = os.path.join(DIR, f'{name}.mat')
    mat = loadmat(path)
    X = mat['X']
    Y = mat['Y']
    Y = Y.squeeze()
    return X, Y
