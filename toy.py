from sklearnex import patch_sklearn
patch_sklearn()
from feature_selection import lfs, ufs_sp_l_2_1
from data import data_loader


def main():
    X, y = data_loader.load('toy')
    # alpha = lfs(X.to_numpy(), y.to_numpy())
    w = ufs_sp_l_2_1(X.to_numpy(), y.to_numpy())


if __name__ == '__main__':
    main()
