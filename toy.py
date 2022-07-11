from sklearnex import patch_sklearn
patch_sklearn()
from feature_selection import lfs, ufs_sp
from data.toy.load import load_toy


def main():
    X, y = load_toy()
    alpha = lfs(X.to_numpy(), y.to_numpy())
    w = ufs_sp(X.to_numpy(), y.to_numpy())


if __name__ == '__main__':
    main()
