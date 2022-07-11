from sklearnex import patch_sklearn
patch_sklearn()
from feature_selection import lfs, ufs_sp
from data.toy.load import load_toy


def main():
    X, y = load_toy()
    alpha = lfs(X, y, gamma=-1)
    w = ufs_sp(X.to_numpy(), 1e3, 1.4, 20, 20)


if __name__ == '__main__':
    main()
