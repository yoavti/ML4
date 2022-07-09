from feature_selection import lfs, ufs_sp
from data.toy.load import load_toy
from sklearnex import patch_sklearn
patch_sklearn()


def main():
    X, y = load_toy()
    alpha = lfs(X, y, gamma=-1)
    w = ufs_sp(X.to_numpy(), 1e3, 2, 100, 100)


if __name__ == '__main__':
    main()
