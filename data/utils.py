import os


class Loader:
    def _load(self, name, parent=''):
        raise NotImplemented

    def is_dataset_available(self, name, parent=''):
        raise NotImplemented

    def load(self, name, parent=''):
        if not self.is_dataset_available(name, parent):
            raise ValueError(name + ' is not an available dataset')
        return self._load(name, parent)


class FileLoader(Loader):
    def __init__(self, label_columns):
        self._label_columns = label_columns

    def is_dataset_available(self, name, parent=''):
        return name in self._label_columns


class DirLoader(Loader):
    def __init__(self, *loaders, directory=''):
        self._loaders = loaders
        self._directory = directory

    def _load(self, name, parent=''):
        child_dir = os.path.join(parent, self._directory)
        for loader in self._loaders:
            if loader.is_dataset_available(name, child_dir):
                return loader.load(name, child_dir)

    def is_dataset_available(self, name, parent=''):
        child_dir = os.path.join(parent, self._directory)
        return any(loader.is_dataset_available(name, child_dir) for loader in self._loaders)


def split_X_y(df, column):
    y = df[column]
    X = df.drop(column, axis=1)
    return X, y
