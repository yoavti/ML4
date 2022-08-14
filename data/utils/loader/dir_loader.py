import os
from .loader import Loader


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

    def available_datasets(self):
        return sum([loader.available_datasets() for loader in self._loaders], [])
