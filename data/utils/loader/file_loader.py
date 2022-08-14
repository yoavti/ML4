from .loader import Loader


class FileLoader(Loader):
    def __init__(self, datasets, dataset_sizes):
        self._datasets = datasets
        self._dataset_sizes = dataset_sizes

    def is_dataset_available(self, name):
        return name in self._datasets

    def available_datasets(self):
        return list(self._datasets)

    def dataset_size(self, name):
        return self._dataset_sizes[name]
