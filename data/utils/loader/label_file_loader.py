from .loader import Loader


class LabelFileLoader(Loader):
    def __init__(self, datasets):
        self._datasets = set(datasets)

    def is_dataset_available(self, name, parent=''):
        return name in self._datasets
