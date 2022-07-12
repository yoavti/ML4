from .loader import Loader


class FileLoader(Loader):
    def __init__(self, label_columns):
        self._label_columns = label_columns

    def is_dataset_available(self, name, parent=''):
        return name in self._label_columns
