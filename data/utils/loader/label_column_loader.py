from .loader import Loader


class LabelColumnLoader(Loader):
    def __init__(self, label_columns, dataset_sizes):
        self._label_columns = label_columns
        self._dataset_sizes = dataset_sizes

    def is_dataset_available(self, name):
        return name in self._label_columns

    def available_datasets(self):
        return list(self._label_columns)

    def dataset_size(self, name):
        return self._dataset_sizes[name]
