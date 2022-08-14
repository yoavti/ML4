class Loader:
    def _load(self, name, parent=''):
        raise NotImplemented

    def is_dataset_available(self, name):
        raise NotImplemented

    def load(self, name, parent=''):
        if not self.is_dataset_available(name):
            raise ValueError(name + ' is not an available dataset')
        return self._load(name, parent)

    def available_datasets(self):
        raise NotImplemented

    def dataset_size(self, name):
        raise NotImplemented
