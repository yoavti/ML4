class Loader:
    def _load(self, name, parent=''):
        raise NotImplemented

    def is_dataset_available(self, name, parent=''):
        raise NotImplemented

    def load(self, name, parent=''):
        if not self.is_dataset_available(name, parent):
            raise ValueError(name + ' is not an available dataset')
        return self._load(name, parent)
