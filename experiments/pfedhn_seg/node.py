from experiments.dataset import gen_random_loaders, gen_loaders


class BaseNodes:
    def __init__(
            self,
            data_names,
            data_path,
            batch_size=128
    ):
        self.data_names = data_names
        self.data_path = data_path

        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = gen_loaders(self.data_names,
                                                                              self.data_path,
                                                                              self.batch_size)

    def __len__(self):
        return len(self.data_names)
