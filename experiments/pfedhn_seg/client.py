import torch
from monai.data import DataLoader

from experiments.pfedhn_seg import monai_datasets


class Client:
    def __init__(self, client_id, client_name, client_data_path, client_net, client_batchnorm_dict):
        self.id = client_id
        self.name = client_name
        self.data_path = client_data_path
        self.net = client_net
        # batch_norm layers dict?
        self.batchnorm_dict = {}

    # get data loaders
    def get_data_loaders(self, batch_size):
        datasets = monai_datasets.get_datasets(self.name, self.data_path)
        train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets[1], batch_size=1, shuffle=False)
        test_loader = DataLoader(datasets[2], batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    def update_net(self, new_net):
        self.net = new_net

    def update_batchnorm_dict(self, new_batchnorm_dict):
        self.batchnorm_dict = new_batchnorm_dict
