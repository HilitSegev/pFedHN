import math
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torchvision.transforms.functional as TF


class CNNHyper(nn.Module):
    def __init__(
            self,
            n_nodes,
            embedding_dim,
            model,
            out_layers=None,
            in_channels=1,
            hidden_dim=100,
            n_hidden=1,
            spec_norm=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model = model
        if embedding_dim == -1:
            embedding_dim = n_nodes
            self.embeddings = lambda x: F.one_hot(x, num_classes=n_nodes).float()
        else:
            self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        if out_layers is None:
            self.out_layers = model.state_dict().keys()
        else:
            self.out_layers = out_layers

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )
        layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        self.predicted_layers = {
            k: nn.Linear(hidden_dim, math.prod(tuple(layer.size()))) for k, layer in model.state_dict().items() if
            (k.endswith('weight') or k.endswith('bias')) and k in self.out_layers
        }

        if spec_norm:
            for k, layer in self.predicted_layers.items():
                self.predicted_layers[k] = spectral_norm(layer)

        for key in self.predicted_layers:
            setattr(self, key, self.predicted_layers[key])

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict()
        for key in self.predicted_layers:
            layer = getattr(self, key)
            weights[key] = layer(features).view(self.model.state_dict()[key].size())

        return weights


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super(DoubleConv, self).__init__()
        # Changed order of BN and RELU
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class CNNTarget(nn.Module):
    """
    This is the Target Network to be used with the HyperNetwork. The current implementation is of the UNet architecture.
    """

    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dropout_p=0.5
    ):
        super(CNNTarget, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_p=dropout_p))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature, dropout_p))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p=dropout_p)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 15, 161, 161))
    model = CNNTarget(in_channels=1, out_channels=1, features=[16, 32, 64, 128])
    preds = model(x)

    test_hyper = CNNHyper(4, 1, hidden_dim=100,
                          n_hidden=10)
    # assert preds.shape == x.shape

    layer_to_size = {
        k: tuple(v.size())
        for k, v in model.state_dict().items() if k.endswith('weight') or k.endswith('bias')
    }

    layer_to_property_name = {
        k: f"self.{k.replace('.', '_')}"
        for k, v in model.state_dict().items() if k.endswith('weight') or k.endswith('bias')
    }

    print('\n'.join(f"{k} {v}" for k, v in layer_to_property_name.items()))

    layer_to_property = {
        k: f"{layer_to_property_name[k]} = nn.Linear(hidden_dim, {' * '.join([str(s) for s in layer_to_size[k]])})"
        for k, v in model.state_dict().items() if k.endswith('weight') or k.endswith('bias')
    }

    print('\n'.join(layer_to_property.values()))

    layer_to_spec_norm = {
        k: f"{layer_to_property_name[k]} = spectral_norm({layer_to_property_name[k]})"
        for k, v in layer_to_property.items()
    }

    print('\n'.join(layer_to_spec_norm.values()))

    layer_to_weights = {
        k: f"'{k}': {layer_to_property_name[k]}(features).view({', '.join([str(s) for s in layer_to_size[k]])})"
        for k, v in model.state_dict().items() if k.endswith('weight') or k.endswith('bias')
    }

    print(',\n'.join(layer_to_weights.values()))


if __name__ == "__main__":
    test()
