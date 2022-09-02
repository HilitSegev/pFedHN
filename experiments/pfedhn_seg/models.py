from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torchvision.transforms.functional as TF


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
            spec_norm=False, n_hidden=1):
        super().__init__()

        # self.in_channels = in_channels
        # self.out_dim = out_dim
        # self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        self.ups_0_weight = nn.Linear(hidden_dim, 1024 * 512 * 2 * 2)
        self.ups_0_bias = nn.Linear(hidden_dim, 512)
        self.ups_1_conv_0_weight = nn.Linear(hidden_dim, 512 * 1024 * 3 * 3)
        self.ups_1_conv_1_weight = nn.Linear(hidden_dim, 512)
        self.ups_1_conv_1_bias = nn.Linear(hidden_dim, 512)
        self.ups_1_conv_3_weight = nn.Linear(hidden_dim, 512 * 512 * 3 * 3)
        self.ups_1_conv_4_weight = nn.Linear(hidden_dim, 512)
        self.ups_1_conv_4_bias = nn.Linear(hidden_dim, 512)
        self.ups_2_weight = nn.Linear(hidden_dim, 512 * 256 * 2 * 2)
        self.ups_2_bias = nn.Linear(hidden_dim, 256)
        self.ups_3_conv_0_weight = nn.Linear(hidden_dim, 256 * 512 * 3 * 3)
        self.ups_3_conv_1_weight = nn.Linear(hidden_dim, 256)
        self.ups_3_conv_1_bias = nn.Linear(hidden_dim, 256)
        self.ups_3_conv_3_weight = nn.Linear(hidden_dim, 256 * 256 * 3 * 3)
        self.ups_3_conv_4_weight = nn.Linear(hidden_dim, 256)
        self.ups_3_conv_4_bias = nn.Linear(hidden_dim, 256)
        self.ups_4_weight = nn.Linear(hidden_dim, 256 * 128 * 2 * 2)
        self.ups_4_bias = nn.Linear(hidden_dim, 128)
        self.ups_5_conv_0_weight = nn.Linear(hidden_dim, 128 * 256 * 3 * 3)
        self.ups_5_conv_1_weight = nn.Linear(hidden_dim, 128)
        self.ups_5_conv_1_bias = nn.Linear(hidden_dim, 128)
        self.ups_5_conv_3_weight = nn.Linear(hidden_dim, 128 * 128 * 3 * 3)
        self.ups_5_conv_4_weight = nn.Linear(hidden_dim, 128)
        self.ups_5_conv_4_bias = nn.Linear(hidden_dim, 128)
        self.ups_6_weight = nn.Linear(hidden_dim, 128 * 64 * 2 * 2)
        self.ups_6_bias = nn.Linear(hidden_dim, 64)
        self.ups_7_conv_0_weight = nn.Linear(hidden_dim, 64 * 128 * 3 * 3)
        self.ups_7_conv_1_weight = nn.Linear(hidden_dim, 64)
        self.ups_7_conv_1_bias = nn.Linear(hidden_dim, 64)
        self.ups_7_conv_3_weight = nn.Linear(hidden_dim, 64 * 64 * 3 * 3)
        self.ups_7_conv_4_weight = nn.Linear(hidden_dim, 64)
        self.ups_7_conv_4_bias = nn.Linear(hidden_dim, 64)
        self.downs_0_conv_0_weight = nn.Linear(hidden_dim, 64 * 15 * 3 * 3)
        self.downs_0_conv_1_weight = nn.Linear(hidden_dim, 64)
        self.downs_0_conv_1_bias = nn.Linear(hidden_dim, 64)
        self.downs_0_conv_3_weight = nn.Linear(hidden_dim, 64 * 64 * 3 * 3)
        self.downs_0_conv_4_weight = nn.Linear(hidden_dim, 64)
        self.downs_0_conv_4_bias = nn.Linear(hidden_dim, 64)
        self.downs_1_conv_0_weight = nn.Linear(hidden_dim, 128 * 64 * 3 * 3)
        self.downs_1_conv_1_weight = nn.Linear(hidden_dim, 128)
        self.downs_1_conv_1_bias = nn.Linear(hidden_dim, 128)
        self.downs_1_conv_3_weight = nn.Linear(hidden_dim, 128 * 128 * 3 * 3)
        self.downs_1_conv_4_weight = nn.Linear(hidden_dim, 128)
        self.downs_1_conv_4_bias = nn.Linear(hidden_dim, 128)
        self.downs_2_conv_0_weight = nn.Linear(hidden_dim, 256 * 128 * 3 * 3)
        self.downs_2_conv_1_weight = nn.Linear(hidden_dim, 256)
        self.downs_2_conv_1_bias = nn.Linear(hidden_dim, 256)
        self.downs_2_conv_3_weight = nn.Linear(hidden_dim, 256 * 256 * 3 * 3)
        self.downs_2_conv_4_weight = nn.Linear(hidden_dim, 256)
        self.downs_2_conv_4_bias = nn.Linear(hidden_dim, 256)
        self.downs_3_conv_0_weight = nn.Linear(hidden_dim, 512 * 256 * 3 * 3)
        self.downs_3_conv_1_weight = nn.Linear(hidden_dim, 512)
        self.downs_3_conv_1_bias = nn.Linear(hidden_dim, 512)
        self.downs_3_conv_3_weight = nn.Linear(hidden_dim, 512 * 512 * 3 * 3)
        self.downs_3_conv_4_weight = nn.Linear(hidden_dim, 512)
        self.downs_3_conv_4_bias = nn.Linear(hidden_dim, 512)
        self.bottleneck_conv_0_weight = nn.Linear(hidden_dim, 1024 * 512 * 3 * 3)
        self.bottleneck_conv_1_weight = nn.Linear(hidden_dim, 1024)
        self.bottleneck_conv_1_bias = nn.Linear(hidden_dim, 1024)
        self.bottleneck_conv_3_weight = nn.Linear(hidden_dim, 1024 * 1024 * 3 * 3)
        self.bottleneck_conv_4_weight = nn.Linear(hidden_dim, 1024)
        self.bottleneck_conv_4_bias = nn.Linear(hidden_dim, 1024)
        self.final_conv_weight = nn.Linear(hidden_dim, 15 * 64 * 1 * 1)
        self.final_conv_bias = nn.Linear(hidden_dim, 15)

        if spec_norm:
            self.ups_0_weight = spectral_norm(self.ups_0_weight)
            self.ups_0_bias = spectral_norm(self.ups_0_bias)
            self.ups_1_conv_0_weight = spectral_norm(self.ups_1_conv_0_weight)
            self.ups_1_conv_1_weight = spectral_norm(self.ups_1_conv_1_weight)
            self.ups_1_conv_1_bias = spectral_norm(self.ups_1_conv_1_bias)
            self.ups_1_conv_3_weight = spectral_norm(self.ups_1_conv_3_weight)
            self.ups_1_conv_4_weight = spectral_norm(self.ups_1_conv_4_weight)
            self.ups_1_conv_4_bias = spectral_norm(self.ups_1_conv_4_bias)
            self.ups_2_weight = spectral_norm(self.ups_2_weight)
            self.ups_2_bias = spectral_norm(self.ups_2_bias)
            self.ups_3_conv_0_weight = spectral_norm(self.ups_3_conv_0_weight)
            self.ups_3_conv_1_weight = spectral_norm(self.ups_3_conv_1_weight)
            self.ups_3_conv_1_bias = spectral_norm(self.ups_3_conv_1_bias)
            self.ups_3_conv_3_weight = spectral_norm(self.ups_3_conv_3_weight)
            self.ups_3_conv_4_weight = spectral_norm(self.ups_3_conv_4_weight)
            self.ups_3_conv_4_bias = spectral_norm(self.ups_3_conv_4_bias)
            self.ups_4_weight = spectral_norm(self.ups_4_weight)
            self.ups_4_bias = spectral_norm(self.ups_4_bias)
            self.ups_5_conv_0_weight = spectral_norm(self.ups_5_conv_0_weight)
            self.ups_5_conv_1_weight = spectral_norm(self.ups_5_conv_1_weight)
            self.ups_5_conv_1_bias = spectral_norm(self.ups_5_conv_1_bias)
            self.ups_5_conv_3_weight = spectral_norm(self.ups_5_conv_3_weight)
            self.ups_5_conv_4_weight = spectral_norm(self.ups_5_conv_4_weight)
            self.ups_5_conv_4_bias = spectral_norm(self.ups_5_conv_4_bias)
            self.ups_6_weight = spectral_norm(self.ups_6_weight)
            self.ups_6_bias = spectral_norm(self.ups_6_bias)
            self.ups_7_conv_0_weight = spectral_norm(self.ups_7_conv_0_weight)
            self.ups_7_conv_1_weight = spectral_norm(self.ups_7_conv_1_weight)
            self.ups_7_conv_1_bias = spectral_norm(self.ups_7_conv_1_bias)
            self.ups_7_conv_3_weight = spectral_norm(self.ups_7_conv_3_weight)
            self.ups_7_conv_4_weight = spectral_norm(self.ups_7_conv_4_weight)
            self.ups_7_conv_4_bias = spectral_norm(self.ups_7_conv_4_bias)
            self.downs_0_conv_0_weight = spectral_norm(self.downs_0_conv_0_weight)
            self.downs_0_conv_1_weight = spectral_norm(self.downs_0_conv_1_weight)
            self.downs_0_conv_1_bias = spectral_norm(self.downs_0_conv_1_bias)
            self.downs_0_conv_3_weight = spectral_norm(self.downs_0_conv_3_weight)
            self.downs_0_conv_4_weight = spectral_norm(self.downs_0_conv_4_weight)
            self.downs_0_conv_4_bias = spectral_norm(self.downs_0_conv_4_bias)
            self.downs_1_conv_0_weight = spectral_norm(self.downs_1_conv_0_weight)
            self.downs_1_conv_1_weight = spectral_norm(self.downs_1_conv_1_weight)
            self.downs_1_conv_1_bias = spectral_norm(self.downs_1_conv_1_bias)
            self.downs_1_conv_3_weight = spectral_norm(self.downs_1_conv_3_weight)
            self.downs_1_conv_4_weight = spectral_norm(self.downs_1_conv_4_weight)
            self.downs_1_conv_4_bias = spectral_norm(self.downs_1_conv_4_bias)
            self.downs_2_conv_0_weight = spectral_norm(self.downs_2_conv_0_weight)
            self.downs_2_conv_1_weight = spectral_norm(self.downs_2_conv_1_weight)
            self.downs_2_conv_1_bias = spectral_norm(self.downs_2_conv_1_bias)
            self.downs_2_conv_3_weight = spectral_norm(self.downs_2_conv_3_weight)
            self.downs_2_conv_4_weight = spectral_norm(self.downs_2_conv_4_weight)
            self.downs_2_conv_4_bias = spectral_norm(self.downs_2_conv_4_bias)
            self.downs_3_conv_0_weight = spectral_norm(self.downs_3_conv_0_weight)
            self.downs_3_conv_1_weight = spectral_norm(self.downs_3_conv_1_weight)
            self.downs_3_conv_1_bias = spectral_norm(self.downs_3_conv_1_bias)
            self.downs_3_conv_3_weight = spectral_norm(self.downs_3_conv_3_weight)
            self.downs_3_conv_4_weight = spectral_norm(self.downs_3_conv_4_weight)
            self.downs_3_conv_4_bias = spectral_norm(self.downs_3_conv_4_bias)
            self.bottleneck_conv_0_weight = spectral_norm(self.bottleneck_conv_0_weight)
            self.bottleneck_conv_1_weight = spectral_norm(self.bottleneck_conv_1_weight)
            self.bottleneck_conv_1_bias = spectral_norm(self.bottleneck_conv_1_bias)
            self.bottleneck_conv_3_weight = spectral_norm(self.bottleneck_conv_3_weight)
            self.bottleneck_conv_4_weight = spectral_norm(self.bottleneck_conv_4_weight)
            self.bottleneck_conv_4_bias = spectral_norm(self.bottleneck_conv_4_bias)
            self.final_conv_weight = spectral_norm(self.final_conv_weight)
            self.final_conv_bias = spectral_norm(self.final_conv_bias)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            'ups.0.weight': self.ups_0_weight(features).view(1024, 512, 2, 2),
            'ups.0.bias': self.ups_0_bias(features).view(512),
            'ups.1.conv.0.weight': self.ups_1_conv_0_weight(features).view(512, 1024, 3, 3),
            'ups.1.conv.1.weight': self.ups_1_conv_1_weight(features).view(512),
            'ups.1.conv.1.bias': self.ups_1_conv_1_bias(features).view(512),
            'ups.1.conv.3.weight': self.ups_1_conv_3_weight(features).view(512, 512, 3, 3),
            'ups.1.conv.4.weight': self.ups_1_conv_4_weight(features).view(512),
            'ups.1.conv.4.bias': self.ups_1_conv_4_bias(features).view(512),
            'ups.2.weight': self.ups_2_weight(features).view(512, 256, 2, 2),
            'ups.2.bias': self.ups_2_bias(features).view(256),
            'ups.3.conv.0.weight': self.ups_3_conv_0_weight(features).view(256, 512, 3, 3),
            'ups.3.conv.1.weight': self.ups_3_conv_1_weight(features).view(256),
            'ups.3.conv.1.bias': self.ups_3_conv_1_bias(features).view(256),
            'ups.3.conv.3.weight': self.ups_3_conv_3_weight(features).view(256, 256, 3, 3),
            'ups.3.conv.4.weight': self.ups_3_conv_4_weight(features).view(256),
            'ups.3.conv.4.bias': self.ups_3_conv_4_bias(features).view(256),
            'ups.4.weight': self.ups_4_weight(features).view(256, 128, 2, 2),
            'ups.4.bias': self.ups_4_bias(features).view(128),
            'ups.5.conv.0.weight': self.ups_5_conv_0_weight(features).view(128, 256, 3, 3),
            'ups.5.conv.1.weight': self.ups_5_conv_1_weight(features).view(128),
            'ups.5.conv.1.bias': self.ups_5_conv_1_bias(features).view(128),
            'ups.5.conv.3.weight': self.ups_5_conv_3_weight(features).view(128, 128, 3, 3),
            'ups.5.conv.4.weight': self.ups_5_conv_4_weight(features).view(128),
            'ups.5.conv.4.bias': self.ups_5_conv_4_bias(features).view(128),
            'ups.6.weight': self.ups_6_weight(features).view(128, 64, 2, 2),
            'ups.6.bias': self.ups_6_bias(features).view(64),
            'ups.7.conv.0.weight': self.ups_7_conv_0_weight(features).view(64, 128, 3, 3),
            'ups.7.conv.1.weight': self.ups_7_conv_1_weight(features).view(64),
            'ups.7.conv.1.bias': self.ups_7_conv_1_bias(features).view(64),
            'ups.7.conv.3.weight': self.ups_7_conv_3_weight(features).view(64, 64, 3, 3),
            'ups.7.conv.4.weight': self.ups_7_conv_4_weight(features).view(64),
            'ups.7.conv.4.bias': self.ups_7_conv_4_bias(features).view(64),
            'downs.0.conv.0.weight': self.downs_0_conv_0_weight(features).view(64, 15, 3, 3),
            'downs.0.conv.1.weight': self.downs_0_conv_1_weight(features).view(64),
            'downs.0.conv.1.bias': self.downs_0_conv_1_bias(features).view(64),
            'downs.0.conv.3.weight': self.downs_0_conv_3_weight(features).view(64, 64, 3, 3),
            'downs.0.conv.4.weight': self.downs_0_conv_4_weight(features).view(64),
            'downs.0.conv.4.bias': self.downs_0_conv_4_bias(features).view(64),
            'downs.1.conv.0.weight': self.downs_1_conv_0_weight(features).view(128, 64, 3, 3),
            'downs.1.conv.1.weight': self.downs_1_conv_1_weight(features).view(128),
            'downs.1.conv.1.bias': self.downs_1_conv_1_bias(features).view(128),
            'downs.1.conv.3.weight': self.downs_1_conv_3_weight(features).view(128, 128, 3, 3),
            'downs.1.conv.4.weight': self.downs_1_conv_4_weight(features).view(128),
            'downs.1.conv.4.bias': self.downs_1_conv_4_bias(features).view(128),
            'downs.2.conv.0.weight': self.downs_2_conv_0_weight(features).view(256, 128, 3, 3),
            'downs.2.conv.1.weight': self.downs_2_conv_1_weight(features).view(256),
            'downs.2.conv.1.bias': self.downs_2_conv_1_bias(features).view(256),
            'downs.2.conv.3.weight': self.downs_2_conv_3_weight(features).view(256, 256, 3, 3),
            'downs.2.conv.4.weight': self.downs_2_conv_4_weight(features).view(256),
            'downs.2.conv.4.bias': self.downs_2_conv_4_bias(features).view(256),
            'downs.3.conv.0.weight': self.downs_3_conv_0_weight(features).view(512, 256, 3, 3),
            'downs.3.conv.1.weight': self.downs_3_conv_1_weight(features).view(512),
            'downs.3.conv.1.bias': self.downs_3_conv_1_bias(features).view(512),
            'downs.3.conv.3.weight': self.downs_3_conv_3_weight(features).view(512, 512, 3, 3),
            'downs.3.conv.4.weight': self.downs_3_conv_4_weight(features).view(512),
            'downs.3.conv.4.bias': self.downs_3_conv_4_bias(features).view(512),
            'bottleneck.conv.0.weight': self.bottleneck_conv_0_weight(features).view(1024, 512, 3, 3),
            'bottleneck.conv.1.weight': self.bottleneck_conv_1_weight(features).view(1024),
            'bottleneck.conv.1.bias': self.bottleneck_conv_1_bias(features).view(1024),
            'bottleneck.conv.3.weight': self.bottleneck_conv_3_weight(features).view(1024, 1024, 3, 3),
            'bottleneck.conv.4.weight': self.bottleneck_conv_4_weight(features).view(1024),
            'bottleneck.conv.4.bias': self.bottleneck_conv_4_bias(features).view(1024),
            'final_conv.weight': self.final_conv_weight(features).view(15, 64, 1, 1),
            'final_conv.bias': self.final_conv_bias(features).view(15)
        })
        return weights


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CNNTarget(nn.Module):
    """
    This is the Target Network to be used with the HyperNetwork. The current implementation is of the UNet architecture.
    """

    def __init__(
            self, in_channels=15, out_channels=15, features=[64, 128, 256, 512],
    ):
        super(CNNTarget, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

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
    model = CNNTarget(in_channels=15, out_channels=15)
    preds = model(x)
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
