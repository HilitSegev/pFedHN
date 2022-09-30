import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    """
    implementation based on https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """

    def __init__(self, weight=None, size_average=True, dice_weight=1):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss * self.dice_weight

        return Dice_BCE
