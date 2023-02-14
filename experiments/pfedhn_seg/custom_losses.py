import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


class Dice(nn.Module):
    def __init__(self, thresh: float = 0.5):
        """Dice coefficient."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None, smooth: float = 1):
        # assert 0 <= targets.min().item() <= targets.max().item() <= 1
        # Binarize prediction
        inputs = torch.where(inputs < self.thresh, 0, 1)
        batch_size = targets.shape[0]
        intersection = torch.logical_and(inputs, targets)
        intersection = intersection.view(batch_size, -1).sum(-1)
        targets_area = targets.view(batch_size, -1).sum(-1)
        inputs_area = inputs.view(batch_size, -1).sum(-1)
        dice = (2. * intersection + smooth) / (inputs_area + targets_area + smooth)

        if weights is not None:
            assert weights.shape == dice.shape, \
                f'"weights" must be in shape of "{dice.shape}"'
            return (dice * weights).sum()

        return dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, thresh: float = 0.5, dice_weight: float = 1, bce_weight: float = 1):
        """Dice loss + binary cross-entropy loss."""
        super().__init__()
        assert 0 < thresh < 1, f"'thresh' must be in range (0, 1)"
        self.thresh = thresh
        self.dice = Dice(self.thresh)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.__name__ = 'DiceBCELoss'

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None, smooth: float = 1):
        batch_size = inputs.shape[0]

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        bce = F.binary_cross_entropy(inputs, targets, reduce=False)
        bce = bce.reshape(batch_size, -1).mean(-1)

        if weights is not None:
            assert weights.shape == bce.shape, \
                f'"weights" must be in shape of "{bce.shape}"'
            bce = (bce * weights).sum()
        else:
            bce = bce.mean()

        dice_loss = 1 - self.dice(inputs, targets, weights, smooth)
        dice_bce = self.bce_weight * bce + self.dice_weight * dice_loss
        return dice_bce

# class DiceBCELossFullBatch(nn.Module):
#     """
#     implementation based on https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
#     """
#
#     def __init__(self, weight=None, size_average=True, dice_weight=1):
#         super(DiceBCELossFullBatch, self).__init__()
#         self.dice_weight = dice_weight
#
#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.sigmoid(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
#         Dice_BCE = BCE + dice_loss * self.dice_weight
#
#         return Dice_BCE
