from enum import Enum

import torch
import torch.nn as nn

from .metrics import dice_score


class DiceLoss(nn.Module):
    def __init__(self, square_denom: bool = False) -> None:
        super(DiceLoss, self).__init__()
        self.square_denom = square_denom

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return dice_loss(y_pred, y_true, square_denom=self.square_denom)


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, square_denom: bool = False) -> torch.Tensor:
    return 1 - dice_score(y_pred, y_true, square_denom=square_denom)


class _LossCriterion(Enum):
    Dice = DiceLoss
    BCE = nn.BCELoss
