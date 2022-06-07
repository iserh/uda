from enum import Enum

import torch
import torch.nn as nn


class DiceWithLogitsLoss(nn.Module):
    """Combines Sigmoid layer and DiceLoss."""

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth_nr: float = 0,
        smooth_dr: float = 1,
    ) -> None:
        super(DiceWithLogitsLoss, self).__init__()
        self.squared_pred = squared_pred
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        if sigmoid:
            self.act = torch.sigmoid
        elif softmax:
            # self.act = torch.softmax
            raise NotImplementedError()
        else:
            self.act = nn.Identity()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = self.act(y_pred)
        return dice_loss(y_pred, y_true, self.squared_pred, self.smooth_nr, self.smooth_dr)


class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.dice_loss = DiceWithLogitsLoss(*args, **kwargs)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)


def dice_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    squared_pred: bool = True,
    smooth_nr: float = 0,
    smooth_dr: float = 1,
) -> torch.Tensor:
    # flatten
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    num = 2 * (y_pred * y_true).sum() + smooth_nr

    if squared_pred:
        denom = (y_pred**2).sum() + (y_true**2).sum() + smooth_dr
    else:
        denom = y_pred.sum() + y_true.sum() + smooth_dr

    return 1 - num / denom


class _LossCriterion(Enum):
    Dice = DiceWithLogitsLoss
    DiceBCE = DiceBCEWithLogitsLoss
    BCE = nn.BCEWithLogitsLoss
