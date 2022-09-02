from typing import Optional

import torch
import torch.nn as nn

from .metrics import dice_score


class DiceWithLogitsLoss(nn.Module):
    """Combines Sigmoid layer and DiceLoss."""

    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None, reduction: str = "mean") -> None:
        super(DiceWithLogitsLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        num_classes = y_pred.shape[1]
        # apply sigmoid / softmax activation depending on binary classification or multiclass
        if num_classes == 1:
            y_pred = y_pred.sigmoid()
        else:
            y_pred = y_pred.softmax(1)

        dsc = dice_score(y_pred, y_true, self.ignore_index, self.smooth)

        if self.reduction == "mean":
            return (1 - dsc).mean()
        elif self.reduction == "sum":
            return (1 - dsc).sum()
        else:
            raise NotImplementedError


class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, ignore_index: Optional[int] = None, *args, **kwargs) -> None:
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.dice_loss = DiceWithLogitsLoss(smooth, ignore_index)
        self.bce_loss = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MSEWithLogitsLoss, self).__init__()
        self.mse_loss = nn.MSELoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(y_pred.sigmoid(), y_true)


def kl_std_div(mean: torch.Tensor, v_log: torch.Tensor) -> torch.Tensor:
    return (v_log.exp() + mean**2 - 1 - v_log).mean()
