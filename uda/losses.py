from enum import Enum

import torch
import torch.nn as nn


class _LossCriterion(Enum):
    ...


class LossCriterion(str, Enum):
    ...


class DiceWithLogitsLoss(nn.Module):
    """Combines Sigmoid layer and DiceLoss."""

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth: float = 1,
    ) -> None:
        super(DiceWithLogitsLoss, self).__init__()
        self.squared_pred = squared_pred
        self.smooth = smooth
        if sigmoid:
            self.act = torch.sigmoid
        elif softmax:
            # self.act = torch.softmax
            raise NotImplementedError()
        else:
            self.act = nn.Identity()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = self.act(y_pred)
        return dice_loss(y_pred, y_true, self.squared_pred, self.smooth)


class DiceBCEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.dice_loss = DiceWithLogitsLoss(*args, **kwargs)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.bce_loss(y_pred, y_true) + self.dice_loss(y_pred, y_true)


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MSEWithLogitsLoss, self).__init__()
        self.mse_loss = nn.MSELoss(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(y_pred.sigmoid(), y_true)


def dice_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    squared_pred: bool = True,
    smooth: float = 1,
) -> torch.Tensor:
    # flatten
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    num = 2 * (y_pred * y_true).sum() + smooth

    if squared_pred:
        denom = (y_pred**2).sum() + (y_true**2).sum() + smooth
    else:
        denom = y_pred.sum() + y_true.sum() + smooth

    return 1 - num / denom


def kl_loss(mean: torch.Tensor, v_log: torch.Tensor) -> torch.Tensor:
    return (v_log.exp() + mean**2 - 1 - v_log).mean()


class _LossCriterion(Enum):  # noqa: F811
    Dice = DiceWithLogitsLoss
    DiceBCE = DiceBCEWithLogitsLoss
    BCE = nn.BCEWithLogitsLoss
    MSE = MSEWithLogitsLoss


class LossCriterion(str, Enum):  # noqa: F811
    """Supported loss functions."""

    Dice = _LossCriterion.Dice.name
    DiceBCE = _LossCriterion.DiceBCE.name
    BCE = _LossCriterion.BCE.name
    MSE = _LossCriterion.MSE.name


class _Optimizer(Enum):
    Adam = torch.optim.Adam


class Optimizer(str, Enum):
    """Supported loss functions."""

    Adam = _Optimizer.Adam.name


def get_criterion(criterion_name: str) -> type[nn.Module]:
    return _LossCriterion[criterion_name].value


def optimizer_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    return _Optimizer[optimizer_name].value
