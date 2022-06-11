from enum import Enum
from typing import Any, Dict, Tuple, Type

import torch
import torch.nn as nn


class _LossCriterion(Enum):
    ...


class LossCriterion(str, Enum):
    ...


class VAELoss(nn.Module):
    def __init__(
        self,
        rec_loss: LossCriterion,
        rec_loss_kwargs: Dict[str, Any] = {},
        lamda: float = 1.0,
        return_sum: bool = False,
    ) -> None:
        super(VAELoss, self).__init__()
        self.lamda = lamda
        self.rec_loss_fn = _LossCriterion[rec_loss].value(**rec_loss_kwargs)
        self.kl_loss_fn = KLLoss(lamda)
        self.return_sum = return_sum

    def forward(self, out: Tuple[torch.Tensor, ...], x_true: torch.Tensor) -> torch.Tensor:
        x_rec, mean, v_log = out

        rec_l = self.rec_loss_fn(x_rec, x_true)  # * np.prod(cc359_dataset.imsize)
        kl_l = self.kl_loss_fn(mean, v_log)

        if self.return_sum:
            return rec_l + self.lamda * kl_l
        else:
            return rec_l, self.lamda * kl_l


class KLLoss(nn.Module):
    def __init__(self, lamda: float = 1.0) -> None:
        super(KLLoss, self).__init__()
        self.lamda = lamda

    def forward(self, mean: torch.Tensor, v_log: torch.Tensor) -> torch.Tensor:
        return self.lamda * kl_loss(mean, v_log)


class DiceWithLogitsLoss(nn.Module):
    """Combines Sigmoid layer and DiceLoss."""

    def __init__(
        self,
        sigmoid: bool = True,
        softmax: bool = False,
        squared_pred: bool = True,
        smooth_nr: float = 1,
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
    smooth_nr: float = 1,
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


def kl_loss(mean: torch.Tensor, v_log: torch.Tensor) -> torch.Tensor:
    return (v_log.exp() + mean**2 - 1 - v_log).mean()


class _LossCriterion(Enum):
    Dice = DiceWithLogitsLoss
    DiceBCE = DiceBCEWithLogitsLoss
    BCE = nn.BCEWithLogitsLoss
    VAELoss = VAELoss


class LossCriterion(str, Enum):
    """Supported loss functions."""

    Dice = _LossCriterion.Dice.name
    DiceBCE = _LossCriterion.DiceBCE.name
    BCE = _LossCriterion.BCE.name
    VAELoss = _LossCriterion.VAELoss.name


class _Optimizer(Enum):
    Adam = torch.optim.Adam


class Optimizer(str, Enum):
    """Supported loss functions."""

    Adam = _Optimizer.Adam.name


def loss_fn(criterion_name: str) -> Type[nn.Module]:
    return _LossCriterion[criterion_name].value


def optimizer_cls(optimizer_name: str) -> Type[torch.optim.Optimizer]:
    return _Optimizer[optimizer_name].value
