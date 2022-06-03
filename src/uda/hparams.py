from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type

import torch
import torch.nn as nn

from .config import Config
from .losses import _LossCriterion


class LossCriterion(str, Enum):
    """Supported loss functions."""

    Dice = _LossCriterion.Dice.name
    BCE = _LossCriterion.BCE.name


class _Optimizer(Enum):
    Adam = torch.optim.Adam


class Optimizer(str, Enum):
    """Supported loss functions."""

    Adam = _Optimizer.Adam.name


@dataclass
class HParams(Config):
    """Configuration for Hyperparameters.

    `epochs`: Number of epochs for training
    `criterion`: Loss function
    `optim`: Optimizer Name
    `learning_rate` : Learning rate
    `train_batch_size`: Batch Size for training
    `train_batch_size`: Batch Size for validation
    `sdice_tolerance`: Tolerance value for surface dice
    `square_dice_denom`: Square the denominator when using dice loss. Defaults to `True`
    """

    epochs: int = 10
    criterion: LossCriterion = LossCriterion.BCE
    optimizer: Optimizer = Optimizer.Adam
    learning_rate: float = 1e-4
    train_batch_size: int = 4
    val_batch_size: int = 4
    sdice_tolerance: float = 0.5
    square_dice_denom: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.square_dice_denom is None and self.criterion == LossCriterion.Dice:
            self.square_dice_denom = True

    def get_optimizer(self) -> Type[torch.optim.Optimizer]:
        return _Optimizer[self.optimizer].value

    def get_criterion(self) -> Type[nn.Module]:
        return _LossCriterion[self.criterion].value
