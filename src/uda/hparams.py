from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from .config import Config
from .losses import _LossCriterion


class LossCriterion(str, Enum):
    """Supported loss functions."""

    Dice = _LossCriterion.Dice.name
    DiceBCE = _LossCriterion.DiceBCE.name
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
    `loss_kwargs`: Additional arguments for loss function
    `sf_dice_tolerance`: Tolerance value for surface dice
    `early_stopping`: Use early stopping
    `early_stopping_patience`: Number of epochs to wait before early stopping
    """

    epochs: int = 10
    criterion: LossCriterion = LossCriterion.BCE
    optimizer: Optimizer = Optimizer.Adam
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    val_batch_size: int = 1
    loss_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    sf_dice_tolerance: float = 1
    early_stopping: bool = True
    early_stopping_patience: Optional[int] = 5

    def get_optimizer(self) -> Type[torch.optim.Optimizer]:
        return _Optimizer[self.optimizer].value

    def get_criterion(self) -> Type[nn.Module]:
        return _LossCriterion[self.criterion].value(**self.loss_kwargs)
