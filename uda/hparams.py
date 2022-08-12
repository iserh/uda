from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .config import Config
from .losses import DiceBCEWithLogitsLoss, DiceWithLogitsLoss, MSEWithLogitsLoss


def get_loss_cls(name: str) -> type[torch.nn.Module]:
    if name == "Dice":
        return DiceWithLogitsLoss
    elif name == "DiceBCE":
        return DiceBCEWithLogitsLoss
    elif name == "BCE":
        return torch.nn.BCEWithLogitsLoss
    elif name == "MSE":
        return MSEWithLogitsLoss
    else:
        NotImplementedError


def get_optimizer_cls(name: str) -> type[torch.optim.Optimizer]:
    if name == "Adam":
        return torch.optim.Adam
    else:
        NotImplementedError


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
    `early_stopping_patience`: Number of epochs to wait before early stopping
    `vae_beta`: Beta value for KL loss
    """

    epochs: int = 10
    criterion: str = "BCE"
    optimizer: str = "Adam"
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    val_batch_size: int = 1
    loss_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)
    sf_dice_tolerance: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    vae_beta: Optional[float] = None
    vae_lambd: Optional[float] = None
