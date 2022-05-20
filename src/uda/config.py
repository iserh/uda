import json
from typing import Callable, Type

import torch.optim as optim

import uda.losses as losses


class Config:
    """Configuration Class."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = cls(**json.load(f))
        return config


class HParamsConfig(Config):
    """Configuration for Hyperparameters."""

    def __init__(
        self,
        epochs: int = 10,
        criterion: str = "dice_loss",
        learning_rate: float = 1e-4,
        optim: str = "adam",
        batch_size: int = 4,
        test_interval: int = 100,
    ) -> None:
        """Args:
        `epochs`: Number of epochs for training
        `criterion`: Loss function
        `learning_rate` : Learning rate
        `optim`: Optimizer Name
        `batch_size`: Batch Size for training
        `test_interval`: Interval of training
        """
        self.epochs = epochs
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.optim = optim
        self.batch_size = batch_size
        self.test_interval = test_interval

    def get_optim(self) -> Type[optim.Optimizer]:
        return getattr(optim, self.optim)

    def get_criterion(self) -> Callable:
        return getattr(losses, self.criterion)
