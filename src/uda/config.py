from enum import Enum
from typing import Any, Callable, Type

import torch.optim as optim
import yaml
from yaml import Node

import uda.losses as losses


class EnumDumper(yaml.SafeDumper):
    def represent_data(self, data: Any) -> Node:
        if isinstance(data, Enum):
            return self.represent_data(data.value)
        return super().represent_data(data)


class Config:
    """Configuration Class."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, Dumper=EnumDumper)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = cls(**yaml.load(f, Loader=yaml.SafeLoader))
        return config


class HParams(Config):
    """Configuration for Hyperparameters."""

    def __init__(
        self,
        epochs: int = 10,
        criterion: str = "dice_loss",
        learning_rate: float = 1e-4,
        optim: str = "Adam",
        train_batch_size: int = 4,
        val_batch_size: int = 4,
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
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def get_optim(self) -> Type[optim.Optimizer]:
        return getattr(optim, self.optim)

    def get_criterion(self) -> Callable:
        return getattr(losses, self.criterion)
