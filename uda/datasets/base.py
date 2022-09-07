"""Loader for the M&Ms dataset."""
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from ..config import Config


class classproperty(object):
    def __init__(self, f: Any) -> None:
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> Any:
        return self.f(owner)


class UDADataset(ABC):
    artifact_name: Optional[str] = None

    class_labels: dict[int, str]
    vendors: list[str]
    config: Config
    imsize: Optional[tuple[int, int, int]]
    patch_size: Optional[tuple[int, int, int]]

    @abstractmethod
    def setup(self) -> None:
        ...

    @abstractmethod
    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        ...

    @abstractmethod
    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        ...

    @abstractmethod
    def test_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        ...

    @abstractmethod
    def get_split(self, split: str, batch_size: Optional[int] = None) -> tuple[DataLoader, torch.Tensor]:
        ...

    def has_split(self, split: str) -> bool:
        try:
            self.get_split(split)
            return True
        except NotImplementedError:
            return False
