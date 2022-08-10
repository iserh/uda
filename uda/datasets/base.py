"""Loader for the M&Ms dataset."""
from abc import abstractmethod, abstractproperty
from typing import Optional

from torch.utils.data import DataLoader

from uda.config import Config


class UDADataset:
    @abstractproperty
    def config(self) -> Config:
        ...

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
