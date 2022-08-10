"""Loader for the M&Ms dataset."""
from abc import abstractmethod
from typing import Optional

from torch.utils.data import DataLoader


class UDADataset:
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
