"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import Union

import ignite.distributed as idist
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from nibabel.spatialimages import SpatialImage
from patchify import patchify
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset_cc359 import CC359


class TeacherData:
    """Calgary Campinas data module."""

    def __init__(self, model: nn.Module, dataset: CC359) -> None:
        """Args:
        `model` : Teacher model
        `dataset` : Dataset
        """
        self.model = model
        self.dataset = dataset

    def setup(self, batch_size: int = 2) -> None:
        self.dataset.setup()

        y_train = self.get_teacher_labels(self.dataset.train_dataloader(batch_size))
        y_val = self.get_teacher_labels(self.dataset.val_dataloader(batch_size))

        self.train_split = TensorDataset(self.dataset.train_split.tensors[0], y_train)
        self.val_split = TensorDataset(self.dataset.val_split.tensors[0], y_val)

    @torch.no_grad()
    def get_teacher_labels(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.to(idist.device()).eval()
        preds = torch.cat(
            [
                self.model(x.to(idist.device())).sigmoid().round().cpu()
                for x, _ in tqdm(dataloader, desc="Predicting Pseudo labels")
            ]
        )
        self.model.cpu()
        return preds

    def train_dataloader(self, batch_size: int):
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int):
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=False)
