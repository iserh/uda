"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import wandb
from nibabel.spatialimages import SpatialImage
from patchify import patchify
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .configuration_cc359 import CC359Config


class CC359:
    """Calgary Campinas data module."""

    def __init__(self, root: str, config: CC359Config) -> None:
        """Args:
        `data_path` : Dataset location
        `config` : CC359Config
        """
        self.root = Path(root)
        self.config = config
        self.imsize = config.imsize
        self.patch_size = config.patch_size
        self.random_state = config.random_state

    @classmethod
    def from_preconfigured(
        cls, config: Union[CC359Config, Path, str], root: str = "/tmp/data/CC359-Skull-stripping"
    ) -> "CC359":
        if isinstance(config, CC359Config):
            cfg = config
        else:
            cfg = CC359Config.from_file(config)
        return cls(root, cfg)

    def prepare_data(self) -> None:
        """Downloads the dataset from `wandb.ai`."""
        # donwload latest version of dataset
        api = wandb.Api()
        artifact = api.artifact("iserh/UDA-Datasets/CC359-Skull-stripping:latest")
        artifact.download(root=self.root)

    def setup(self) -> None:
        images_dir = self.root / "Original" / self.config.vendor
        labels_dir = self.root / "Silver-standard" / self.config.vendor

        # sorted is absolutely crucial here:
        # we cannot expect that files are downloaded in the same order on each system
        files = sorted(list(images_dir.glob("*.nii.gz")))[:9]

        scaler = MinMaxScaler()

        images, labels, spacings_mm = [], [], []
        for file in tqdm(files, desc="Loading files"):
            nib_img: SpatialImage = nib.load(file)
            img = nib_img.get_fdata("unchanged", dtype=np.float32)

            nib_label: SpatialImage = nib.load(labels_dir / (file.name[:-7] + "_ss.nii.gz"))
            targets = nib_label.get_fdata("unchanged", dtype=np.float32)

            # clip & scale the images
            if self.config.clip_intensities is not None:
                img = img.clip(min=self.config.clip_intensities[0], max=self.config.clip_intensities[1])
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            if self.config.rotate is not None:
                img = np.rot90(img, k=self.config.rotate, axes=(1, 2))
                targets = np.rot90(targets, k=self.config.rotate, axes=(1, 2))

            # pad the images
            img = self._pad_array(img)
            targets = self._pad_array(targets)

            images.append(img)
            labels.append(targets)
            spacings_mm.append(np.array(nib_img.header.get_zooms()))

        data = np.stack(images)
        targets = np.stack(labels)
        self.spacings_mm = np.stack(spacings_mm)

        # split data & targets into train/val
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        train_indices, val_indices = list(kf.split(files))[self.config.fold]
        X_train, X_val = data[train_indices], data[val_indices]
        y_train, y_val = targets[train_indices], targets[val_indices]

        if self.patch_size is not None:
            # patchify X_train
            X_train = patchify(X_train.reshape(-1, *X_train.shape[-2:]), self.patch_size, self.patch_size).reshape(
                -1, *self.patch_size
            )
            # patchify X_val
            X_val = patchify(X_val.reshape(-1, *X_val.shape[-2:]), self.patch_size, self.patch_size).reshape(
                -1, *self.patch_size
            )
            # patchify y_train
            y_train = patchify(y_train.reshape(-1, *y_train.shape[-2:]), self.patch_size, self.patch_size).reshape(
                -1, *self.patch_size
            )
            # patchify y_val
            y_val = patchify(y_val.reshape(-1, *y_val.shape[-2:]), self.patch_size, self.patch_size).reshape(
                -1, *self.patch_size
            )

        if self.config.flatten:
            # flatten 3d volumes to 2d images
            X_train = X_train.reshape(-1, *X_train.shape[-2:])
            X_val = X_val.reshape(-1, *X_val.shape[-2:])
            y_train = y_train.reshape(-1, *y_train.shape[-2:])
            y_val = y_val.reshape(-1, *y_val.shape[-2:])

        # add channel dimension
        X_train = np.expand_dims(X_train, 1)
        X_val = np.expand_dims(X_val, 1)
        y_train = np.expand_dims(y_train, 1)
        y_val = np.expand_dims(y_val, 1)

        self.train_split = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.val_split = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    def train_dataloader(self, batch_size: int):
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int):
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=False)

    def _pad_array(self, arr: np.ndarray, mode: str = "edge") -> np.ndarray:
        depth, width, height = self.imsize

        # center crop image if too large
        if arr.shape[0] > depth:
            z = (
                int(np.floor((arr.shape[0] - depth) / 2)),
                int(np.ceil((arr.shape[0] - depth) / 2)),
            )
            arr = arr[z[0] : -z[1]]
        if arr.shape[1] > width:
            x = (
                int(np.floor((arr.shape[1] - width) / 2)),
                int(np.ceil((arr.shape[1] - width) / 2)),
            )
            arr = arr[:, x[0] : -x[1]]
        if arr.shape[2] > height:
            y = (
                int(np.floor((arr.shape[2] - height) / 2)),
                int(np.ceil((arr.shape[2] - height) / 2)),
            )
            arr = arr[:, :, y[0] : -y[1]]

        # pad image
        if arr.shape[0] < depth:
            arr = np.pad(arr, ((0, depth - arr.shape[0]), (0, 0), (0, 0)), mode=mode)
        if arr.shape[1] < width:
            arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1]), (0, 0)), mode=mode)
        if arr.shape[2] < height:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, height - arr.shape[2])), mode=mode)

        return arr
