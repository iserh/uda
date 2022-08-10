"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from patchify import patchify
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from uda.models.modules import center_pad_crop

from .base import UDADataset
from .configuration_cc359 import CC359Config


class CC359(UDADataset):
    """Calgary Campinas data module.

    Args:
        config (Union[CC359Config, str]): Either path to config file or config object itself.
        root (str, optional): Path where dataset is located. Defaults to "/tmp/data/CC359-Skull-stripping".
    """

    artifact_name = "iserh/UDA-Datasets/CC359-Skull-stripping:latest"
    class_labels = {1: "brain"}

    def __init__(self, config: Union[CC359Config, str], root: str = "/tmp/data/CC359-Skull-stripping") -> None:
        if not isinstance(config, CC359Config):
            config = CC359Config.from_file(config)

        self.root = Path(root)
        self.config = config
        self.vendor = config.vendor
        self.fold = config.fold
        self.rotate = config.rotate
        self.flatten = config.flatten
        self.imsize = config.imsize
        self.patch_size = config.patch_size
        self.clip_intensities = config.clip_intensities
        self.random_state = config.random_state

    def setup(self) -> None:
        """Load data from disk, preprocess and split."""
        images_dir = self.root / "Original" / self.vendor
        labels_dir = self.root / "Silver-standard" / self.vendor

        # sorted is absolutely crucial here:
        # we cannot expect that files are downloaded in the same order on each system
        files = sorted(list(images_dir.glob("*.nii.gz")))

        scaler = MinMaxScaler()

        images, masks, spacings_mm = [], [], []
        for file in tqdm(files, desc="Loading files"):
            nib_img: SpatialImage = nib.load(file)
            img = nib_img.get_fdata("unchanged", dtype=np.float32)

            nib_label: SpatialImage = nib.load(labels_dir / (file.name[:-7] + "_ss.nii.gz"))
            mask = nib_label.get_fdata("unchanged", dtype=np.float32)

            # clip & scale the images
            if self.clip_intensities is not None:
                img = img.clip(min=self.clip_intensities[0], max=self.clip_intensities[1])
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            # from here on -> torch backend
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)

            if self.rotate is not None:
                img = torch.rot90(img, k=self.rotate, dims=(1, 2))
                mask = torch.rot90(mask, k=self.rotate, dims=(1, 2))

            img = center_pad_crop(img, self.imsize)
            mask = center_pad_crop(mask, self.imsize)

            images.append(img)
            masks.append(mask)
            spacings_mm.append(torch.Tensor(nib_img.header.get_zooms()))

        # stack and pad/crop
        data = torch.stack(images)
        targets = torch.stack(masks)
        self.spacings_mm = torch.stack(spacings_mm)

        if self.patch_size is not None:
            n = len(data)
            # sadly patchify only works with numpy arrays
            data = patchify(data.reshape(-1, *data.shape[-2:]).numpy(), self.patch_size, self.patch_size).reshape(
                n, -1, *self.patch_size
            )
            data = torch.from_numpy(data)
            targets = patchify(
                targets.reshape(-1, *targets.shape[-2:]).numpy(), self.patch_size, self.patch_size
            ).reshape(n, -1, *self.patch_size)
            targets = torch.from_numpy(targets)

        if self.fold is not None:
            # split data & targets into train/val
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            train_indices, val_indices = list(kf.split(files))[self.fold]
            X_train, X_val = data[train_indices], data[val_indices]
            y_train, y_val = targets[train_indices], targets[val_indices]
        else:
            X_train = X_val = data
            y_train = y_val = targets

        # optional flatten & add channel dim
        if self.flatten:
            # flatten 3d volumes to 2d images
            X_train = X_train.reshape(-1, *X_train.shape[-2:]).unsqueeze(1)
            X_val = X_val.reshape(-1, *X_val.shape[-2:]).unsqueeze(1)
            y_train = y_train.reshape(-1, *y_train.shape[-2:]).unsqueeze(1)
            y_val = y_val.reshape(-1, *y_val.shape[-2:]).unsqueeze(1)
        else:
            X_train = X_train.reshape(-1, *X_train.shape[-3:]).unsqueeze(1)
            X_val = X_val.reshape(-1, *X_val.shape[-3:]).unsqueeze(1)
            y_train = y_train.reshape(-1, *y_train.shape[-3:]).unsqueeze(1)
            y_val = y_val.reshape(-1, *y_val.shape[-3:]).unsqueeze(1)

        self.train_split = TensorDataset(X_train, y_train)
        self.val_split = TensorDataset(X_val, y_val)

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.train_split)
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.val_split)
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=False)
