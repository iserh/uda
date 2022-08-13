"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from pypatchify.pt import pt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..transforms import center_pad
from .base import UDADataset
from .configuration_cc359 import CC359Config


class CC359(UDADataset):
    """Calgary Campinas data module.

    Args:
        config (Union[CC359Config, str]): Either path to config file or config object itself.
        root (str, optional): Path where dataset is located. Defaults to "/tmp/data".
    """

    artifact_name = "iserh/UDA-Datasets/CC359-Skull-stripping:latest"
    class_labels = {0: "Background", 1: "brain"}

    def __init__(self, config: Union[CC359Config, str], root: str = "/tmp/data") -> None:
        if not isinstance(config, CC359Config):
            config = CC359Config.from_file(config)

        self.root = Path(root) / self.__class__.__name__
        self.config = config
        self.vendor = config.vendor
        self.fold = config.fold
        self.rotate = config.rotate
        self.flatten = config.flatten
        self.imsize = config.imsize
        self.patch_size = config.patch_size
        self.clip_intensities = config.clip_intensities
        self.limit = config.limit
        self.random_state = config.random_state

    def setup(self) -> None:
        """Load data from disk, preprocess and split."""
        images_dir = self.root / "Original" / self.vendor
        labels_dir = self.root / "Silver-standard" / self.vendor

        # sorted is absolutely crucial here:
        # we cannot expect that files are downloaded in the same order on each system
        files = sorted(list(images_dir.glob("*.nii.gz")))[: self.limit]

        scaler = MinMaxScaler()

        images, masks, spacings = [], [], []
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

            img = center_pad(img, self.imsize)
            mask = center_pad(mask, self.imsize)

            images.append(img)
            masks.append(mask)
            spacings.append(torch.Tensor(nib_img.header.get_zooms()))

        # stack and add one dimension for patches / image flattening
        data = torch.stack(images).unsqueeze(1)
        targets = torch.stack(masks).unsqueeze(1)
        spacings = torch.stack(spacings).unsqueeze(1)

        if self.patch_size is not None:
            data = pt.patchify_to_batches(data, self.patch_size, batch_dim=1)
            targets = pt.patchify_to_batches(targets, self.patch_size, batch_dim=1)

        if self.flatten:
            data = pt.collapse_dims(data, dims=(1, data.ndim-3))
            targets = pt.collapse_dims(targets, dims=(1, targets.ndim-3))

        if self.fold is not None:
            # split data & targets into train/val
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            train_indices, val_indices = list(kf.split(files))[self.fold]
            X_train, X_val = data[train_indices], data[val_indices]
            y_train, y_val = targets[train_indices], targets[val_indices]
            self.train_spacings = spacings[train_indices]
            self.val_spacings = spacings[val_indices]
        else:
            X_train = X_val = data
            y_train = y_val = targets

        # collapse batch_dim and flatten_dim/patch_dim; unsqueeze for channel dim
        X_train = pt.collapse_dims(X_train, dims=(0, 1)).unsqueeze(1)
        X_val = pt.collapse_dims(X_val, dims=(0, 1)).unsqueeze(1)
        y_train = pt.collapse_dims(y_train, dims=(0, 1)).unsqueeze(1)
        y_val = pt.collapse_dims(y_val, dims=(0, 1)).unsqueeze(1)

        self.train_split = TensorDataset(X_train, y_train)
        self.val_split = TensorDataset(X_val, y_val)

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.train_split)
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.val_split)
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=False)

    def test_dataloader(self, batch_size: Optional[int] = None) -> NotImplementedError:
        raise NotImplementedError

    def get_split(self, split: str, batch_size: Optional[int] = None) -> tuple[DataLoader, torch.Tensor]:
        if split == "training":
            return self.train_dataloader(batch_size), self.train_spacings
        elif split == "validation":
            return self.val_dataloader(batch_size), self.val_spacings
        else:
            raise NotImplementedError
