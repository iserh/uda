"""Loader for the M&Ms dataset."""
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
from ignite.utils import to_onehot
from nibabel.spatialimages import SpatialImage
from pypatchify.pt import pt
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..transforms import center_pad
from .base import UDADataset
from .configuration_mms import MAndMsConfig


class MAndMs(UDADataset):
    """M&Ms data module.

    Args:
        config (Union[MAndMsConfig, str]): Either path to config file or config object itself.
        root (str, optional): Path where dataset is located. Defaults to "/tmp/data".
    """

    artifact_name = "iserh/UDA-Datasets/MAndMs:latest"
    class_labels = {0: "Background", 1: "left ventricle (LV)", 2: "myocardium (MYO)", 3: "right ventricle (RV)"}
    vendors = ["Canon", "GE", "Philips", "Siemens"]

    def __init__(self, config: Union[MAndMsConfig, str], root: str = "/tmp/data") -> None:
        if not isinstance(config, MAndMsConfig):
            config = MAndMsConfig.from_file(config)

        self.root = Path(root) / self.__class__.__name__
        self.config = config

        phases_dict = {"ED": 0, "ES": 1}
        self.vendor = config.vendor
        self.fold = config.fold
        self.selected_phases = [phases_dict[p] for p in config.phases]
        self.flatten = config.flatten
        self.imsize = config.imsize
        self.patch_size = config.patch_size
        self.clip_intensities = config.clip_intensities
        self.limit = config.limit
        self.random_state = config.random_state

    def setup(self) -> None:
        """Load data from disk and preprocess."""
        self._load_train_val_files()
        self._load_test_files()

    def train_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = True) -> DataLoader:
        batch_size = batch_size or len(self.train_split)
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=shuffle)

    def val_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
        batch_size = batch_size or len(self.val_split)
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=shuffle)

    def test_dataloader(self, batch_size: Optional[int] = None, shuffle: bool = False) -> DataLoader:
        batch_size = batch_size or len(self.test_split)
        return DataLoader(self.test_split, batch_size=batch_size, shuffle=shuffle)

    def get_split(self, split: str, batch_size: Optional[int] = None) -> tuple[DataLoader, torch.Tensor]:
        if split == "training":
            return self.train_dataloader(batch_size), self.train_spacings
        elif split == "validation":
            return self.val_dataloader(batch_size), self.val_spacings
        elif split == "testing":
            return self.test_dataloader(batch_size), self.test_spacings
        else:
            raise NotImplementedError

    def _load_train_val_files(self) -> None:
        data, targets, spacings = self._load_files(self.root / self.vendor / "Training")

        if self.fold is not None:
            # split data & targets into train/val
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            train_indices, val_indices = list(kf.split(data))[self.fold]
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
        y_train = pt.collapse_dims(y_train, dims=(0, 1))
        y_train = to_onehot(y_train.long(), num_classes=4).float()
        y_val = pt.collapse_dims(y_val, dims=(0, 1))
        y_val = to_onehot(y_val.long(), num_classes=4).float()

        self.train_split = TensorDataset(X_train, y_train)
        self.val_split = TensorDataset(X_val, y_val)

    def _load_test_files(self) -> None:
        data, targets, spacings = self._load_files(self.root / self.vendor / "Testing")

        # collapse batch_dim and flatten_dim/patch_dim; unsqueeze for channel dim
        data = pt.collapse_dims(data, dims=(0, 1)).unsqueeze(1)
        targets = pt.collapse_dims(targets, dims=(0, 1))
        targets = to_onehot(targets.long(), num_classes=4).float()

        self.test_split = TensorDataset(data, targets)
        self.test_spacings = spacings

    def _load_files(self, directory: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        items = sorted(list(directory.iterdir()))
        if self.limit is not None:
            items = items[: self.limit]

        data_files = [next(subdir.glob("*sa.nii.gz")) for subdir in items]
        mask_files = [next(subdir.glob("*sa_gt.nii.gz")) for subdir in items]

        scaler = MinMaxScaler()
        images, masks, spacings = [], [], []

        for data_file, mask_file in tqdm(
            zip(data_files, mask_files), total=len(data_files), desc=f"Loading {directory.name} data"
        ):
            # get image
            nib_img: SpatialImage = nib.load(data_file)
            img = nib_img.get_fdata("unchanged", dtype=np.float32)

            # get mask
            nib_label: SpatialImage = nib.load(mask_file)
            mask = nib_label.get_fdata("unchanged", dtype=np.float32)

            # get spacing info
            spacing = np.array(nib_img.header.get_zooms())[:3]

            # shape: (X, Y, Z, time) -> (time, Z, X, Y)
            img = img.transpose(3, 2, 0, 1)
            mask = mask.transpose(3, 2, 0, 1)
            spacing = spacing[None, [2, 0, 1]]

            # get the end-diastolic(ED) and end-systolic(ES) frame
            phase_indices = np.where((mask != 0).any((1, 2, 3)))[0]
            phase_indices = phase_indices[self.selected_phases]
            # select the phase frames
            img = img[phase_indices]
            mask = mask[phase_indices]
            spacing = np.repeat(spacing, len(phase_indices), axis=0)

            # clip & scale the images
            if self.clip_intensities is not None:
                img = img.clip(min=self.clip_intensities[0], max=self.clip_intensities[1])
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            # from here on -> torch backend
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            spacing = torch.from_numpy(spacing)

            # pad & crop
            img = center_pad(img, self.imsize)
            mask = center_pad(mask, self.imsize)

            images.extend(img)
            masks.extend(mask)
            spacings.extend(spacing)

        data = torch.stack(images).unsqueeze(1)
        targets = torch.stack(masks).unsqueeze(1)
        spacings = torch.stack(spacings)

        if self.patch_size is not None:
            data = pt.patchify_to_batches(data, self.patch_size, batch_dim=1)
            targets = pt.patchify_to_batches(targets, self.patch_size, batch_dim=1)

        if self.flatten:
            data = pt.collapse_dims(data, dims=(1, data.ndim - 3), target_dim=1)
            targets = pt.collapse_dims(targets, dims=(1, targets.ndim - 3), target_dim=1)

        return data, targets, spacings
