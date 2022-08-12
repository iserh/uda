"""Loader for the M&Ms dataset."""
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from pypatchify.pt import pt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
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
    class_labels = {1: "left ventricle (LV)", 2: "myocardium (MYO)", 3: "right ventricle (RV)"}

    def __init__(self, config: Union[MAndMsConfig, str], root: str = "/tmp/data") -> None:
        if not isinstance(config, MAndMsConfig):
            config = MAndMsConfig.from_file(config)

        self.root = Path(root) / self.__class__.__name__
        self.config = config

        phases_dict = {"ED": 0, "ES": 1}
        self.selected_phases = [phases_dict[p] for p in config.phases]
        self.unlabeled = config.unlabeled
        self.flatten = config.flatten
        self.imsize = config.imsize
        self.offset = config.offset
        self.patch_size = config.patch_size
        self.clip_intensities = config.clip_intensities
        self.limit = config.limit

    def setup(self) -> None:
        """Load data from disk and preprocess."""
        self.train_split, self.train_spacings = self._load_files(self.root / "Training" / "Labeled")
        if self.unlabeled:
            unlabeled_split, unlabeled_spacings = self._load_files(self.root / "Training" / "Unlabeled")
            self.train_split = ConcatDataset([self.train_split, unlabeled_split])
            self.train_spacings = torch.cat([self.train_spacings, unlabeled_spacings])
        self.val_split, self.val_spacings = self._load_files(self.root / "Validation")
        self.test_split, self.test_spacings = self._load_files(self.root / "Testing")

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.train_split)
        return DataLoader(self.train_split, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.val_split)
        return DataLoader(self.val_split, batch_size=batch_size, shuffle=False)

    def test_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        batch_size = batch_size or len(self.test_split)
        return DataLoader(self.test_split, batch_size=batch_size, shuffle=False)

    def get_split(self, split: str, batch_size: Optional[int] = None) -> tuple[DataLoader, torch.Tensor]:
        if split == "training":
            return self.train_dataloader(batch_size), self.train_spacings
        elif split == "validation":
            return self.val_dataloader(batch_size), self.val_spacings
        elif split == "testing":
            return self.test_dataloader(batch_size), self.test_spacings
        else:
            raise NotImplementedError

    def _load_files(self, directory: Path) -> tuple[TensorDataset, torch.Tensor]:
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
            mask = torch.from_numpy(mask).long()
            spacing = torch.from_numpy(spacing)

            # pad & crop
            img = center_pad(img, self.imsize, self.offset)
            mask = center_pad(mask, self.imsize, self.offset)

            images.extend(img)
            masks.extend(mask)
            spacings.extend(spacing)

        data = torch.stack(images)
        targets = torch.stack(masks)
        spacings = torch.stack(spacings)

        if self.patch_size is not None:
            # sadly patchify only works with numpy arrays
            data = pt.patchify_to_batches(data, self.patch_size, batch_dim=0)
            targets = pt.patchify_to_batches(targets, self.patch_size, batch_dim=0)

        # optional flatten & add channel dim
        if self.flatten:
            # transpose z_dim next to batch_dim
            data = pt.collapse_dims(data, dims=(0, -3))
            targets = pt.collapse_dims(targets, dims=(0, -3))

        return TensorDataset(data, targets), spacings
