"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialImage
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from uda.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from patchify import patchify

from .configuration_cc359 import CC359Config


class CC359(Dataset):
    """Dataset class for loading Calgary Campinas dataset."""

    PADDING_SHAPE = (192, 256, 256)

    def __init__(self, data_path: str, config: CC359Config, train: bool = False) -> None:
        """Args:
        `data_path` : Dataset location
        `config` : CC359Config
        """
        self.train = train
        self.vendor = config.vendor
        self.fold = config.fold
        self.rotate = config.rotate
        self.flatten = config.flatten
        self.patch_size = config.patch_size
        self.clip_intensities = config.clip_intensities
        self.random_state = config.random_state
        self.load_files(data_path)

    def select_fold(self, files: List[str]) -> List[str]:
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        files = np.array(files)
        for i, (train_indices, test_indices) in enumerate(kf.split(files)):
            if i == self.fold:
                indices = train_indices if self.train else test_indices

        return files[indices].tolist()

    def pad_array(self, arr: np.ndarray, mode: str = "edge") -> np.ndarray:
        depth, width, height = self.PADDING_SHAPE

        # center crop image if too large
        if arr.shape[0] > depth:
            z = (int(np.floor((arr.shape[0] - depth) / 2)), int(np.ceil((arr.shape[0] - depth) / 2)))
            arr = arr[z[0] : -z[1]]
        if arr.shape[1] > width:
            x = (int(np.floor((arr.shape[1] - width) / 2)), int(np.ceil((arr.shape[1] - width) / 2)))
            arr = arr[:, x[0] : -x[1]]
        if arr.shape[2] > height:
            y = (int(np.floor((arr.shape[2] - height) / 2)), int(np.ceil((arr.shape[2] - height) / 2)))
            arr = arr[:, :, y[0] : -y[1]]

        # pad image
        if arr.shape[0] < depth:
            arr = np.pad(arr, ((0, depth - arr.shape[0]), (0, 0), (0, 0)), mode=mode)
        if arr.shape[1] < width:
            arr = np.pad(arr, ((0, 0), (0, width - arr.shape[1]), (0, 0)), mode=mode)
        if arr.shape[2] < height:
            arr = np.pad(arr, ((0, 0), (0, 0), (0, height - arr.shape[2])), mode=mode)

        return arr

    def load_files(self, data_path: str) -> None:
        data_path = Path(data_path)

        images_dir = data_path / "Original" / self.vendor
        # sorted is absolutely crucial here:
        # we cannot expect that files are downloaded in the same order on each system
        files = sorted(list(images_dir.glob("*.nii.gz")))
        if self.fold is not None:
            files = self.select_fold(files)

        scaler = MinMaxScaler()

        images, labels, spacings_mm = [], [], []
        for file in tqdm(files, desc="Loading files"):
            nib_img: SpatialImage = nib.load(file)
            img = nib_img.get_fdata("unchanged", dtype=np.float32)

            label_path = data_path / "Silver-standard" / self.vendor / (file.name[:-7] + "_ss.nii.gz")
            nib_label: SpatialImage = nib.load(label_path)
            targets = nib_label.get_fdata("unchanged", dtype=np.float32)

            # clip & scale the images
            if self.clip_intensities is not None:
                img = img.clip(min=self.clip_intensities[0], max=self.clip_intensities[1])
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            if self.rotate is not None:
                img = np.rot90(img, k=self.rotate, axes=(1, 2))
                targets = np.rot90(targets, k=self.rotate, axes=(1, 2))

            # pad the images
            img = self.pad_array(img)
            targets = self.pad_array(targets)

            images.append(img)
            labels.append(targets)
            spacings_mm.append(np.array(nib_img.header.get_zooms()))

        data = np.stack(images)
        targets = np.stack(labels)
        spacings_mm = np.stack(spacings_mm)
        self.imsize = self.PADDING_SHAPE

        # patchify the data
        if self.patch_size is not None:
            data = patchify(data.reshape(-1, *data.shape[-2:]), self.patch_size, self.patch_size)
            targets = patchify(targets.reshape(-1, *data.shape[-2:]), self.patch_size, self.patch_size)

            # flatten the patches
            data = data.reshape(-1, *self.patch_size)
            targets = targets.reshape(-1, *self.patch_size)

        # flatten 3d volumes to 2d images
        if self.flatten:
            data = data.reshape(-1, *data.shape[-2:])
            targets = targets.reshape(-1, *targets.shape[-2:])

        self.spacings_mm = spacings_mm
        # add channel dimension
        self.data = np.expand_dims(data, 1)
        self.targets = np.expand_dims(targets, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.data[idx]
        labels = self.targets[idx]

        return data, labels
