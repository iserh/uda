"""Loader for the Calgary Campinas dataset.

Author: Rasha Sheikh
"""
import os
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CalgaryCampinasDataset(Dataset):
    """Dataset class for loading Calgary Campinas dataset."""

    def __init__(
        self,
        data_path: str,
        site: int = 2,
        train: bool = True,
        fold: int = -1,
        rotate: bool = True,
        scale: bool = True,
        subj_index: List = [],
    ) -> None:
        """Initialize Calgary Campinas dataset.

        Args:
            data_path : Dataset location
            site : Site (Vendor). Defaults to 2.
            train : Use train partition. Defaults to True.
            fold : Fold index. Defaults to -1.
            rotate : Rotate the images. Defaults to True.
            scale : Scale the images. Defaults to True.
            subj_index : Index of the subject. Defaults to [].
        """
        self.rotate = rotate
        self.scale = scale
        self.fold = fold
        self.train = train
        self.subj_index = subj_index
        self.site = site

        if site == 1:
            self.folder = "GE_15"
        elif site == 2:
            self.folder = "GE_3"
        elif site == 3:
            self.folder = "Philips_15"
        elif site == 4:
            self.folder = "Philips_3"
        elif site == 5:
            self.folder = "Siemens_15"
        elif site == 6:
            self.folder = "Siemens_3"
        else:
            self.folder = "GE_3"

        self.load_files(data_path)

    def get_fold(self, files: np.ndarray) -> List[str]:
        """Get the files list for the current fold.

        Args:
            files : Fille name array

        Returns:
            List of filenames
        """
        kf = KFold(n_splits=3)
        folds = kf.split(files)
        k_i = 1
        for train_indices, test_indices in folds:
            if k_i == self.fold:
                if self.train:
                    indices = train_indices
                else:
                    indices = test_indices
                break
            k_i += 1
        return files[indices]

    def pad_image(self, img: np.ndarray) -> np.ndarray:
        """Pad image to the same size.

        Args:
            img : images array

        Returns:
            Padded images array
        """
        s, h, w = img.shape
        if h < w:
            b = (w - h) // 2
            a = w - (b + h)
            return np.pad(img, ((0, 0), (b, a), (0, 0)), mode="edge")
        elif w < h:
            b = (h - w) // 2
            a = h - (b + w)
            return np.pad(img, ((0, 0), (0, 0), (b, a)), mode="edge")
        else:
            return img

    def pad_image_w_size(self, data_array: np.ndarray, max_size: int) -> np.ndarray:
        """Pad images.

        Args:
            img : images array

        Returns:
            Padded images array
        """
        current_size = data_array.shape[-1]
        b = (max_size - current_size) // 2
        a = max_size - (b + current_size)
        return np.pad(data_array, ((0, 0), (b, a), (b, a)), mode="edge")

    def unify_sizes(self, input_images: np.ndarray, input_labels: np.ndarray) -> np.ndarray:
        """Unify the size of the images.

        Args:
            input_images : Input images array
            input_labels : Input labels array

        Returns:
            Unified images array
        """
        sizes = np.zeros(len(input_images), np.int)
        for i in range(len(input_images)):
            sizes[i] = input_images[i].shape[-1]
        max_size = np.max(sizes)
        for i in range(len(input_images)):
            if sizes[i] != max_size:
                input_images[i] = self.pad_image_w_size(input_images[i], max_size)
                input_labels[i] = self.pad_image_w_size(input_labels[i], max_size)
        return input_images, input_labels

    def load_files(self, data_path: str) -> None:
        """Load files.

        Args:
            data_path : Dataset location
        """
        self.sagittal = True

        scaler = None
        if self.scale:
            scaler = MinMaxScaler()
        images = []
        labels = []
        self.voxel_dim = []

        images_path = os.path.join(data_path, "Original", self.folder)

        files = np.array(sorted(os.listdir(images_path)))
        if self.fold > 0:
            files = self.get_fold(files)
        if len(self.subj_index) > 0:
            files = files[self.subj_index]
        for i, f in enumerate(files):
            nib_file = nib.load(os.path.join(images_path, f))
            img = nib_file.get_fdata("unchanged", dtype=np.float32)
            print(i, f, img.shape)
            lbl = nib.load(os.path.join(data_path, "Silver-standard", self.folder, f[:-7] + "_ss.nii.gz")).get_fdata(
                "unchanged", dtype=np.float32
            )
            if self.scale:
                transformed = scaler.fit_transform(np.reshape(img, (-1, 1)))
                img = np.reshape(transformed, img.shape)
            if not self.sagittal:
                img = np.moveaxis(img, -1, 0)
            if self.rotate:
                img = np.rot90(img, axes=(1, 2))
            if img.shape[1] != img.shape[2]:
                img = self.pad_image(img)
            images.append(img)

            if not self.sagittal:
                lbl = np.moveaxis(lbl, -1, 0)
            if self.rotate:
                lbl = np.rot90(lbl, axes=(1, 2))
            if lbl.shape[1] != lbl.shape[2]:
                lbl = self.pad_image(lbl)
            labels.append(lbl)
            spacing = [nib_file.header.get_zooms()] * img.shape[0]
            self.voxel_dim.append(np.array(spacing))

        images, labels = self.unify_sizes(images, labels)

        self.data = np.expand_dims(np.vstack(images), axis=1)
        self.label = np.expand_dims(np.vstack(labels), axis=1)
        self.voxel_dim = np.vstack(self.voxel_dim)

        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.voxel_dim = torch.from_numpy(self.voxel_dim)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the item.

        Returns:
            Tuple of data, label and voxel_dim
        """
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]

        return data, labels, voxel_dim
