"""Loader for the Calgary Campinas dataset."""
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from nibabel.spatialimages import SpatialImage
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm


class CalgaryCampinasDataset(Dataset):
    """Dataset class for loading Calgary Campinas dataset."""

    D, W, H = 200, 256, 256

    def __init__(
        self,
        data_path: str,
        vendor: str,
        fold: Optional[int] = None,
        train: bool = True,
        rotate: bool = True,
        flatten: bool = False,
        random_state: int = 42,
    ) -> None:
        """Args:
        `data_path` : Dataset location
        `vendor` : vendor
        `fold` : Fold index for cross-validation
        `train` : Whether to load training or test partition (only when fold is not None)
        `rotate` : Rotate the images
        `flatten` : Flatten the Z dimension
        `random_state` : Random state for cross-validation
        """
        self.vendor = vendor
        self.fold = fold
        self.train = train
        self.rotate = rotate
        self.flatten = flatten
        self.random_state = random_state
        self.load_files(data_path)

    def select_fold(self, files: List[str]) -> List[str]:
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        files = np.array(files)
        for i, (train_indices, test_indices) in enumerate(kf.split(files)):
            if i == self.fold:
                indices = train_indices if self.train else test_indices

        return files[indices].tolist()

    def pad_images(self, images: List[np.ndarray], mode: str = "edge") -> List[np.ndarray]:
        padded_images = []
        for x in images:
            # crop image
            if x.shape[0] > self.D or x.shape[1] > self.W or x.shape[2] > self.H:
                x = x[: self.D, : self.W, : self.H]
            # pad image
            if x.shape[0] < self.D or x.shape[1] < self.W or x.shape[2] < self.H:
                x = np.pad(x, ((0, self.D - x.shape[0]), (0, self.W - x.shape[1]), (0, self.H - x.shape[2])), mode=mode)

            padded_images.append(x)

        return padded_images

    def pad_voxel_dims(self, voxel_dims: List[np.ndarray], mode: str = "constant") -> List[np.ndarray]:
        padded_voxel_dims = []
        for v in voxel_dims:
            # crop array
            if v.shape[0] > self.D:
                v = v[: self.D]
            # pad array
            if v.shape[0] < self.D:
                v = np.pad(v, ((0, self.D - v.shape[0]), (0, 0)), mode=mode)

            padded_voxel_dims.append(v)

        return padded_voxel_dims

    def load_files(self, data_path: str) -> None:
        data_path = Path(data_path)

        images_dir = data_path / "Original" / self.vendor
        files = list(images_dir.glob("*.nii.gz"))
        if self.fold is not None:
            files = self.select_fold(files)

        scaler = MinMaxScaler()

        images, labels, voxel_dims = [], [], []
        for file in tqdm(files, desc="Loading files"):
            nib_img: SpatialImage = nib.load(file)
            img = nib_img.get_fdata("unchanged", dtype=np.float32)

            label_path = data_path / "Silver-standard" / self.vendor / (file.name[:-7] + "_ss.nii.gz")
            nib_label: SpatialImage = nib.load(label_path)
            label = nib_label.get_fdata("unchanged", dtype=np.float32)

            # scale the images
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            if self.rotate:
                img = np.rot90(img, k=-1, axes=(1, 2))
                label = np.rot90(label, k=-1, axes=(1, 2))

            spacing = np.array([nib_img.header.get_zooms()] * img.shape[0])

            images.append(img)
            labels.append(label)
            voxel_dims.append(spacing)

        images = np.stack(self.pad_images(images))
        labels = np.stack(self.pad_images(labels))
        voxel_dims = np.stack(self.pad_voxel_dims(voxel_dims))

        if self.flatten:
            images = images.reshape(-1, *images.shape[-2:])
            labels = labels.reshape(-1, *labels.shape[-2:])
            voxel_dims = voxel_dims.reshape(-1, voxel_dims.shape[-1])

        # insert channel dimension
        self.data = torch.from_numpy(images).unsqueeze(1)
        self.label = torch.from_numpy(labels).unsqueeze(1)
        self.voxel_dim = torch.from_numpy(voxel_dims).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        labels = self.label[idx]
        voxel_dim = self.voxel_dim[idx]

        return data, labels, voxel_dim


if __name__ == "__main__":
    data_path = Path("/home/iailab36/iser/uda-data")
    dataset = CalgaryCampinasDataset(data_path, vendor="GE_3", fold=1, train=True, flatten=True)

    print(dataset.data.shape)
    print(dataset.label.shape)
    print(dataset.voxel_dim.shape)
