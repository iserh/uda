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

    PADDING_SHAPE = (192, 256, 256)

    def __init__(
        self,
        data_path: str,
        vendor: str,
        fold: Optional[int] = None,
        train: bool = True,
        rotate: bool = True,
        flatten: bool = False,
        patchify: Optional[Tuple[int]] = None,
        squash_patches: bool = True,
        random_state: int = 42,
    ) -> None:
        """Args:
        `data_path` : Dataset location
        `vendor` : vendor
        `fold` : Fold index for cross-validation
        `train` : Whether to load training or test partition (only when fold is not None)
        `rotate` : Rotate the images
        `flatten` : Flatten the Z dimension
        `patchify` : Patchify the images
        `squash_patches` : Squash the patches
        `random_state` : Random state for cross-validation
        """
        self.vendor = vendor
        self.fold = fold
        self.train = train
        self.rotate = rotate
        self.flatten = flatten
        self.patchify = patchify
        self.squash_patches = squash_patches
        self.random_state = random_state
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
        elif arr.shape[1] > width:
            x = (int(np.floor((arr.shape[1] - width) / 2)), int(np.ceil((arr.shape[1] - width) / 2)))
            arr = arr[:, x[0] : -x[1]]
        elif arr.shape[2] > height:
            y = (int(np.floor((arr.shape[2] - height) / 2)), int(np.ceil((arr.shape[2] - height) / 2)))
            arr = arr[:, :, y[0] : -y[1]]

        # pad image
        if arr.shape[0] < depth or arr.shape[1] < width or arr.shape[2] < height:
            arr = np.pad(
                arr, ((0, depth - arr.shape[0]), (0, width - arr.shape[1]), (0, height - arr.shape[2])), mode=mode
            )

        return arr

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

            # clip & scale the images
            img = img.clip(min=-200, max=400)
            img = scaler.fit_transform(img.reshape(-1, 1)).reshape(img.shape)

            if self.rotate:
                img = np.rot90(img, k=-1, axes=(1, 2))
                label = np.rot90(label, k=-1, axes=(1, 2))

            # pad the images
            img = self.pad_array(img)
            label = self.pad_array(label)

            voxel_dim = np.array([nib_img.header.get_zooms()] * img.shape[0])
            # padding for voxel dim
            if voxel_dim.shape[0] > self.PADDING_SHAPE[0]:
                voxel_dim = voxel_dim[: self.PADDING_SHAPE[0]]
            if voxel_dim.shape[0] < self.PADDING_SHAPE[0]:
                voxel_dim = np.pad(
                    voxel_dim, ((0, self.PADDING_SHAPE[0] - voxel_dim.shape[0]), (0, 0)), mode="constant"
                )

            images.append(img)
            labels.append(label)
            voxel_dims.append(voxel_dim)

        # stack and convert to torch tensors
        data = torch.from_numpy(np.stack(images))
        label = torch.from_numpy(np.stack(labels))
        voxel_dim = torch.from_numpy(np.stack(voxel_dims))

        if self.flatten:
            data = data.reshape(-1, *data.shape[-2:])
            label = label.reshape(-1, *label.shape[-2:])
            voxel_dim = voxel_dim.reshape(-1, voxel_dim.shape[-1])

        # add channel dimension
        data = data.unsqueeze(1)
        label = label.unsqueeze(1)
        voxel_dim = voxel_dim.unsqueeze(1)

        # patchify the data
        if self.patchify is not None:
            data = patchify(data, self.patchify)
            label = patchify(label, self.patchify)
            self.n_patches = data.shape[2 : 2 + len(self.patchify)]

            if self.squash_patches:
                data = data.reshape(-1, 1, *self.patchify)
                label = label.reshape(-1, 1, *self.patchify)
            else:
                data = data.reshape(data.shape[0], -1, 1, *self.patchify)
                label = label.reshape(label.shape[0], -1, 1, *self.patchify)

        self.data = data
        self.label = label
        self.voxel_dim = voxel_dim

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        labels = self.label[idx]

        return data, labels


def patchify(t: torch.Tensor, size: Tuple[int], start: int = 2) -> torch.Tensor:
    """Args:
    `t` : Tensor to patchify
    `size` : Patch size
    `start` : Starting index of the dimensions to patchify
    """
    # offset is needed because in each iteration one axis gets added
    for offset, (i, dim_size) in enumerate(enumerate(size, start=start)):
        t = torch.stack(t.split(dim_size, dim=i + offset), dim=i)
    return t


def unpatchify(t: torch.Tensor, size: Tuple[int], start: int = 3, patch_dim: int = 1) -> torch.Tensor:
    """Args:
    `t` : Tensor to unpatchify
    `size` : Unpatchified size
    `start` : Starting index of the dimensions to unpatchify
    `patch_dim` : Dimension of the patch
    """
    # compute number of patches for each patch dimension
    n_patches = [patch_size // t_size for patch_size, t_size in zip(size, t.shape[start:])]
    # reshape tensor with unfolded patches
    t = t.reshape(*t.shape[:patch_dim], *n_patches, *t.shape[start:])
    # reset start dimension (unfolding added dimensions in front)
    patch_dim_end = patch_dim + len(size) - 1
    # concatenate patches
    for i in range(len(size)):
        t = torch.cat(t.split(1, patch_dim_end - i), dim=-i - 1)
    # squash patch dimensions
    return t.reshape(*t.shape[:patch_dim], *size)


if __name__ == "__main__":
    data_path = Path("/home/iailab36/iser/uda-data")
    dataset = CalgaryCampinasDataset(data_path, vendor="GE_3", fold=1, train=True, flatten=True)

    print(dataset.data.shape)
    print(dataset.label.shape)
    print(dataset.voxel_dim.shape)
