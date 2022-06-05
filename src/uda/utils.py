from typing import Tuple

import numpy as np
import torch


def is_notebook() -> bool:
    try:
        import os

        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except Exception:
        return False
    else:  # pragma: no cover
        return True


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


def reshape_to_volume(
    data: torch.Tensor, orig_shape: Tuple[int, int, int], patch_dims: Tuple[int, int, int]
) -> torch.Tensor:
    if patch_dims is not None:
        # move n_patches in seperate dimension
        n_patches = [int(dim_size / patch_size) for patch_size, dim_size in zip(patch_dims, orig_shape)]
        data = data.reshape(-1, np.prod(n_patches), *patch_dims)
        # unpatchify
        data = unpatchify(data, orig_shape, start=2)

    # reshape to 3d volume
    return data.reshape(-1, *orig_shape)
