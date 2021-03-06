from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
from ignite.utils import to_onehot
from patchify import unpatchify


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


if is_notebook():
    from tqdm.notebook import tqdm  # noqa: F401
else:
    from tqdm import tqdm  # noqa: F401


def reshape_to_volume(
    data: Union[np.ndarray, torch.Tensor], imsize: Tuple[int, int, int], patch_size: Tuple[int, int, int]
) -> Union[np.ndarray, torch.Tensor]:
    # check if torch.Tensor (patchify uses numpy backend)
    if isinstance(data, torch.Tensor):
        data = data.numpy()
        output_type_tensor = True
    else:
        output_type_tensor = False

    # unpatchify if data is patchified
    if patch_size is not None:
        # compute number of patches for each axis
        n_patches = [axis_size // patch_size for axis_size, patch_size in zip(imsize, patch_size)]
        batch_size = data.shape[0] // np.prod(n_patches)
        # subsume batch_size in first patch axis (z-axis)
        data = data.reshape(batch_size * n_patches[0], *n_patches[1:], *patch_size)
        # unpatchify (subsume batch_size in first image axis)
        data = unpatchify(data, imsize=(batch_size * imsize[0], *imsize[1:]))

    # extract batch_size in first axis
    data = data.reshape(-1, *imsize)

    return torch.from_numpy(data) if output_type_tensor else data


def pipe(*transforms: Callable) -> Callable:
    def output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Any:
        for transform in transforms:
            output = transform(output)

        return output

    return output_transform


def binary_one_hot_output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    y_pred, y = output
    y_pred = y_pred.sigmoid().round().long()
    y_pred = to_onehot(y_pred, 2)
    return y_pred, y.long()


def to_cpu(output: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    return [tensor.cpu() for tensor in output]


def pred_from_vae_output(output: Tuple[Tuple[torch.Tensor, ...], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    vae_output, x_true = output
    return vae_output[0], x_true


def distr_from_vae_output(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    vae_output, _ = output
    return vae_output[1:]


def sigmoid_round(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output
    return y_pred.sigmoid().round().long(), y_true


def flatten_output_transform(output: Tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    y_pred, y_true = output
    return y_pred.sigmoid().round().long().flatten(dim), y_true.flatten(dim)
