from collections.abc import Callable
from typing import Any, Union

import numpy as np
import torch
from ignite.utils import to_onehot
from patchify import unpatchify


def reshape_to_volume(
    data: Union[np.ndarray, torch.Tensor], imsize: tuple[int, int, int], patch_size: tuple[int, int, int]
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
    def output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> Any:
        for transform in transforms:
            output = transform(output)

        return output

    return output_transform


def binary_one_hot_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output[:2]
    rest = output[2:] if len(output) > 2 else []

    y_pred = y_pred.sigmoid().round()
    y_pred = to_onehot(y_pred.long(), 2)

    return y_pred, y_true.long(), *rest


def to_cpu_output_transform(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return [tensor.cpu() for tensor in output]


def sigmoid_round_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred = output[0]
    return y_pred.sigmoid().round(), output[1:]


def flatten_output_transform(output: tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    return [o.flatten(dim) for o in output]
