from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
import torch
from ignite.utils import to_onehot
from patchify import unpatchify


def reshape_to_volume(
    data: Union[np.ndarray, torch.Tensor],
    dim: int,
    imsize: tuple[int, int, int],
    patch_size: Optional[tuple[int, int, int]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    imsize = tuple(imsize)
    # check if torch.Tensor (patchify uses numpy backend)
    output_type_tensor = isinstance(data, torch.Tensor)
    data = data.numpy() if output_type_tensor else data

    # unpatchify if data is patchified
    if patch_size is not None:
        # ensure tuple typing
        patch_size = tuple(patch_size)
        # compute number of patches for each axis
        n_patches = [axis_size // patch_size for axis_size, patch_size in zip(imsize, patch_size)]
        # size of the actual data patches (might be cropped due to down-/upsampling)
        cropped_patch_size = patch_size[:-dim] + data.shape[-dim:]
        # size of the final (cropped) image
        cropped_imsize = [ps * np for ps, np in zip(cropped_patch_size, n_patches)]
        # get the batch size
        batch_size = int(data.shape[0] // (np.prod(n_patches[-dim:]) * np.prod(imsize[:-dim])))
        # subsume batch_size in first patch axis (z-axis)
        data = data.reshape(batch_size * n_patches[0], *n_patches[1:], *cropped_patch_size)
        # unpatchify (subsume batch_size in first image axis)
        data = unpatchify(data, imsize=(batch_size * cropped_imsize[0], *cropped_imsize[1:]))
    else:
        cropped_imsize = imsize[:-dim] + data.shape[-dim:]

    # extract batch_size in first axis
    data = data.reshape(-1, *cropped_imsize)

    return torch.from_numpy(data) if output_type_tensor else data


def pipe(*transforms: Callable) -> Callable:
    def output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> Any:
        for transform in transforms:
            output = transform(output)

        return output

    return output_transform


def one_hot_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output[:2]
    rest = output[2:] if len(output) > 2 else []

    if y_pred.shape[1] == 1:
        n_classes = 2
        preds = y_pred.sigmoid().round().long()
        targets = y_true.long()
    else:
        n_classes = y_pred.shape[1]
        preds = y_pred.argmax(1)
        targets = y_true.argmax(1)

    preds = to_onehot(preds, num_classes=n_classes)

    return preds, targets, *rest


def to_cpu_output_transform(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return [tensor.cpu() for tensor in output]


def get_preds_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output[:2]
    rest = output[2:] if len(output) > 2 else []

    if y_pred.shape[1] == 1:
        preds = y_pred.sigmoid().round().long()
        targets = y_true.long()
    else:
        preds = y_pred.argmax(1)
        targets = y_true.argmax(1)

    return preds, targets, *rest


def flatten_output_transform(output: tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    return [o.flatten(dim) for o in output]
