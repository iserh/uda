from collections.abc import Callable
from typing import Any

import torch
from ignite.utils import to_onehot


def pipe(*transforms: Callable) -> Callable:
    def output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> Any:
        for transform in transforms:
            output = transform(output)

        return output

    return output_transform


def to_cpu_output_transform(output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return [tensor.cpu() for tensor in output]


def get_preds_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output[:2]
    other = output[2:] if len(output) > 2 else []

    if y_pred.shape[1] == 1:
        preds = y_pred.sigmoid().round().long().squeeze()
        targets = y_true.long().squeeze()
    else:
        preds = y_pred.argmax(1)
        targets = y_true.argmax(1)

    return preds, targets, *other


def one_hot_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred, y_true = output[:2]
    other = output[2:] if len(output) > 2 else []

    num_classes = max(y_pred.shape[1], 2)  # if only one channel it's binary segmentation with sigmoid act
    preds, targets = get_preds_output_transform((y_pred, y_true))

    return to_onehot(preds, num_classes), targets, *other


def flatten_output_transform(output: tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    return [o.flatten(dim) for o in output]
