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
    y_pred = output[0]
    other = output[1:] if len(output) > 1 else []

    if y_pred.shape[1] == 1:
        preds = y_pred.sigmoid().round().long()
    else:
        preds = y_pred.argmax(1)

    return preds, *other


def one_hot_output_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred = output[0]
    other = output[1:] if len(output) > 1 else []

    num_classes = y_pred.shape[1]
    y_pred = get_preds_output_transform((y_pred,))

    return to_onehot(y_pred, num_classes), *other


def flatten_output_transform(output: tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    return [o.flatten(dim) for o in output]
