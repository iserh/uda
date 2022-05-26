from typing import Optional, Tuple

import numpy as np
import torch
from ignite.metrics import EpochMetric
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from .utils import unpatchify


def flatten_output(output: Tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    y_pred, y = output
    return y_pred.round().long().flatten(dim), y.flatten(dim)


def dice_score(pred: torch.Tensor, target: torch.Tensor, dim: int = 0, square_denom: bool = False) -> torch.Tensor:
    pred = pred.flatten(dim)
    target = target.flatten(dim)

    num = 2 * (pred * target).sum(dim)

    if square_denom:
        denom = pred.pow(2).sum(dim) + target.pow(2).sum(dim)
    else:
        denom = pred.sum(dim) + target.sum(dim)

    return num / denom


def EpochDice(
    reduce_mean: bool = True,
    orig_shape: Optional[Tuple[int, int, int]] = None,
    patch_dims: Optional[Tuple[int, int, int]] = None,
    check_compute_fn: bool = False,
) -> EpochMetric:
    def compute_fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if patch_dims is not None:
            # move n_patches in seperate dimension
            n_patches = [int(s / pd) for pd, s in zip(patch_dims, orig_shape)]
            y_pred = y_pred.reshape(-1, np.prod(n_patches), *patch_dims)
            # unpatchify
            y_pred = unpatchify(y_pred, orig_shape, start=2)
            y = y.reshape(-1, *orig_shape)

        return dice_score(y_pred, y, 0 if reduce_mean else 1)

    def output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return flatten_output(output, dim=0 if reduce_mean else 1)

    return EpochMetric(compute_fn, output_transform, check_compute_fn)


def SurfaceDice(
    spacing_mm: torch.Tensor,
    tolerance_mm: float,
    orig_shape: Tuple[int, int, int],
    reduce_mean: bool = True,
    patch_dims: Optional[Tuple[int, int, int]] = None,
    check_compute_fn: bool = False,
    prog_bar: bool = False,
) -> EpochMetric:
    def compute_fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if patch_dims is not None:
            # move n_patches in seperate dimension
            n_patches = [int(s / pd) for pd, s in zip(patch_dims, orig_shape)]
            y_pred = y_pred.reshape(-1, np.prod(n_patches), *patch_dims)
            # unpatchify
            y_pred = unpatchify(y_pred, orig_shape, start=2)

        # reshape to 3d volume
        y_pred = y_pred.reshape(-1, *orig_shape)
        y = y.reshape(-1, *orig_shape)

        iterator = (
            tqdm(zip(y_pred, y, spacing_mm), total=len(y_pred), desc="Computing surface dice", leave=False)
            if prog_bar
            else zip(y_pred, y, spacing_mm)
        )

        surface_dice_vals = torch.Tensor(
            [
                compute_surface_dice_at_tolerance(
                    compute_surface_distances(y_.bool().numpy(), y_pred_.bool().numpy(), spacing), tolerance_mm
                )
                for y_pred_, y_, spacing in iterator
            ]
        )

        return surface_dice_vals.mean() if reduce_mean else surface_dice_vals

    return EpochMetric(compute_fn, flatten_output, check_compute_fn)
