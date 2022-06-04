from typing import Optional, Tuple

import numpy as np
import torch
from ignite.metrics import EpochMetric
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from .utils import unpatchify


class EpochDice(EpochMetric):
    def __init__(
        self,
        orig_shape: Tuple[int, int, int],
        reduce_mean: bool = True,
        patch_dims: Optional[Tuple[int, int, int]] = None,
        check_compute_fn: bool = False,
    ) -> None:
        super(EpochDice, self).__init__(self.compute_fn, flatten_output, check_compute_fn)
        self.orig_shape = orig_shape
        self.reduce_mean = reduce_mean
        self.patch_dims = patch_dims

    def compute_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.patch_dims is not None:
            # move n_patches in seperate dimension
            n_patches = [int(s / pd) for pd, s in zip(self.patch_dims, self.orig_shape)]
            y_pred = y_pred.reshape(-1, np.prod(n_patches), *self.patch_dims)
            # unpatchify
            y_pred = unpatchify(y_pred, self.orig_shape, start=2)

        # reshape to 3d volume
        y_pred = y_pred.reshape(-1, *self.orig_shape)
        y_true = y_true.reshape(-1, *self.orig_shape)

        return dice_score(y_pred, y_true, 0 if self.reduce_mean else 1)


class SurfaceDice(EpochMetric):
    def __init__(
        self,
        spacing_mm: torch.Tensor,
        tolerance_mm: float,
        orig_shape: Tuple[int, int, int],
        reduce_mean: bool = True,
        patch_dims: Optional[Tuple[int, int, int]] = None,
        check_compute_fn: bool = False,
        prog_bar: bool = False,
    ) -> None:
        super(SurfaceDice, self).__init__(self.compute_fn, flatten_output, check_compute_fn)
        self.spacing_mm = spacing_mm
        self.tolerance_mm = tolerance_mm
        self.orig_shape = orig_shape
        self.reduce_mean = reduce_mean
        self.patch_dims = patch_dims
        self.prog_bar = prog_bar

    def compute_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.patch_dims is not None:
            # move n_patches in seperate dimension
            n_patches = [int(s / pd) for pd, s in zip(self.patch_dims, self.orig_shape)]
            y_pred = y_pred.reshape(-1, np.prod(n_patches), *self.patch_dims)
            # unpatchify
            y_pred = unpatchify(y_pred, self.orig_shape, start=2)

        # reshape to 3d volume
        y_pred = y_pred.reshape(-1, *self.orig_shape)
        y_true = y_true.reshape(-1, *self.orig_shape)

        sfdice_values = surface_dice(y_pred, y_true, self.spacing_mm, self.tolerance_mm, self.prog_bar)

        return sfdice_values.mean() if self.reduce_mean else sfdice_values


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor, dim: int = 0, square_denom: bool = False) -> torch.Tensor:
    y_pred = y_pred.flatten(dim)
    y_true = y_true.flatten(dim)

    num = 2 * (y_pred * y_true).sum(dim)

    if square_denom:
        denom = y_pred.pow(2).sum(dim) + y_true.pow(2).sum(dim)
    else:
        denom = y_pred.sum(dim) + y_true.sum(dim)

    return num / denom


def surface_dice(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    spacing_mm: Tuple[int, int, int],
    tolerance_mm: float,
    prog_bar: bool = False,
) -> torch.Tensor:
    iterator = (
        tqdm(zip(y_pred, y_true, spacing_mm), total=len(y_pred), desc="Computing surface dice", leave=False)
        if prog_bar
        else zip(y_pred, y_true, spacing_mm)
    )

    surface_dice_vals = torch.Tensor(
        [
            compute_surface_dice_at_tolerance(
                compute_surface_distances(y_.bool().numpy(), y_pred_.bool().numpy(), spacing), tolerance_mm
            )
            for y_pred_, y_, spacing in iterator
        ]
    )

    return surface_dice_vals


def flatten_output(output: Tuple[torch.Tensor, torch.Tensor], dim: int = 0) -> None:
    y_pred, y_true = output
    return y_pred.round().long().flatten(dim), y_true.flatten(dim)
