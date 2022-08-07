from typing import Optional, Union

import numpy as np
import torch
from ignite.metrics import EpochMetric
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from .utils import flatten_output_transform, pipe, reshape_to_volume, sigmoid_round_output_transform


class DiceScore(EpochMetric):
    def __init__(
        self,
        dim: int,
        imsize: tuple[int, int, int],
        patch_size: Optional[tuple[int, int, int]] = None,
        reduce_mean: bool = True,
        check_compute_fn: bool = False,
    ) -> None:
        output_transform = pipe(sigmoid_round_output_transform, flatten_output_transform)
        super(DiceScore, self).__init__(self.compute_fn, output_transform, check_compute_fn)
        self.dim = dim
        self.imsize = imsize
        self.patch_size = patch_size
        self.reduce_mean = reduce_mean

    def compute_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = reshape_to_volume(y_pred, self.dim, self.imsize, self.patch_size)
        y_true = reshape_to_volume(y_true, self.dim, self.imsize, self.patch_size)

        return dice_score(y_pred, y_true, axis=1 if self.reduce_mean else 0)


class SurfaceDice(EpochMetric):
    def __init__(
        self,
        spacings_mm: torch.Tensor,
        tolerance_mm: float,
        dim: int,
        imsize: tuple[int, int, int],
        patch_size: Optional[tuple[int, int, int]] = None,
        reduce_mean: bool = True,
        prog_bar: bool = True,
        check_compute_fn: bool = False,
    ) -> None:
        output_transform = pipe(sigmoid_round_output_transform, flatten_output_transform)
        super(SurfaceDice, self).__init__(self.compute_fn, output_transform, check_compute_fn)
        self.spacings_mm = spacings_mm
        self.tolerance_mm = tolerance_mm
        self.dim = dim
        self.imsize = imsize
        self.patch_size = patch_size
        self.reduce_mean = reduce_mean
        self.prog_bar = prog_bar

    def compute_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = reshape_to_volume(y_pred, self.dim, self.imsize, self.patch_size)
        y_true = reshape_to_volume(y_true, self.dim, self.imsize, self.patch_size)

        sf_dice_vals = surface_dice(y_pred, y_true, self.spacings_mm, self.tolerance_mm, self.prog_bar)

        return sf_dice_vals.mean() if self.reduce_mean else sf_dice_vals


def dice_score(
    y_pred: Union[np.ndarray, torch.Tensor], y_true: Union[np.ndarray, torch.Tensor], axis: int = 0
) -> Union[np.ndarray, torch.Tensor]:
    # flatten
    y_pred = y_pred.reshape(*y_pred.shape[:axis], -1)
    y_true = y_true.reshape(*y_true.shape[:axis], -1)

    num = 2 * (y_pred * y_true).sum(axis)
    denom = y_pred.sum(axis) + y_true.sum(axis)

    return num / denom


def surface_dice(
    preds: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    spacings_mm: Union[np.ndarray, torch.Tensor],
    tolerance_mm: float,
    prog_bar: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    # check if torch.Tensor (surface-distance uses numpy backend)
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
        output_type_tensor = True
    else:
        output_type_tensor = False
    if isinstance(preds, torch.Tensor):
        targets = targets.numpy()
    if isinstance(preds, torch.Tensor):
        spacings_mm = spacings_mm.numpy()

    iterator = (
        tqdm(zip(preds, targets, spacings_mm), total=len(preds), desc="Computing surface dice", leave=False)
        if prog_bar
        else zip(preds, targets, spacings_mm)
    )

    sf_dice_vals = np.array(
        [
            compute_surface_dice_at_tolerance(
                compute_surface_distances(y_true.astype(bool), y_pred_.astype(bool), spacing_mm), tolerance_mm
            )
            for y_pred_, y_true, spacing_mm in iterator
        ]
    )

    return torch.from_numpy(sf_dice_vals) if output_type_tensor else sf_dice_vals
