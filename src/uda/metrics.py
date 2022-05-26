from typing import Optional, Tuple

import torch
from ignite.metrics import EpochMetric
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances


class EpochDice(EpochMetric):
    def __init__(
        self,
        check_compute_fn: bool = True,
        patch_dims: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.patch_dims = patch_dims
        super(SurfaceDice, self).__init__(self.compute_surface_dice_at_tolerance, flatten_output, check_compute_fn)

    def compute_surface_dice_at_tolerance(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.patch_dims is not None:
            # y_pred has shape []
            # unpatchify
            data = data.reshape(-1, np.prod(test_dataset.n_patches), *test_dataset.patchify)
            label = label.reshape(-1, np.prod(test_dataset.n_patches), *test_dataset.patchify)
            pred = pred.reshape(-1, np.prod(test_dataset.n_patches), *test_dataset.patchify)
            data = unpatchify(data, test_dataset.PADDING_SHAPE, start=2)
            label = unpatchify(label, test_dataset.PADDING_SHAPE, start=2)
            pred = unpatchify(pred, test_dataset.PADDING_SHAPE, start=2)

        pass


def dice_score(pred: torch.Tensor, target: torch.Tensor, dim: int = 0, square_denom: bool = False) -> torch.Tensor:
    pred = pred.flatten(dim)
    target = target.flatten(dim)

    num = 2 * (pred * target).sum(dim)

    if square_denom:
        denom = pred.pow(2).sum(dim) + target.pow(2).sum(dim)
    else:
        denom = pred.sum(dim) + target.sum(dim)

    return num / denom


class SurfaceDice(EpochMetric):
    def __init__(self, spacing_mm: torch.Tensor, tolerance_mm: float, check_compute_fn: bool = True) -> None:
        self.spacing_mm = spacing_mm
        self.tolerance_mm = tolerance_mm
        super(SurfaceDice, self).__init__(self.compute_surface_dice_at_tolerance, flatten_output, check_compute_fn)

    def compute_surface_dice_at_tolerance(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        surface_distances = compute_surface_distances(y.numpy(), y_pred.numpy(), self.spacing_mm)
        return torch.Tensor(compute_surface_dice_at_tolerance(surface_distances, self.tolerance_mm))


def flatten_output(output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    y_pred, y = output
    return y_pred.round().long().flatten(), y.flatten()
