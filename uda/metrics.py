import torch
from ignite.utils import to_onehot
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Computes the class-wise dice coefficient.

    Args:
        y_pred (torch.Tensor): Expected shape :math:`(N, C, ...)`, where :math:`N` is the batch size and
            :math:`C` is the number of classes. :math:`C=1` is interpreted as the foreground
            class of a binary segmentation.
        y_true (torch.Tensor): If containing class indices, shape :math:`(N)`.
            If containing class probabilities, same shape as the input and each value should be
            between :math:`[0, 1]`.
        eps (float, optional): Smoothing to avoid NaN's. Defaults to :math:`1e-15`.

    Returns:
        torch.Tensor: Shape :math:`(C)`
    """
    # check targets given as indices
    if len(y_true.shape) == len(y_pred.shape) - 1:
        # convert to onehot
        y_true = to_onehot(y_true, y_pred.shape[1])

    assert y_pred.shape == y_true.shape

    # flatten
    y_pred = y_pred.transpose(1, 0).flatten(1)
    y_true = y_true.transpose(1, 0).flatten(1)

    num = 2 * (y_pred * y_true).sum(1)
    denom = y_pred.sum(1) + y_true.sum(1)

    return (num + eps) / (denom + eps)


def surface_dice(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    spacings_mm: torch.Tensor,
    tolerance_mm: float,
    prog_bar: bool = True,
) -> torch.Tensor:
    """Computes the class-wise surface-dice.

    Args:
        y_pred (torch.Tensor): Expected shape :math:`(N, C, ...)`, where :math:`N` is the batch size and
            :math:`C` is the number of classes. :math:`C=1` is interpreted as the foreground
            class of a binary segmentation.
        y_true (torch.Tensor): If containing class indices, shape :math:`(N)`.
            If containing class probabilities, same shape as the input and each value should be
            between :math:`[0, 1]`.
        eps (float, optional): Smoothing to avoid NaN's. Defaults to :math:`1e-15`.

    Returns:
        torch.Tensor: Shape :math:`(C)`
    """
    # check targets given as indices
    if len(y_true.shape) == len(y_pred.shape) - 1:
        # convert to onehot
        y_true = to_onehot(y_true, y_pred.shape[1])

    assert y_pred.shape == y_true.shape

    batch_size, num_classes = y_pred.shape[:2]
    # collect batch & classes in one axis
    y_pred = y_pred.reshape(-1, *y_pred.shape[2:])
    y_true = y_true.reshape(-1, *y_true.shape[2:])
    spacings_mm = spacings_mm.repeat_interleave(num_classes, 0)

    iterator = (
        tqdm(zip(y_pred, y_true, spacings_mm), total=len(y_pred), desc="Computing surface dice", leave=False)
        if prog_bar
        else zip(y_pred, y_true, spacings_mm)
    )

    sf_dice_vals = torch.Tensor(
        [
            compute_surface_dice_at_tolerance(
                compute_surface_distances(mask_gt.bool().numpy(), mask_pred.bool().numpy(), spacing_mm), tolerance_mm
            )
            for mask_pred, mask_gt, spacing_mm in iterator
        ]
    )

    return sf_dice_vals.reshape(batch_size, num_classes)
