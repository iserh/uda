import torch

from .metrics import dice_score


def dice_loss(pred: torch.Tensor, target: torch.Tensor, square_denom: bool = False) -> torch.Tensor:
    return 1 - dice_score(pred, target, square_denom=square_denom)
