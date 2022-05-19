import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, dim: int = 0, square_denom: bool = False) -> torch.Tensor:
    pred = pred.flatten(dim)
    target = target.flatten(dim)

    num = 2 * (pred * target).sum(dim)

    if square_denom:
        denom = pred.pow(2).sum(dim) + target.pow(2).sum(dim)
    else:
        denom = pred.sum(dim) + target.sum(dim)

    return num / denom
