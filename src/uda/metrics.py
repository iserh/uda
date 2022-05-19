import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, dim: int = 0) -> torch.Tensor:
    pred = pred.flatten(dim)
    target = target.flatten(dim)
    intersection = (pred * target).sum(dim)
    # 2 * intersection / union
    return 2 * intersection / (pred.sum(dim) + target.sum(dim))
