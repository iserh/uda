import torch
import torch.nn as nn
from ignite.utils import to_onehot


class CenterPad(nn.Module):
    """Center pad (or crop) nd-Tensor.

    See `uda.transforms.center_pad` for more information.
    """

    def __init__(self, *shape: int, offset: tuple[int, ...] = 0, mode: str = "constant", value: float = 0) -> None:
        super().__init__()
        self.shape = shape
        self.offset = offset
        self.mode = mode
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return center_pad(x, self.shape, self.offset, self.mode, self.value)


def center_pad(
    x: torch.Tensor,
    size: tuple[int, ...],
    offset: tuple[int, ...] = 0,
    mode: str = "constant",
    value: float = 0,
) -> torch.Tensor:
    """Center pad (or crop) nd-Tensor.

    Args:
        x (torch.Tensor): The Tensor to pad. The last `len(size)` dimensions will be padded.
        size (tuple[int, ...]): The desired pad size.
        offset (tuple[int, ...], optional): Shift the Tensor while padding. Defaults to :math:`0`.
        mode (str): Padding mode. Defaults to "constant".
        value (float): Padding value. Defaults to :math:`0`.
    """
    # offset gets subtracted from the left and added to the right
    offset = (torch.LongTensor([offset]) * torch.LongTensor([[-1], [1]])).flip(1)
    # compute the excess in each dim (negative val -> crop, positive val -> pad)
    excess = torch.Tensor([(size[-i] - x.shape[-i]) / 2 for i in range(1, len(size) + 1)])
    # floor excess on left side, ceil on right side, add offset
    pad = torch.stack([excess.floor(), excess.ceil()], dim=0).long() + offset

    return torch.nn.functional.pad(x, tuple(pad.T.flatten()), mode, value)


def binarize_prediction(t: torch.FloatTensor) -> torch.LongTensor:
    num_classes = t.shape[1]
    if num_classes == 1:  # binary segmentation
        return t.sigmoid().round()
    else:
        return to_onehot(t.argmax(1), num_classes).float()
