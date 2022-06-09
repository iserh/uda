from typing import List, Tuple, Union

import torch
import torch.nn as nn


class ConvModule(torch.nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvNd(
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            BatchNormNd(dim, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(self, dim: int, channels: List[int]) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(ConvBlock, self).__init__()
        self.convs = nn.Sequential(
            *[
                ConvModule(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class UpsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, dim: int, channels: List[int]) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            # downsampling convolution
            ConvTransposeNd(dim=dim, in_channels=channels[0], out_channels=channels[0], kernel_size=2, stride=2),
            ConvBlock(dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(self, dim: int, channels: List[int]) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(DownsampleBlock, self).__init__()
        self.block = nn.Sequential(
            # downsampling convolution
            ConvNd(dim=dim, in_channels=channels[0], out_channels=channels[0], kernel_size=2, stride=2),
            ConvBlock(dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def ConvNd(dim: int, *args, **kwargs) -> Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"Conv{dim}d")
    return cls(*args, **kwargs)


def ConvTransposeNd(dim: int, *args, **kwargs) -> Union[nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"ConvTranspose{dim}d")
    return cls(*args, **kwargs)


def BatchNormNd(dim: int, *args, **kwargs) -> Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"BatchNorm{dim}d")
    return cls(*args, **kwargs)


def MaxPoolNd(dim: int, *args, **kwargs) -> Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]:
    assert dim > 0 and dim < 4, "Attribute 'dim' has to be >0 and <4."
    cls = getattr(nn, f"MaxPool{dim}d")
    return cls(*args, **kwargs)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.modules.conv._ConvNd):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
