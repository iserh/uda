from typing import Union

import torch
import torch.nn as nn


class ConvWithNorm(nn.Module):
    """Convolution + BatchNorm + ReLU."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        padding: int = 0,
        stride: int = 1,
        batch_norm: bool = True,
    ) -> None:
        super(ConvWithNorm, self).__init__()
        self.conv = ConvNd(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm = BatchNormNd(dim, out_channels) if batch_norm else nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Sequential):
    """Convolutional block."""

    def __init__(
        self,
        dim: int,
        channels: list[int],
        batch_norm: bool = True,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(ConvBlock, self).__init__(
            *[
                ConvWithNorm(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    batch_norm=batch_norm,
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )


class UpsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        dim: int,
        channels: list[int],
        batch_norm: bool = True,
        cut_channels_on_upsample: bool = False,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(UpsampleBlock, self).__init__()
        self.upsample = ConvTransposeNd(
            dim=dim,
            in_channels=channels[0],
            out_channels=channels[0] // 2 if cut_channels_on_upsample else channels[0],
            kernel_size=2,
            stride=2,
        )
        self.conv_block = ConvBlock(dim, channels, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class DownsampleBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        dim: int,
        channels: list[int],
        use_pooling: bool = False,
        batch_norm: bool = True,
    ) -> None:
        """Args:
        `channels` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(DownsampleBlock, self).__init__()
        if use_pooling:
            self.downsampling = MaxPoolNd(dim=dim, kernel_size=2, stride=2)
        else:
            self.downsampling = ConvNd(
                dim=dim,
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=2,
                stride=2,
            )
        self.conv_block = ConvBlock(dim, channels, batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsampling(x)
        x = self.conv_block(x)
        return x


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
