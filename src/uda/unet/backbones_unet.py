"""U-Net backbones."""
from enum import Enum
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BatchNormNd, ConvNd


class VanillaBlock(nn.Module):
    """U-Net block consisting of a convolutional stack with ReLU activations in between."""

    def __init__(self, hidden_sizes: List[int], dim: int, batch_norm: bool = True) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(VanillaBlock, self).__init__()

        self.convs = nn.ModuleList(
            [
                ConvNd(
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    # bias=False,
                )
                for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:])
            ]
        )

        if batch_norm:
            self.batch_norms = nn.ModuleList([BatchNormNd(dim, hidden_size) for hidden_size in zip(hidden_sizes[1:])])
        else:
            self.batch_norms = nn.ModuleList([nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x)
            x = batch_norm(x)
            x = F.relu(x, inplace=True)

        return x


class ResBlock(nn.Module):
    """Residual block with shortcut connection."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super(ResBlock, self).__init__()

        self.conv1 = ConvNd(
            dim=dim,
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
            # bias=False,
        )
        self.bn1 = BatchNormNd(dim, hidden_sizes[1])
        self.conv2 = ConvNd(
            dim=dim,
            in_channels=hidden_sizes[1],
            out_channels=hidden_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
            # bias=False,
        )
        self.bn2 = BatchNormNd(dim, hidden_sizes[2])
        if hidden_sizes[0] != hidden_sizes[-1]:
            self.conv_shortcut = ConvNd(
                dim=dim,
                in_channels=hidden_sizes[0],
                out_channels=hidden_sizes[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                # bias=False,
            )
            self.bn_shortcut = BatchNormNd(dim, hidden_sizes[-1])
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv_shortcut is not None:
            identity = self.conv_shortcut(x)
            identity = self.bn_shortcut(identity)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class ResNetBlock(nn.Module):
    """U-Net block consisting of ResNet."""

    def __init__(self, hidden_sizes: List[int], dim: int, batch_norm: bool = True) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each residual block.
        `dim` : Dimensionality of the input tensor.
        """
        super(ResNetBlock, self).__init__()

        if not batch_norm:
            raise NotImplementedError("ResNet without BatchNorm not implemented.")

        self.res_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            res_block = ResBlock(hidden_sizes=[in_channels, out_channels, out_channels], dim=dim)
            self.res_blocks.append(res_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


class _UNetBackbones(Enum):
    """Backbones for the U-Net."""

    Vanilla = VanillaBlock
    ResNet = ResNetBlock
