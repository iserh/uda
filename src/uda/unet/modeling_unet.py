"""U-Net implementation."""
from dataclasses import dataclass
from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_unet import UNetBackbones, UNetConfig


@dataclass
class NNModules:
    """`torch.nn` Modules with different dimensionality."""

    def __init__(self, dim: int = 2) -> None:
        """Initialize the `NNModules` class.

        Args:
            dim : The dimensionality of the modules. Defaults to 2.
        """
        if dim == 1:
            self.ConvNd = nn.Conv1d
            self.ConvTransposeNd = nn.ConvTranspose1d
            self.BatchNormNd = nn.BatchNorm1d
            self.max_poolNd = F.max_pool1d
        elif dim == 2:
            self.ConvNd = nn.Conv2d
            self.ConvTransposeNd = nn.ConvTranspose2d
            self.BatchNormNd = nn.BatchNorm2d
            self.max_poolNd = F.max_pool2d
        elif dim == 3:
            self.ConvNd = nn.Conv3d
            self.ConvTransposeNd = nn.ConvTranspose3d
            self.BatchNormNd = nn.BatchNorm3d
            self.max_poolNd = F.max_pool3d
        else:
            raise ValueError(f"Invalid dimensionality: {dim}")


class StackedConvBlock(nn.Module):
    """Stacked Convolutions backbone block."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Create a stack of convolutional layers with ReLU activations in between.

        Args:
            hidden_sizes : Number of channels in each convolutional layer.
            dim : Dimensionality of the input tensor.
        """
        super().__init__()
        nn_ = NNModules(dim)

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            conv = nn.Sequential(
                nn_.ConvNd(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.ReLU(),
            )
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the stack of convolutional layers.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        for conv in self.convs:
            x = conv(x)
        return x


class ResBlock(nn.Module):
    """Residual block with shortcut connection."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Create a residual block.

        Args:
            hidden_sizes : Number of channels in each convolutional layer.
            dim : Dimensionality of the input tensor.
        """
        super().__init__()
        nn_ = NNModules(dim)

        self.conv1 = nn_.ConvNd(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn_.BatchNormNd(hidden_sizes[1])
        self.conv2 = nn_.ConvNd(
            in_channels=hidden_sizes[1],
            out_channels=hidden_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn_.BatchNormNd(hidden_sizes[2])
        if hidden_sizes[0] != hidden_sizes[-1]:
            self.conv_shortcut = nn_.ConvNd(
                in_channels=hidden_sizes[0],
                out_channels=hidden_sizes[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.bn_shortcut = nn_.BatchNormNd(hidden_sizes[-1])
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv_shortcut is not None:
            identity = self.conv_shortcut(x)
            identity = self.bn_shortcut(identity)

        out += identity
        out = F.relu(out)
        return out


class ResNetBlock(nn.Module):
    """ResNet backbone block."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Create a ResNet backbone block.

        Args:
            hidden_sizes : Number of channels in each residual block.
            dim : Dimensionality of the input tensor.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            res_block = ResBlock(hidden_sizes=[in_channels, out_channels, out_channels], dim=dim)
            self.res_blocks.append(res_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet backbone block.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


def _get_backbone_class(backbone: str) -> Union[Type[StackedConvBlock], Type[ResNetBlock]]:
    """Get the class of the backbone.

    Args:
        backbone : Name of the backbone.

    Returns:
        The class of the backbone.
    """
    if backbone == UNetBackbones.StackedConvolutions:
        return StackedConvBlock
    elif backbone == UNetBackbones.ResNet:
        return ResNetBlock
    else:
        raise ValueError(f"Invalid backbone: {backbone}")


class UNetEncoder(nn.Module):
    """Encoder for the U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        """Create a U-Net encoder.

        Args:
            config : Configuration for the U-Net.
        """
        super().__init__()
        nn_ = NNModules(dim=config.dim)

        # encoder blocks
        Backbone = _get_backbone_class(config.encoder_backbone)
        self.blocks = nn.ModuleList([Backbone(channels, config.dim) for channels in config.encoder_blocks])

        # batch norm after last encoder block
        self.batch_norm = nn_.BatchNormNd(config.encoder_blocks[-1][-1]) if config.batch_norm_after_encoder else None
        self.max_poolNd = nn_.max_poolNd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        # apply initial encoder block
        x = self.blocks[0](x)

        hidden_states = []  # save hidden states for use in decoder
        for block in self.blocks[1:]:
            hidden_states.append(x)
            # max pooling downsampling
            x = self.max_poolNd(x, kernel_size=2, stride=2)
            x = block(x)

        # batch norm after last encoder block
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder for the U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        """Create a U-Net encoder.

        Args:
            config : Configuration for the U-Net.
        """
        super().__init__()
        nn_ = NNModules(dim=config.dim)

        Backbone = _get_backbone_class(config.encoder_backbone)
        self.blocks = nn.ModuleList([Backbone(b, config.dim) for b in config.decoder_blocks])

        # upsampling
        if config.bilinear_upsampling:
            raise NotImplementedError()
            # self.upsamples = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.up_convs = nn.ModuleList(
                [
                    nn_.ConvTransposeNd(
                        in_channels=channels[0],
                        out_channels=channels[0] // 2,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    )
                    for channels in config.decoder_blocks
                ]
            )

        # maps to classes
        self.mapping_conv = nn_.ConvNd(
            in_channels=config.decoder_blocks[-1][-1],
            out_channels=config.n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x : Input tensor.
            hidden_states : Hidden states from encoder.

        Returns:
            Output tensor.
        """
        for block, up_conv, hidden_state in zip(self.blocks, self.up_convs, reversed(hidden_states)):
            x = up_conv(x)
            x = torch.cat([x, hidden_state], dim=1)
            x = block(x)
        # map to classes
        x = self.mapping_conv(x)
        return x


class UNet(nn.Module):
    """Base class for all UNets."""

    def __init__(self, config: UNetConfig) -> None:
        """Create U-Net.

        Args:
            config : Configuration for the U-Net.
        """
        super().__init__()
        self.config = config
        self.encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D U-Net.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        return x

    def save(self, path: str) -> None:
        """Save the model.

        Args:
            path : Path to save the model to.
        """
        # save parameters
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config: UNetConfig) -> "UNet":
        """Load a pretrained model.

        Args:
            path : Path to load the model from.
            config : Configuration for the U-Net.
        """
        # load parameters
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
