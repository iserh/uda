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
        if dim == 1:
            self.ConvNd = nn.Conv1d
            self.ConvTransposeNd = nn.ConvTranspose1d
            self.BatchNormNd = nn.BatchNorm1d
            self.MaxPoolNd = nn.MaxPool1d
        elif dim == 2:
            self.ConvNd = nn.Conv2d
            self.ConvTransposeNd = nn.ConvTranspose2d
            self.BatchNormNd = nn.BatchNorm2d
            self.MaxPoolNd = nn.MaxPool2d
        elif dim == 3:
            self.ConvNd = nn.Conv3d
            self.ConvTransposeNd = nn.ConvTranspose3d
            self.BatchNormNd = nn.BatchNorm3d
            self.MaxPoolNd = nn.MaxPool3d
        else:
            raise ValueError(f"Invalid dimensionality: {dim}")

        self.bias = False


class StackedConvBlock(nn.Module):
    """U-Net block consisting of a convolutional stack with ReLU activations in between."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super().__init__()
        nn_ = NNModules(dim)

        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            conv = nn_.ConvNd(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=nn_.bias,
            )
            self.convs.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.relu(conv(x))

        return x


class ResBlock(nn.Module):
    """Residual block with shortcut connection."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each convolutional layer.
        `dim` : Dimensionality of the input tensor.
        """
        super().__init__()
        nn_ = NNModules(dim)

        self.conv1 = nn_.ConvNd(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=nn_.bias,
        )
        self.bn1 = nn_.BatchNormNd(hidden_sizes[1])
        self.conv2 = nn_.ConvNd(
            in_channels=hidden_sizes[1],
            out_channels=hidden_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=nn_.bias,
        )
        self.bn2 = nn_.BatchNormNd(hidden_sizes[2])
        if hidden_sizes[0] != hidden_sizes[-1]:
            self.conv_shortcut = nn_.ConvNd(
                in_channels=hidden_sizes[0],
                out_channels=hidden_sizes[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=nn_.bias,
            )
            self.bn_shortcut = nn_.BatchNormNd(hidden_sizes[-1])
        else:
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """U-Net block consisting of ResNet."""

    def __init__(self, hidden_sizes: List[int], dim: int) -> None:
        """Args:
        `hidden_sizes` : Number of channels in each residual block.
        `dim` : Dimensionality of the input tensor.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            res_block = ResBlock(hidden_sizes=[in_channels, out_channels, out_channels], dim=dim)
            self.res_blocks.append(res_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


def _get_backbone_class(backbone: str) -> Union[Type[StackedConvBlock], Type[ResNetBlock]]:
    if backbone == UNetBackbones.StackedConvolutions:
        return StackedConvBlock
    elif backbone == UNetBackbones.ResNet:
        return ResNetBlock
    else:
        raise ValueError(f"Invalid backbone: {backbone}")


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        nn_ = NNModules(dim=config.dim)

        # encoder blocks
        Backbone = _get_backbone_class(config.encoder_backbone)
        self.blocks = nn.ModuleList([Backbone(channels, config.dim) for channels in config.encoder_blocks])
        self.max_pool = nn_.MaxPoolNd(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply initial encoder block
        x = self.blocks[0](x)

        hidden_states = []  # save hidden states for use in decoder
        for block in self.blocks[1:]:
            hidden_states.append(x)
            x = self.max_pool(x)
            x = block(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
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
                        bias=nn_.bias,
                    )
                    for channels in config.decoder_blocks
                ]
            )

        # maps to classes
        self.mapping_conv = nn_.ConvNd(
            in_channels=config.decoder_blocks[-1][-1],
            out_channels=config.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=nn_.bias,
        )
        self.final_act = nn.Sigmoid() if config.out_channels == 1 else nn.Softmax(dim=1)
        # cropping
        self.cropping = CropNd(dim=config.dim)

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        for block, up_conv, hidden_state in zip(self.blocks, self.up_convs, reversed(hidden_states)):
            x = up_conv(x)
            # crop and concatenate
            # hidden_state = self.cropping(hidden_state, x.size())
            x = torch.cat([x, hidden_state], dim=1)
            x = block(x)
        # map to classes
        x = self.mapping_conv(x)
        return self.final_act(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config)

        # initialize weights !!
        init_weights = WeightInitializer(dim=config.dim)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        # use xavier initialization mapping convolution because of sigmoid activation
        self.decoder.mapping_conv.apply(lambda m: nn.init.xavier_uniform_(m.weight))

    def forward(self, x: torch.Tensor, ret_hidden_states: bool = False) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        return x if not ret_hidden_states else (x, hidden_states)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config: UNetConfig) -> "UNet":
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model


class WeightInitializer:
    def __init__(self, dim: int) -> None:
        self.nn_ = NNModules(dim=dim)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, self.nn_.ConvNd) or isinstance(m, self.nn_.ConvTransposeNd):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, self.nn_.BatchNormNd):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def __call__(self, m: nn.Module) -> None:
        self.init_weights(m)


class CropNd(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim == 1:
            self.crop = crop1d
        elif dim == 2:
            self.crop = crop2d
        elif dim == 3:
            self.crop = crop3d
        else:
            raise ValueError(f"Invalid dimensionality: {dim}")

    def forward(self, x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return self.crop(x, shape)


def crop1d(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if x.shape == shape:
        return x
    else:
        return x[..., : shape[-1]]


def crop2d(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if x.shape == shape:
        return x
    else:
        return x[..., : shape[-2], : shape[-1]]


def crop3d(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if x.shape == shape:
        return x
    else:
        return x[..., : shape[-3], : shape[-2], : shape[-1]]
