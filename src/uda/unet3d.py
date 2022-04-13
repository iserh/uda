"""Unet 3D Network."""
from typing import List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

DIMS = 3
CONVOLUTION = nn.Conv2d if DIMS == 2 else nn.Conv3d
MAX_POOL = F.max_pool2d if DIMS == 2 else F.max_pool3d
UP_CONVOLUTION = nn.ConvTranspose2d if DIMS == 2 else nn.ConvTranspose3d
BATCH_NORM = nn.BatchNorm2d if DIMS == 2 else nn.BatchNorm3d


class StackedConvBlock(nn.Module):
    """Stacked Convolutions backbone block."""

    def __init__(self, hidden_sizes: Tuple[int]) -> None:
        """Create a stack of convolutional layers with ReLU activations in between.

        Args:
            hidden_sizes : Number of channels in each convolutional layer.
        """
        super().__init__()
        self.convs = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            conv = nn.Sequential(
                CONVOLUTION(
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

    def __init__(self, hidden_sizes: Tuple[int]) -> None:
        """Create a residual block.

        Args:
            hidden_sizes : Number of channels in each convolutional layer.
        """
        super().__init__()
        self.conv1 = CONVOLUTION(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = BATCH_NORM(hidden_sizes[1])
        self.conv2 = CONVOLUTION(
            in_channels=hidden_sizes[1],
            out_channels=hidden_sizes[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = BATCH_NORM(hidden_sizes[2])
        if hidden_sizes[0] != hidden_sizes[-1]:
            self.conv_shortcut = CONVOLUTION(
                in_channels=hidden_sizes[0],
                out_channels=hidden_sizes[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.bn_shortcut = BATCH_NORM(hidden_sizes[-1])
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

    def __init__(self, hidden_sizes: Tuple[int]) -> None:
        """Create a ResNet backbone block.

        Args:
            hidden_sizes : Number of channels in each residual block.
        """
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for in_channels, out_channels in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            res_block = ResBlock(hidden_sizes=[in_channels, out_channels, out_channels])
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


class Encoder(nn.Module):
    """Encoder for the U-Net."""

    def __init__(
        self, blocks: Tuple[Tuple[int]], Backbone: Type[nn.Module] = StackedConvBlock, batch_norm: bool = True
    ) -> None:
        """Create a U-Net encoder.

        Args:
            blocks : Channel sizes for each encoder block.
            Backbone : Backbone to use for the encoder blocks.
            batch_norm : Whether to use batch normalization after last encoder block.
        """
        super().__init__()
        # encoder blocks
        self.blocks = nn.ModuleList([Backbone(channels) for channels in blocks])
        # batch norm after last encoder block
        self.batch_norm = BATCH_NORM(blocks[-1][-1]) if batch_norm else None

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
            x = MAX_POOL(x, kernel_size=2, stride=2)
            x = block(x)

        # batch norm after last encoder block
        if self.batch_norm is not None:
            x = self.batch_norm(x)

        return x, hidden_states


class Decoder(nn.Module):
    """Decoder for the U-Net."""

    def __init__(
        self,
        blocks: Tuple[Tuple[int]],
        n_classes: int,
        Backbone: Type[nn.Module] = StackedConvBlock,
        bilinear: bool = False,
    ) -> None:
        """Create a U-Net decoder.

        Args:
            blocks : Channel sizes for each decoder block.
            n_classes : Number of classes in the output.
            Backbone : Backbone to use for the decoder blocks.
            bilinear: Whether to use bilinear upsampling.
        """
        super().__init__()
        self.blocks = nn.ModuleList([Backbone(b) for b in blocks])

        # upsampling
        if bilinear:
            raise NotImplementedError()
            # self.upsamples = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.up_convs = nn.ModuleList(
                [
                    UP_CONVOLUTION(
                        in_channels=channels[0],
                        out_channels=channels[0] // 2,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    )
                    for channels in blocks
                ]
            )

        # maps to classes
        self.mapping_conv = CONVOLUTION(
            in_channels=blocks[-1][-1],
            out_channels=n_classes,
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


class Unet3D(nn.Module):
    """3D U-Net."""

    def __init__(
        self,
        encoder_blocks: Tuple[Tuple[int]],
        decoder_blocks: Tuple[Tuple[int]],
        n_classes: int,
        encoder_backbone: Type[nn.Module] = StackedConvBlock,
        decoder_backbone: Type[nn.Module] = StackedConvBlock,
        batch_norm_after_encoder: bool = True,
    ) -> None:
        """Create a 3D U-Net.

        Args:
            encoder_blocks : Number of channels for each convolutional layer in each encoder block.
            decoder_blocks : Number of channels for each convolutional layer in each decoder block.
            n_classes : Number of classes in the output.
            encoder_backbone : Backbone to use for the encoder blocks.
            decoder_backbone : Backbone to use for the decoder blocks.
            batch_norm_after_encoder : Whether to use batch normalization after last encoder block.
        """
        super().__init__()
        self.encoder = Encoder(encoder_blocks, encoder_backbone, batch_norm_after_encoder)
        self.decoder = Decoder(decoder_blocks, n_classes, decoder_backbone)

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


if __name__ == "__main__":
    from torchsummaryX import summary

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # StackedConvolutions U-Net (like paper)
    model = Unet3D(
        n_classes=2,
        encoder_blocks=(
            (1, 8, 8, 8),
            (8, 8, 16, 16, 16),
            (16, 16, 32, 32, 32),
            (32, 32, 64, 64, 64),
            (64, 64, 128, 128, 128),
            (128, 128, 256, 256, 256),
        ),
        decoder_blocks=(
            (256, 128, 128, 128),
            (128, 64, 64, 64),
            (64, 32, 32, 32),
            (32, 16, 16, 16),
            (16, 8, 8, 8),
        ),
    )

    # # ResNet UNet
    # model = Unet3D(
    #     n_classes=2,
    #     encoder_blocks=(
    #         (1, 8, 8),
    #         (8, 16, 16),
    #         (16, 32, 32),
    #         (32, 64, 64),
    #         (64, 128, 128),
    #         (128, 256, 256),
    #     ),
    #     decoder_blocks=(
    #         (256, 128),
    #         (128, 64),
    #         (64, 32),
    #         (32, 16),
    #         (16, 8),
    #     ),
    #     encoder_backbone=ResNetBlock,
    #     decoder_backbone=ResNetBlock,
    #     batch_norm_after_encoder=False,
    # )

    model = model.to(device)
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    # test model
    y = model(x).detach().cpu()
    print("Works! Output Shape: ", y.shape)
    print()

    # model summary
    df = summary(model, x).reset_index()
    # print only convolutional layers
    print(df[df["Layer"].apply(lambda x: "Conv3d" in x and "conv_shortcut" not in x)])
