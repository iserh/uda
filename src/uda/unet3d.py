"""Unet 3D Network."""
from typing import List, Tuple

import torch
import torch.nn as nn

DIMS = 3
CONVOLUTION = nn.Conv2d if DIMS == 2 else nn.Conv3d
MAX_POOL = nn.MaxPool2d if DIMS == 2 else nn.MaxPool3d
UP_CONVOLUTION = nn.ConvTranspose2d if DIMS == 2 else nn.ConvTranspose3d
BATCH_NORM = nn.BatchNorm2d if DIMS == 2 else nn.BatchNorm3d


class ConvolutionStack(nn.Module):
    """Stacks convolutional layers with ReLU activation."""

    def __init__(self, hidden_sizes: Tuple[int], **kwargs) -> None:
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
                    kernel_size=kwargs.get("kernel_size", 3),
                    stride=kwargs.get("stride", 1),
                    padding=kwargs.get("padding", 1),
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


class EncoderBlock(nn.Module):
    """Encoder block for the U-Net."""

    def __init__(self, hidden_sizes: Tuple[int]) -> None:
        """Max Pooling followed by a stack of convolutional layers.

        Args:
            hidden_sizes : Channel sizes of the convolutional layers.
        """
        super().__init__()
        self.pooling = MAX_POOL(kernel_size=2, stride=2)
        self.convolutions = ConvolutionStack(hidden_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        x = self.pooling(x)
        x = self.convolutions(x)
        return x


class Encoder(nn.Module):
    """Encoder for the U-Net."""

    def __init__(self, layers: Tuple[Tuple[int]]) -> None:
        """Create a U-Net encoder.

        Args:
            layers : Number of channels for each convolutional layer in each encoder block.
        """
        super().__init__()
        self.input_convs = ConvolutionStack(layers[0])
        # first encoder block has no pooling
        self.blocks = nn.ModuleList([EncoderBlock(l) for l in layers[1:]])
        # batch norm after last encoder block
        self.batch_norm = BATCH_NORM(layers[-1][-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x : Input tensor.

        Returns:
            Output tensor.
        """
        x = self.input_convs(x)

        hidden_states = []  # save hidden states for use in decoder
        for block in self.blocks:
            hidden_states.append(x)
            x = block(x)

        # batch norm after last encoder block
        x = self.batch_norm(x)

        return x, hidden_states


class DecoderBlock(nn.Module):
    """Decoder block for the U-Net."""

    def __init__(self, hidden_sizes: Tuple[int]) -> None:
        """Create a U-Net decoder block.

        Args:
            hidden_sizes : Channel sizes of the convolutional layers.
        """
        super().__init__()
        self.up_conv = UP_CONVOLUTION(
            in_channels=hidden_sizes[0],
            out_channels=hidden_sizes[0] // 2,
            kernel_size=2,
            stride=2,
        )
        self.convs = ConvolutionStack(hidden_sizes)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder block.

        Args:
            x : Input tensor.
            hidden_state : Hidden state from encoder block.

        Returns:
            Output tensor.
        """
        x = self.up_conv(x)
        x = torch.cat([x, hidden_state], dim=1)
        x = self.convs(x)
        return x


class Decoder(nn.Module):
    """Decoder for the U-Net."""

    def __init__(self, layers: Tuple[Tuple[int]], n_classes: int) -> None:
        """Create a U-Net decoder.

        Args:
            layers : Number of channels for each convolutional layer in each decoder block.
            n_classes : Number of classes in the output.
        """
        super().__init__()
        self.blocks = nn.ModuleList([DecoderBlock(l) for l in layers])
        self.mapping_conv = CONVOLUTION(
            in_channels=layers[-1][-1],
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x : Input tensor.
            hidden_states : Hidden states from encoder.

        Returns:
            Output tensor.
        """
        for block in self.blocks:
            x = block(x, hidden_states.pop())
        x = self.mapping_conv(x)
        return x


class Unet3D(nn.Module):
    """3D U-Net."""

    def __init__(self, encoder_layers: Tuple[Tuple[int]], decoder_layers: Tuple[Tuple[int]], n_classes: int) -> None:
        """Create a 3D U-Net.

        Args:
            encoder_layers : Number of channels for each convolutional layer in each encoder block.
            decoder_layers : Number of channels for each convolutional layer in each decoder block.
            n_classes : Number of classes in the output.
        """
        super().__init__()
        self.encoder = Encoder(encoder_layers)
        self.decoder = Decoder(decoder_layers, n_classes)

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

    encoder_layers = (
        (1, 8, 8, 8),
        (8, 8, 16, 16, 16),
        (16, 16, 32, 32, 32),
        (32, 32, 64, 64, 64),
        (64, 64, 128, 128, 128),
        (128, 128, 256, 256, 256),
    )
    decoder_layers = (
        (256, 128, 128, 128),
        (128, 64, 64, 64),
        (64, 32, 32, 32),
        (32, 16, 16, 16),
        (16, 8, 8, 8),
    )

    model = Unet3D(encoder_layers, decoder_layers, n_classes=2).to(device)

    x = torch.randn(56, 1, 64, 64, 64).to(device)

    summary(model, x)
    print()

    y = model(x).detach().cpu()
    print("Works! Output Shape: ", y.shape)
