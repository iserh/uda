"""U-Net implementation."""
from typing import List, Tuple

import torch
import torch.nn as nn

from .configuration_unet import UNetConfig
from .modules import ConvModule, ConvNd, DownsampleBlock, UpsampleBlock, init_weights


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(self, dim: int, in_channels: int, blocks: Tuple[Tuple[int, ...]]) -> None:
        super(UNetEncoder, self).__init__()

        self.in_block = ConvModule(
            dim=dim,
            in_channels=in_channels,
            out_channels=blocks[0][0],
            kernel_size=3,
            padding=1,
        )

        self.downsample_blocks = nn.ModuleList([DownsampleBlock(dim, channels) for channels in blocks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        hidden_states = []
        for block in self.downsample_blocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(self, dim: int, out_channels: int, blocks: Tuple[Tuple[int, ...]]) -> None:
        super(UNetDecoder, self).__init__()
        self.upsample_blocks = nn.ModuleList([UpsampleBlock(dim, channels) for channels in blocks])

        self.out_block = ConvNd(
            dim=dim,
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.upsample_blocks, reversed(hidden_states)):
            x = block(x) + h

        return self.out_block(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNet, self).__init__()
        self.config = config
        self.encoder = UNetEncoder(config.dim, config.in_channels, config.encoder_blocks)
        self.decoder = UNetDecoder(config.dim, config.out_channels, config.decoder_blocks)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config: UNetConfig) -> "UNet":
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
