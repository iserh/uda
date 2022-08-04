"""U-Net implementation."""
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from .configuration_unet import UNetConfig
from .modules import ConvBlock, ConvNd, DownsampleBlock, UpsampleBlock, init_weights


class UNetEncoder(nn.ModuleDict):
    """Encoder part of U-Net."""

    def __init__(self, dim: int, blocks: tuple[tuple[int, ...]], use_pooling: bool = False) -> None:
        super(UNetEncoder, self).__init__(
            {
                "InBlock": ConvBlock(dim, blocks[0]),
                "DownsampleBlocks": nn.ModuleList(
                    [DownsampleBlock(dim, channels, use_pooling) for channels in blocks[1:]]
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.InBlock(x)

        hidden_states = []
        for block in self.DownsampleBlocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states


class UNetDecoder(nn.ModuleDict):
    """Decoder part of U-Net."""

    def __init__(
        self, dim: int, out_channels: int, blocks: tuple[tuple[int, ...]], concat_hidden: bool = False
    ) -> None:
        super(UNetDecoder, self).__init__(
            {
                "UpsampleBlocks": nn.ModuleList([UpsampleBlock(dim, channels, concat_hidden) for channels in blocks]),
                "OutBlock": ConvNd(
                    dim=dim,
                    in_channels=blocks[-1][-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            }
        )

        self.concat_hidden = concat_hidden

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.UpsampleBlocks, reversed(hidden_states)):
            if self.concat_hidden:
                x = block.Upsample(x)
                x = torch.cat([x, h], dim=1)
                x = block.ConvBlock(x)
            else:
                x = block(x) + h

        return self.OutBlock(x)


class UNet(nn.ModuleDict):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNet, self).__init__(
            {
                "Encoder": UNetEncoder(config.dim, config.encoder_blocks, config.use_pooling),
                "Decoder": UNetDecoder(config.dim, config.out_channels, config.decoder_blocks, config.concat_hidden),
            }
        )
        self.config = config

        self.Encoder.apply(init_weights)
        self.Decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden_states = self.Encoder(x)
        x = self.Decoder(x, hidden_states)
        return x

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: Union[Path, str]) -> "UNet":
        model_dict = torch.load(path / "best_model.pt")
        model = cls(model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])

        return model
