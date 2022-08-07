"""U-Net implementation."""
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from .configuration_unet import UNetConfig
from .modules import ConvBlock, ConvNd, DownsampleBlock, UpsampleBlock, center_crop_nd, init_weights


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(
        self,
        dim: int,
        blocks: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
        track_running_stats: bool = False,
    ) -> None:
        super(UNetEncoder, self).__init__()
        self.in_block = ConvBlock(dim, blocks[0], batch_norm, track_running_stats)
        self.downsample_blocks = nn.ModuleList(
            [DownsampleBlock(dim, channels, use_pooling, batch_norm, track_running_stats) for channels in blocks[1:]]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_block(x)

        hidden_states = []
        for block in self.downsample_blocks:
            hidden_states.append(x)
            x = block(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(
        self,
        dim: int,
        out_channels: int,
        blocks: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        track_running_stats: bool = False,
        concat_hidden: bool = False,
    ) -> None:
        super(UNetDecoder, self).__init__()
        self.upsample_blocks = nn.ModuleList(
            [UpsampleBlock(dim, channels, batch_norm, track_running_stats, concat_hidden) for channels in blocks]
        )
        self.out_block = ConvNd(
            dim=dim,
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.concat_hidden = concat_hidden

    def forward(self, x: torch.Tensor, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        for block, h in zip(self.upsample_blocks, reversed(hidden_states)):
            if self.concat_hidden:
                x = block.upsample(x)
                h = center_crop_nd(h, x.shape[1:])
                x = torch.cat([x, h], dim=1)
                x = block.conv_block(x)
            else:
                x = block(x)
                h = center_crop_nd(h, x.shape[1:])
                x = x + h

        return self.out_block(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNet, self).__init__()
        self.config = config

        self.encoder = UNetEncoder(
            config.dim, config.encoder_blocks, config.use_pooling, config.batch_norm, config.track_running_stats
        )
        self.decoder = UNetDecoder(
            config.dim,
            config.out_channels,
            config.decoder_blocks,
            config.batch_norm,
            config.track_running_stats,
            config.concat_hidden,
        )

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: Union[Path, str], config: Optional[Union[Path, str, UNetConfig]] = None) -> "UNet":
        path = Path(path)
        if config is None:
            config = UNetConfig.from_file(path.parent / "model.yaml")
        elif not isinstance(config, UNetConfig):
            config = UNetConfig.from_file(config)

        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
