"""U-Net implementation."""
from math import floor
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .configuration_vae import VAEConfig
from .modules import ConvBlock, ConvNd, DownsampleBlock, UpsampleBlock, init_weights


def compute_hidden_size(
    n_blocks: int, input_size: tuple[int, ...], kernel_size: int = 2, padding: int = 0, stride: int = 2
) -> list[int]:
    K, P, S = kernel_size, padding, stride

    hidden_sizes = []
    for size in input_size:
        W = size
        hidden_sizes.append([(W := floor((W - K + 2 * P) / S + 1)) for _ in range(n_blocks)][-1])  # noqa: F841

    return hidden_sizes


class VAEEncoder(nn.Module):
    """Encoder part of Variational Autoencoder."""

    def __init__(
        self,
        dim: int,
        input_size: tuple[int, ...],
        latent_dim: int,
        blocks: tuple[tuple[int, ...]],
        use_pooling: bool = False,
        batch_norm: bool = True,
        track_running_stats: bool = False,
    ) -> None:
        super(VAEEncoder, self).__init__()
        # hidden_size = [ #channels , *img_sizes_after_final_downsampling ]
        hidden_size = [blocks[-1][-1]] + compute_hidden_size(len(blocks[1:]), input_size)

        self.in_block = ConvBlock(dim, blocks[0], batch_norm, track_running_stats)
        self.downsample_blocks = nn.Sequential(
            *[DownsampleBlock(dim, channels, use_pooling, batch_norm, track_running_stats) for channels in blocks[1:]]
        )
        self.mean = nn.Linear(np.prod(hidden_size), latent_dim)
        self.variance_log = nn.Linear(np.prod(hidden_size), latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_block(x)
        x = self.downsample_blocks(x)
        x = x.flatten(1)

        mean = self.mean(x)
        variance_log = self.variance_log(x)

        return mean, variance_log


class VAEDecoder(nn.Module):
    """Dencoder part of Variational Autoencoder."""

    def __init__(
        self,
        dim: int,
        out_channels: int,
        output_size: tuple[int, ...],
        latent_dim: int,
        blocks: tuple[tuple[int, ...]],
        batch_norm: bool = True,
        track_running_stats: bool = False,
    ) -> None:
        super(VAEDecoder, self).__init__()
        # hidden_size = [ #channels , *img_sizes_after_final_upsampling ]
        self.hidden_size = [blocks[0][0]] + [size // (2 ** len(blocks)) for size in output_size]

        self.linear = nn.Linear(latent_dim, np.prod(self.hidden_size))
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(dim, channels, batch_norm, track_running_stats) for channels in blocks]
        )
        self.out_block = ConvNd(
            dim=dim,
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = x.reshape(-1, *self.hidden_size)
        x = self.upsample_blocks(x)
        x = self.out_block(x)
        return x


class VAE(nn.Module):
    """Variational Autoencoder."""

    def __init__(self, config: VAEConfig) -> None:
        super(VAE, self).__init__()
        self.config = config

        self.encoder = VAEEncoder(
            config.dim,
            config.input_size,
            config.latent_dim,
            config.encoder_blocks,
            config.use_pooling,
            config.batch_norm,
            config.track_running_stats,
        )
        self.decoder = VAEDecoder(
            config.dim,
            config.encoder_blocks[0][0],
            config.input_size,
            config.latent_dim,
            config.decoder_blocks,
            config.batch_norm,
            config.track_running_stats,
        )

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, v_log = self.encoder(x)

        if self.training:
            # reparametrization trick
            eps = torch.empty_like(mean).normal_()
            z = (v_log / 2).exp() * eps + mean
        else:
            z = mean

        x_rec = self.decoder(z)

        return x_rec, mean, v_log

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: Union[Path, str], config: Optional[Union[Path, str, VAEConfig]] = None) -> "VAE":
        path = Path(path)
        if config is None:
            config = VAEConfig.from_file(path.parent / "model.yaml")
        elif not isinstance(config, VAEConfig):
            config = VAEConfig.from_file(config)

        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
