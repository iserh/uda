"""U-Net implementation."""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .configuration_vae import VAEConfig
from .modules import ConvModule, ConvNd, DownsampleBlock, UpsampleBlock, init_weights


class VAEEncoder(nn.Module):
    """Encoder part of Variational Autoencoder."""

    def __init__(self, dim: int, in_channels: int, input_size: Tuple[int, ...], latent_dim: int, blocks: Tuple[Tuple[int, ...]]) -> None:
        super(VAEEncoder, self).__init__()

        self.in_block = ConvModule(
            dim=dim,
            in_channels=in_channels,
            out_channels=blocks[0][0],
            kernel_size=3,
            padding=1,
        )

        self.downsample_blocks = nn.Sequential(
            *[DownsampleBlock(dim, channels) for channels in blocks]
        )

        hidden_size = [blocks[-1][-1]] + [size // (2 ** len(blocks)) for size in input_size]

        self.fc_mean = nn.Linear(np.prod(hidden_size), latent_dim)
        self.fc_variance_log = nn.Linear(np.prod(hidden_size), latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_block(x)
        x = self.downsample_blocks(x)

        x = x.flatten(1)

        mean = self.fc_mean(x)
        variance_log = self.fc_variance_log(x)

        return mean, variance_log


class VAEDecoder(nn.Module):
    """Dencoder part of Variational Autoencoder."""

    def __init__(self, dim: int, out_channels: int, output_size: Tuple[int, ...], latent_dim: int, blocks: Tuple[Tuple[int, ...]]) -> None:
        super(VAEDecoder, self).__init__()

        self.hidden_size = [blocks[0][0]] + [size // (2 ** len(blocks)) for size in output_size]

        self.fc = nn.Linear(latent_dim, np.prod(self.hidden_size))

        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(dim, channels) for channels in blocks]
        )

        self.out_block = ConvNd(
            dim=dim,
            in_channels=blocks[-1][-1],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.reshape(-1, *self.hidden_size)
        x = self.upsample_blocks(x)
        x = self.out_block(x)
        return x


class VAE(nn.Module):
    """Variational Autoencoder."""

    def __init__(self, config: VAEConfig) -> None:
        super(VAE, self).__init__()
        self.config = config
        self.encoder = VAEEncoder(config.dim, config.n_channels, config.input_size, config.latent_dim, config.encoder_blocks)
        self.decoder = VAEEncoder(config.dim, config.n_channels, config.input_size, config.latent_dim, config.decoder_blocks)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, v_log = self.encoder(x)

        # reparametrization trick
        eps = torch.empty_like(mean).normal_()
        z = (v_log / 2).exp() * eps + mean

        x = self.decoder(z)

        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config: VAEConfig) -> "VAE":
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model
