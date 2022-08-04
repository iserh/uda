"""U-Net implementation."""
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .configuration_vae import VAEConfig
from .modules import ConvBlock, ConvNd, DownsampleBlock, UpsampleBlock, init_weights


class VAEEncoder(nn.ModuleDict):
    """Encoder part of Variational Autoencoder."""

    def __init__(
        self,
        dim: int,
        input_size: tuple[int, ...],
        latent_dim: int,
        blocks: tuple[tuple[int, ...]],
        use_pooling: bool = False,
    ) -> None:
        hidden_size = [blocks[-1][-1]] + [size // (2 ** len(blocks[1:])) for size in input_size]

        super(VAEEncoder, self).__init__(
            {
                "InBlock": ConvBlock(dim, blocks[0]),
                "DownsampleBlocks": nn.Sequential(
                    *[DownsampleBlock(dim, channels, use_pooling) for channels in blocks[1:]]
                ),
                "Mean": nn.Linear(np.prod(hidden_size), latent_dim),
                "VarianceLog": nn.Linear(np.prod(hidden_size), latent_dim),
            }
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.InBlock(x)
        x = self.DownsampleBlocks(x)
        x = x.flatten(1)

        mean = self.Mean(x)
        variance_log = self.VarianceLog(x)

        return mean, variance_log


class VAEDecoder(nn.ModuleDict):
    """Dencoder part of Variational Autoencoder."""

    def __init__(
        self, dim: int, out_channels: int, output_size: tuple[int, ...], latent_dim: int, blocks: tuple[tuple[int, ...]]
    ) -> None:
        self.hidden_size = [blocks[0][0]] + [size // (2 ** len(blocks)) for size in output_size]

        super(VAEDecoder, self).__init__(
            {
                "Linear": nn.Linear(latent_dim, np.prod(self.hidden_size)),
                "UpsampleBlocks": nn.Sequential(*[UpsampleBlock(dim, channels) for channels in blocks]),
                "OutBlock": ConvNd(
                    dim=dim,
                    in_channels=blocks[-1][-1],
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Linear(x)
        x = x.reshape(-1, *self.hidden_size)
        x = self.UpsampleBlocks(x)
        x = self.OutBlock(x)
        return x


class VAE(nn.ModuleDict):
    """Variational Autoencoder."""

    def __init__(self, config: VAEConfig) -> None:
        super(VAE, self).__init__(
            {
                "Encoder": VAEEncoder(
                    config.dim, config.input_size, config.latent_dim, config.encoder_blocks, config.use_pooling
                ),
                "Decoder": VAEDecoder(
                    config.dim, config.encoder_blocks[0][0], config.input_size, config.latent_dim, config.decoder_blocks
                ),
            }
        )

        self.config = config

        self.Encoder.apply(init_weights)
        self.Decoder.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, v_log = self.Encoder(x)

        if self.training:
            # reparametrization trick
            eps = torch.empty_like(mean).normal_()
            z = (v_log / 2).exp() * eps + mean
        else:
            z = mean

        x_rec = self.Decoder(z)

        return x_rec, mean, v_log

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: Union[Path, str]) -> "VAE":
        model_dict = torch.load(path / "best_model.pt")
        model = cls(model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])

        return model
