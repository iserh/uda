"""U-Net configuration."""
from dataclasses import dataclass
from typing import ClassVar

from ..config import Config


@dataclass
class VAEConfig(Config):
    """Configuration for Variational Autoencoder."""

    name: ClassVar[str] = "VAE"

    input_size: tuple[int, ...]
    encoder_blocks: tuple[tuple[int, ...], ...]
    decoder_blocks: tuple[tuple[int, ...], ...]
    latent_dim: int = 1024
    dim: int = 2
    use_pooling: bool = False
    batch_norm: bool = False
    track_running_stats: bool = True
