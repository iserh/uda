"""U-Net configuration."""
from dataclasses import dataclass
from typing import Tuple

from uda.config import Config


@dataclass
class VAEConfig(Config):
    """Configuration for Variational Autoencoder."""

    input_size: Tuple[int, ...]
    encoder_blocks: Tuple[Tuple[int, ...], ...]
    decoder_blocks: Tuple[Tuple[int, ...], ...]
    latent_dim: int = 1024
    dim: int = 2
    use_pooling: bool = False
