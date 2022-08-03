"""U-Net configuration."""
from dataclasses import dataclass

from uda.config import Config


@dataclass
class UNetConfig(Config):
    """Configuration for U-Net."""

    out_channels: int
    encoder_blocks: list[list[int]]
    decoder_blocks: list[list[int]]
    dim: int = 2
    concat_hidden: bool = False
    use_pooling: bool = False
