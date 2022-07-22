"""U-Net configuration."""
from dataclasses import dataclass
from typing import List

from uda.config import Config


@dataclass
class UNetConfig(Config):
    """Configuration for U-Net."""

    out_channels: int
    encoder_blocks: List[List[int]]
    decoder_blocks: List[List[int]]
    dim: int = 2
    concat_hidden: bool = False
    use_pooling: bool = False
