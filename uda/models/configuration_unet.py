"""U-Net configuration."""
from dataclasses import dataclass
from typing import Optional

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
    batch_norm: bool = True
    track_running_stats: Optional[bool] = None
    model_name: str = "UNet"
