"""U-Net configuration."""
from dataclasses import dataclass
from typing import List

from uda.config import Config

from .backbones import Backbone


@dataclass
class UNetConfig(Config):
    """Configuration for U-Net.

    `out_channels` : Number of output channels. If `n_channels=1` `Sigmoid` is used as final activation.
    `encoder_blocks` : Number of channels for each convolutional layer in each encoder block.
    `decoder_blocks` : Number of channels for each convolutional layer in each decoder block.
    `encoder_backbone` : Backbone to use for the encoder blocks.
    `decoder_backbone` : Backbone to use for the decoder blocks.
    `dim` : Dimensionality of model (U-Net 3D/2D/1D).
    `batch_norm` : Whether to use batch normalization after convolutions (Only relevant for Vanilla Backbone).
    """

    out_channels: int
    encoder_blocks: List[List[int]]
    decoder_blocks: List[List[int]]
    encoder_backbone: Backbone = Backbone.Vanilla
    decoder_backbone: Backbone = Backbone.Vanilla
    dim: int = 2
    batch_norm: bool = True
