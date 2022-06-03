"""U-Net configuration."""
from dataclasses import dataclass
from enum import Enum
from typing import List

import torch.nn as nn

from uda.config import Config

from .backbones_unet import _UNetBackbones


class UNetBackbones(str, Enum):
    Vanilla = _UNetBackbones.Vanilla.name
    ResNet = _UNetBackbones.ResNet.name


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
    encoder_backbone: UNetBackbones = UNetBackbones.Vanilla
    decoder_backbone: UNetBackbones = UNetBackbones.Vanilla
    dim: int = 2
    batch_norm: bool = False

    def get_encoder_backbone(self) -> nn.Module:
        return _UNetBackbones[self.encoder_backbone].value

    def get_decoder_backbone(self) -> nn.Module:
        return _UNetBackbones[self.decoder_backbone].value
