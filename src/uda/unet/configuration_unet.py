"""U-Net configuration."""
import json
from enum import Enum
from typing import List


class UNetBackbones(str, Enum):
    """Backbones for the U-Net."""

    StackedConvolutions = "StackedConvolutions"
    ResNet = "ResNet"


class UNetConfig:
    """Configuration for U-Net."""

    def __init__(
        self,
        out_channels: int,
        encoder_blocks: List[List[int]],
        decoder_blocks: List[List[int]],
        encoder_backbone: UNetBackbones = UNetBackbones.StackedConvolutions,
        decoder_backbone: UNetBackbones = UNetBackbones.StackedConvolutions,
        dim: int = 2,
        batch_norm_after_encoder: bool = True,
        bilinear_upsampling: bool = False,
    ) -> None:
        """Args:
        `out_channels` : Number of output channels. If `n_channels=1` `Sigmoid` is used as final activation.
        `encoder_blocks` : Number of channels for each convolutional layer in each encoder block.
        `decoder_blocks` : Number of channels for each convolutional layer in each decoder block.
        `encoder_backbone` : Backbone to use for the encoder blocks.
        `decoder_backbone` : Backbone to use for the decoder blocks.
        `dim` : Dimensionality of model (U-Net 3D/2D/1D).
        `batch_norm_after_encoder` : Whether to use batch normalization after last encoder block.
        `bilinear_upsampling` : Whether to use bilinear upsampling instead of transposed convolution.
        """
        self.out_channels = out_channels
        self.encoder_blocks = encoder_blocks
        self.decoder_blocks = decoder_blocks
        self.encoder_backbone = encoder_backbone
        self.decoder_backbone = decoder_backbone
        self.dim = dim
        self.batch_norm_after_encoder = batch_norm_after_encoder
        self.bilinear_upsampling = bilinear_upsampling

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_file(cls, path: str) -> "UNetConfig":
        with open(path, "r") as f:
            config = cls(**json.load(f))
        return config
