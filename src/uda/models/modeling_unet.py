"""U-Net implementation."""
from typing import List

import torch
import torch.nn as nn

from .configuration_unet import UNetConfig
from .modules import ConvNd, ConvTransposeNd, MaxPoolNd
from .backbones import get_backbone_impl


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNetEncoder, self).__init__()
        BackboneClass = get_backbone_impl(config.encoder_backbone)
        # encoder blocks
        self.blocks = nn.ModuleList(
            [BackboneClass(channels, config.dim, config.batch_norm) for channels in config.encoder_blocks]
        )
        self.max_pool = MaxPoolNd(dim=config.dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply initial encoder block
        x = self.blocks[0](x)

        hidden_states = [x]  # save hidden states for use in decoder
        for block in self.blocks[1:]:
            x = self.max_pool(x)
            x = block(x)
            hidden_states.append(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNetDecoder, self).__init__()
        BackboneClass = get_backbone_impl(config.decoder_backbone)

        # edit decoder blocks to add encoder channels to first part of the block
        decoder_blocks = []
        for enc_block, dec_block in zip(reversed(config.encoder_blocks[:-1]), config.decoder_blocks):
            dec_block = dec_block.copy()
            dec_block[0] += enc_block[-1]
            decoder_blocks.append(dec_block)

        self.blocks = nn.ModuleList([BackboneClass(b, config.dim, config.batch_norm) for b in decoder_blocks])

        # first upwards convolution goes from encoder output to first decoder block
        up_conv_channels = [(config.encoder_blocks[-1][-1], config.decoder_blocks[0][0])]
        up_conv_channels += [
            (low_block[-1], high_block[0])
            for low_block, high_block in zip(config.decoder_blocks[:-1], config.decoder_blocks[1:])
        ]
        # upwards convolutions
        self.up_convs = nn.ModuleList(
            [
                ConvTransposeNd(
                    dim=config.dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    stride=2,
                    # bias=False,
                )
                for in_channels, out_channels in up_conv_channels
            ]
        )

        # maps to classes
        self.mapping_conv = ConvNd(
            dim=config.dim,
            in_channels=config.decoder_blocks[-1][-1],
            out_channels=config.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            # bias=False,
        )

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        for block, up_conv, hidden_state in zip(self.blocks, self.up_convs, reversed(hidden_states)):
            x = up_conv(x)
            x = torch.cat([x, hidden_state], dim=1)
            x = block(x)
        # map to classes
        return self.mapping_conv(x)


class UNet(nn.Module):
    """U-Net segmentation model."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNet, self).__init__()
        self.config = config
        self.encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config)

        # initialize weights !!
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        # use xavier initialization mapping convolution because of sigmoid activation
        # self.decoder.mapping_conv.apply(lambda m: nn.init.xavier_uniform_(m.weight))

    def forward(self, x: torch.Tensor, ret_hidden_states: bool = False) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states[:-1])
        return x if not ret_hidden_states else (x, hidden_states)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str, config: UNetConfig) -> "UNet":
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return model


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.modules.conv._ConvNd):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
