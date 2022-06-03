"""U-Net implementation."""
from typing import List

import torch
import torch.nn as nn

from .configuration_unet import UNetConfig
from .modules import ConvNd, ConvTransposeNd, MaxPoolNd


class UNetEncoder(nn.Module):
    """Encoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNetEncoder, self).__init__()
        BackboneClass = config.get_encoder_backbone()
        # encoder blocks
        self.blocks = nn.ModuleList([BackboneClass(channels, config.dim) for channels in config.encoder_blocks])
        self.max_pool = MaxPoolNd(dim=config.dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply initial encoder block
        x = self.blocks[0](x)

        hidden_states = []  # save hidden states for use in decoder
        for block in self.blocks[1:]:
            hidden_states.append(x)
            x = self.max_pool(x)
            x = block(x)

        return x, hidden_states


class UNetDecoder(nn.Module):
    """Decoder part of U-Net."""

    def __init__(self, config: UNetConfig) -> None:
        super(UNetDecoder, self).__init__()
        BackboneClass = config.get_decoder_backbone()
        self.blocks = nn.ModuleList([BackboneClass(b, config.dim) for b in config.decoder_blocks])

        self.up_convs = nn.ModuleList(
            [
                ConvTransposeNd(
                    dim=config.dim,
                    in_channels=channels[0],
                    out_channels=channels[0] // 2,
                    kernel_size=2,
                    stride=2,
                    bias=False,
                )
                for channels in config.decoder_blocks
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
            bias=False,
        )
        self.final_act = nn.Sigmoid() if config.out_channels == 1 else nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        for block, up_conv, hidden_state in zip(self.blocks, self.up_convs, reversed(hidden_states)):
            x = up_conv(x)
            x = torch.cat([x, hidden_state], dim=1)
            x = block(x)
        # map to classes
        x = self.mapping_conv(x)
        return self.final_act(x)


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
        self.decoder.mapping_conv.apply(lambda m: nn.init.xavier_uniform_(m.weight))

    def forward(self, x: torch.Tensor, ret_hidden_states: bool = False) -> torch.Tensor:
        x, hidden_states = self.encoder(x)
        x = self.decoder(x, hidden_states)
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
