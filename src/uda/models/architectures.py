from math import log2
from typing import Tuple

from .configuration_unet import UNetConfig
from .configuration_vae import VAEConfig


def vanilla_unet(
    in_channels: int,
    out_channels: int,
    n_blocks: int = 5,
    start_channels: int = 64,
    dim: int = 2,
    concat_hidden: bool = True,
    use_pooling: bool = True,
) -> UNetConfig:
    """Vanilla U-Net configuration.

    `in_channels` -> 64 -> 128 -> 256 -> 512 -> 1024 -> ... -> `out_channels`
    """

    start_base = int(log2(start_channels))

    encoder_blocks = [[in_channels, start_channels, start_channels]] + [
        [2**i, 2 ** (i + 1), 2 ** (i + 1)] for i in range(start_base, start_base + n_blocks - 1)
    ]

    decoder_blocks = [[2**i, 2 ** (i - 1), 2 ** (i - 1)] for i in range(start_base + n_blocks - 1, start_base, -1)]

    return UNetConfig(
        out_channels=out_channels,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        dim=dim,
        concat_hidden=concat_hidden,
        use_pooling=use_pooling,
    )


def uda_unet(
    in_channels: int, out_channels: int, n_blocks: int = 5, start_channels: int = 8, dim: int = 2
) -> UNetConfig:
    """U-Net from UDA Paper."""

    start_base = int(log2(start_channels))

    encoder_blocks = [[in_channels, start_channels]] + [
        [2**i, 2 ** (i + 1), 2 ** (i + 1), 2 ** (i + 1)] for i in range(start_base, start_base + n_blocks - 1)
    ]

    decoder_blocks = [
        [2**i, 2 ** (i - 1), 2 ** (i - 1), 2 ** (i - 1)] for i in range(start_base + n_blocks - 1, start_base, -1)
    ]

    return UNetConfig(
        out_channels=out_channels,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        dim=dim,
        concat_hidden=False,
        use_pooling=False,
    )


def uda_vae(
    input_size: Tuple[int, ...],
    in_channels: int,
    latent_dim: int = 1024,
    n_blocks: int = 6,
    start_channels: int = 8,
    dim: int = 2,
) -> UNetConfig:
    """U-Net from UDA Paper."""

    start_base = int(log2(start_channels))

    encoder_blocks = [[in_channels, start_channels]] + [
        [2**i, 2 ** (i + 1), 2 ** (i + 1), 2 ** (i + 1)] for i in range(start_base, start_base + n_blocks - 1)
    ]

    decoder_blocks = [
        [2**i, 2 ** (i - 1), 2 ** (i - 1), 2 ** (i - 1)] for i in range(start_base + n_blocks - 1, start_base, -1)
    ]

    return VAEConfig(
        input_size=input_size,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        latent_dim=latent_dim,
        dim=dim,
        use_pooling=False,
    )
