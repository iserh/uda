"""Unit tests for the unet module."""
from typing import Tuple

import pytest
import torch

from uda.unet import Decoder, ResNetBlock, Unet1D, Unet2D, Unet3D


@pytest.fixture
def device() -> str:
    """Return the device to use for testing.

    Returns:
        str: The device to use for testing.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def encoder_blocks() -> Tuple[Tuple[int]]:
    """Return the encoder blocks to use for testing.

    Returns:
        Tuple[Tuple[int]]: The encoder blocks to use for testing.
    """
    return (
        (1, 8, 8),
        (8, 16, 16),
        (16, 32, 32),
        (32, 64, 64),
        (64, 128, 128),
        (128, 256, 256),
    )


@pytest.fixture
def decoder_blocks() -> Tuple[Tuple[int]]:
    """Return the decoder blocks to use for testing.

    Returns:
        Tuple[Tuple[int]]: The decoder blocks to use for testing.
    """
    return (
        (256, 128, 128),
        (128, 64, 64),
        (64, 32, 32),
        (32, 16, 16),
        (16, 8, 8),
    )


def test_unet_1d(encoder_blocks: Tuple[Tuple[int]], decoder_blocks: Tuple[Tuple[int]], device: str) -> None:
    """Test the Unet1D model.

    Args:
        encoder_blocks : The encoder blocks to use for testing.
        decoder_blocks : The decoder blocks to use for testing.
        device : The device to use for testing.
    """
    # create model
    model = Unet1D(
        n_classes=2,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
    )

    # create 1D input - (batch_size, n_channels, dim_1)
    x = torch.randn(8, 1, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64)


def test_unet_2d(encoder_blocks: Tuple[Tuple[int]], decoder_blocks: Tuple[Tuple[int]], device: str) -> None:
    """Test the Unet2D model.

    Args:
        encoder_blocks : The encoder blocks to use for testing.
        decoder_blocks : The decoder blocks to use for testing.
        device : The device to use for testing.
    """
    # create model
    model = Unet2D(
        n_classes=2,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
    )

    # create 1D input - (batch_size, n_channels, dim_1, dim_2)
    x = torch.randn(8, 1, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64)


def test_unet_3d(encoder_blocks: Tuple[Tuple[int]], decoder_blocks: Tuple[Tuple[int]], device: str) -> None:
    """Test the Unet3D model.

    Args:
        encoder_blocks : The encoder blocks to use for testing.
        decoder_blocks : The decoder blocks to use for testing.
        device : The device to use for testing.
    """
    # create model
    model = Unet3D(
        n_classes=2,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
    )

    # create 1D input - (batch_size, n_channels, dim_1, dim_2, dim_3)
    x = torch.randn(8, 1, 64, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64, 64)


def test_unet_res_net(encoder_blocks: Tuple[Tuple[int]], decoder_blocks: Tuple[Tuple[int]], device: str) -> None:
    """Test the Unet3D model with ResNet backbones.

    Args:
        encoder_blocks : The encoder blocks to use for testing.
        decoder_blocks : The decoder blocks to use for testing.
        device : The device to use for testing.
    """
    # create model
    model = Unet2D(
        n_classes=2,
        encoder_blocks=encoder_blocks,
        decoder_blocks=decoder_blocks,
        encoder_backbone=ResNetBlock,
        decoder_backbone=ResNetBlock,
        batch_norm_after_encoder=False,
    )

    # create 1D input - (batch_size, n_channels, dim_1, dim_2)
    x = torch.randn(8, 1, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64)


def test_bilinear_decoder(decoder_blocks: Tuple[Tuple[int]]) -> None:
    """Test bilinear decoder.

    Args:
        decoder_blocks : The decoder blocks to use for testing.
    """
    with pytest.raises(NotImplementedError):
        Decoder(
            n_classes=2,
            blocks=decoder_blocks,
            bilinear=True,
        )
