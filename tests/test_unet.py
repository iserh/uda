"""Unit tests for the unet module."""
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from uda.unet import UNet, UNetBackbones, UNetConfig


@pytest.fixture
def device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def unet_config_default() -> UNetConfig:
    return UNetConfig(
        n_classes=2,
        encoder_blocks=[
            [1, 8, 8],
            [8, 16, 16],
            [16, 32, 32],
            [32, 64, 64],
            [64, 128, 128],
            [128, 256, 256],
        ],
        decoder_blocks=[
            [256, 128, 128],
            [128, 64, 64],
            [64, 32, 32],
            [32, 16, 16],
            [16, 8, 8],
        ],
    )


def test_unet_1d(unet_config_default: UNetConfig, device: str) -> None:
    # change the default configuration
    unet_config_default.dim = 1
    # create model
    model = UNet(unet_config_default)

    # create 1D input - (batch_size, n_channels, dim_1)
    x = torch.randn(8, 1, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64)


def test_unet_2d(unet_config_default: UNetConfig, device: str) -> None:
    # create model
    model = UNet(unet_config_default)

    # create 1D input - (batch_size, n_channels, dim_1, dim_2)
    x = torch.randn(8, 1, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64)


def test_unet_3d(unet_config_default: UNetConfig, device: str) -> None:
    # change the default configuration
    unet_config_default.dim = 3
    # create model
    model = UNet(unet_config_default)

    # create 1D input - (batch_size, n_channels, dim_1, dim_2, dim_3)
    x = torch.randn(8, 1, 64, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64, 64)


def test_unet_invalid_dimension(unet_config_default: UNetConfig) -> None:
    # change the default configuration
    unet_config_default.dim = 4
    # create model
    with pytest.raises(ValueError):
        UNet(unet_config_default)


def test_unet_res_net(unet_config_default: UNetConfig, device: str) -> None:
    # change the default configuration
    unet_config_default.dim = 2
    unet_config_default.encoder_backbone = UNetBackbones.ResNet
    unet_config_default.decoder_backbone = UNetBackbones.ResNet
    unet_config_default.batch_norm_after_encoder = False
    # create model
    model = UNet(unet_config_default)

    # create 1D input - (batch_size, n_channels, dim_1, dim_2)
    x = torch.randn(8, 1, 64, 64)
    # move to device
    model = model.to(device)
    x = x.to(device)
    # test model
    y = model(x).detach().cpu()

    assert y.shape == (8, 2, 64, 64)


def test_unet_invalid_backbone(unet_config_default: UNetConfig) -> None:
    # change the default configuration
    unet_config_default.encoder_backbone = "InvalidBackbone"
    # create model
    with pytest.raises(ValueError):
        UNet(unet_config_default)


def test_bilinear_decoder(unet_config_default: UNetConfig) -> None:
    # change the default configuration
    unet_config_default.bilinear_upsampling = True
    # create model
    with pytest.raises(NotImplementedError):
        UNet(unet_config_default)


def test_unet_save_load(unet_config_default: UNetConfig) -> None:
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # create model
        model_pre_save = UNet(unet_config_default)
        unet_config_default.save(tmpdir / "unet_config.json")
        model_pre_save.save(tmpdir / "unet_model.pt")

        assert (tmpdir / "unet_config.json").exists()
        assert (tmpdir / "unet_model.pt").exists()

        # load model
        config_post_load = UNetConfig.from_file(tmpdir / "unet_config.json")
        model_post_load = UNet.from_pretrained(tmpdir / "unet_model.pt", config_post_load)

        assert model_post_load.config.__dict__ == unet_config_default.__dict__
