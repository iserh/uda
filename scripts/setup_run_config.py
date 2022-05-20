from pathlib import Path

from uda import HParamsConfig, UNetBackbones, UNetConfig
from uda.datasets import CC359Config

config_dir = Path("config")

ds_config = CC359Config(
    vendor="GE_3",
    fold=1,
    rotate=True,
    flatten=False,
    patchify=[64, 256, 256],
    flatten_patches=True,
    clip_intensities=None,
    random_state=None,
)

unet_config = UNetConfig(
    out_channels=1,
    dim=3,
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
    encoder_backbone=UNetBackbones.StackedConvolutions,
    decoder_backbone=UNetBackbones.StackedConvolutions,
)

hparams = HParamsConfig(
    epochs=10,
    criterion="dice_loss",
    learning_rate=1e-4,
    optim="Adam",
    batch_size=4,
    test_interval=100,
)

ds_config.save(config_dir / "cc359.json")
unet_config.save(config_dir / "unet.json")
hparams.save(config_dir / "hparams.json")
