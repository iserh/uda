from pathlib import Path

from uda import HParams, UNetConfig
from uda.datasets import CC359Config

config_dir = Path("config")

ds_config = CC359Config(
    vendor="GE_3",
    fold=1,
    rotate=True,
    flatten=False,
    patch_dims=[64, 256, 256],
    flatten_patches=True,
)

unet_config = UNetConfig(
    out_channels=1,
    dim=3,
    encoder_blocks=((1, 8),),
    decoder_blocks=((8, 8),),
)

hparams = HParams(
    epochs=2,
    criterion="dice_loss",
    learning_rate=1e-4,
    optim="Adam",
    train_batch_size=4,
    val_batch_size=4,
)

ds_config.save(config_dir / "cc359.yml")
unet_config.save(config_dir / "unet.yml")
hparams.save(config_dir / "hparams.yml")
