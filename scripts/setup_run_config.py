from pathlib import Path

from uda import CC359Config, HParams, LossCriterion, Optimizer, UNetConfig

config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

ds_config = CC359Config(
    vendor="GE_3",
    fold=1,
    flatten=True,
    # patch_dims=[64, 256, 256],
    random_state=42,
)

unet_config = UNetConfig(
    out_channels=1,
    dim=2,
    encoder_blocks=((1, 8),),
    decoder_blocks=((8, 8),),
)

hparams = HParams(
    epochs=2,
    criterion=LossCriterion.Dice,
    square_dice_denom=True,
    learning_rate=1e-4,
    optimizer=Optimizer.Adam,
    train_batch_size=24,
    val_batch_size=24,
)

ds_config.save(config_dir / "cc359.yml")
unet_config.save(config_dir / "unet.yml")
hparams.save(config_dir / "hparams.yml")
