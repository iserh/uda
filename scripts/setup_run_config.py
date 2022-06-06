from pathlib import Path

from uda import CC359Config, HParams, LossCriterion, Optimizer, UNetConfig

config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

dataset_config = CC359Config(
    vendor="GE_3",
    fold=1,
    flatten=True,
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
    dice_pow=True,
    learning_rate=1e-4,
    optimizer=Optimizer.Adam,
    train_batch_size=24,
    val_batch_size=24,
)

dataset_config.save(config_dir / "cc359.yaml")
unet_config.save(config_dir / "unet.yaml")
hparams.save(config_dir / "hparams.yaml")
