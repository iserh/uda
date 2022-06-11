from pathlib import Path

from uda import CC359Config, HParams, LossCriterion, Optimizer
from uda.models import uda_unet

config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

dataset_config = CC359Config(
    vendor="GE_3",
    fold=0,
    # flatten=True,
    patch_size=(64, 256, 256),
    random_state=42,
)
dataset_config.save(config_dir / "cc359.yaml")

hparams = HParams(
    epochs=60,
    criterion=LossCriterion.Dice,
    loss_kwargs={},
    learning_rate=1e-4,
    optimizer=Optimizer.Adam,
    train_batch_size=4,
    val_batch_size=4,
)
hparams.save(config_dir / "hparams.yaml")

unet_config = uda_unet(1, 1, dim=3)
unet_config.save(config_dir / "unet.yaml")
