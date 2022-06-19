from pathlib import Path

from uda import CC359Config, HParams, LossCriterion, Optimizer
from uda.models.architectures import uda_vae

config_dir = Path("config")
config_dir.mkdir(exist_ok=True)

dataset_config = CC359Config(
    vendor="GE_3",
    fold=0,
    # flatten=True,
    # imsize=(128, 256, 256),
    patch_size=(64, 256, 256),
    random_state=42,
)
dataset_config.save(config_dir / "cc359.yaml")

hparams = HParams(
    epochs=80,
    criterion=LossCriterion.BCE,
    optimizer=Optimizer.Adam,
    learning_rate=1e-4,
    train_batch_size=2,
    val_batch_size=2,
    sf_dice_tolerance=1.0,
    vae_beta=1.0,
)
hparams.save(config_dir / "hparams.yaml")

vae_config = uda_vae((64, 256, 256), 1, n_blocks=7, dim=3)
vae_config.save(config_dir / "vae.yaml")
