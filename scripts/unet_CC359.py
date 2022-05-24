from pathlib import Path
from typing import Tuple

import ignite
import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import ConfusionMatrix, DiceCoefficient, EpochMetric, Loss
from ignite.utils import setup_logger
from torch.utils.data import DataLoader
from wandb.wandb_run import Run

from uda import HParams, UNet, UNetConfig
from uda.datasets import CC359, CC359Config
from uda.metrics import dice_score


def flatten_output(output: Tuple[torch.Tensor, torch.Tensor]) -> None:
    y_pred, y = output
    return y_pred.round().long().flatten(), y.flatten()


def get_data_loaders(
    run: Run, dataset_conf: CC359Config, train_batch_size: int, val_batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    cc359 = run.use_artifact("CC359-Skull-stripping:latest")
    data_dir = cc359.download(root="/tmp/data/CC359")
    print(data_dir)

    train_loader = DataLoader(CC359(data_dir, dataset_conf, train=True), batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(CC359(data_dir, dataset_conf, train=False), batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def binary_one_hot_output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    y_pred, y = output
    y_pred = y_pred.round().long()
    y_pred = ignite.utils.to_onehot(y_pred, 2)
    return y_pred, y.long()


def run(hparams: HParams, dataset_conf: CC359Config, unet_conf: UNetConfig) -> None:
    run = wandb.init(
        project="UDA",
        name="CC359-UNet",
        config={
            "hparams": hparams.__dict__,
            "dataset_configuration": dataset_conf.__dict__,
            "unet_configuration": unet_conf.__dict__,
        },
    )

    train_loader, val_loader = get_data_loaders(run, dataset_conf, hparams.train_batch_size, hparams.val_batch_size)

    # -------------------- model, loss, metrics --------------------

    model = UNet(unet_conf)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# parameters: {n_params:,}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)  # Move model before creating optimizer
    optimizer = hparams.get_optim()(model.parameters(), lr=hparams.learning_rate)
    criterion = hparams.get_criterion()

    # metrics
    dice_metric = EpochMetric(compute_fn=dice_score, output_transform=flatten_output)
    cm = ConfusionMatrix(num_classes=2, output_transform=binary_one_hot_output_transform)
    metrics = {"dice": DiceCoefficient(cm, ignore_index=0), "loss": Loss(criterion), "dice2": dice_metric}

    # -------------------- trainer & evaluators --------------------

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")

    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    # evaluation callback
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine: Engine) -> None:
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

    # -------------------- Handlers --------------------
    # -----------------------------------------------------

    # ---------- tqdm ----------
    pbar = ProgressBar()
    pbar.attach(trainer)

    # -------------------- WandB --------------------

    wandb_logger = WandBLogger(id=run.id)
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", validation_evaluator)]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=list(metrics.keys()),
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    # -------------------- model checkpoint --------------------

    # score function used for model checkpoint
    def score_function(engine: Engine) -> None:
        return engine.state.metrics["dice"]

    model_checkpoint = ModelCheckpoint(
        wandb_logger.run.dir,
        n_saved=1,
        filename_prefix="best",
        score_function=score_function,
        score_name="validation_dice",
        global_step_transform=global_step_from_engine(trainer),
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    # -------------------- training --------------------

    # kick everything off
    trainer.run(train_loader, max_epochs=hparams.epochs)

    wandb_logger.close()


if __name__ == "__main__":
    # directories
    data_dir = Path("/tmp/data/CC359")
    config_dir = Path("config")

    # load configuration files
    dataset_conf = CC359Config.from_file(config_dir / "cc359.yml")
    unet_conf = UNetConfig.from_file(config_dir / "unet.yml")
    hparams = HParams.from_file(config_dir / "hparams.yml")

    run(hparams, dataset_conf, unet_conf)
