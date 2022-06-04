from pathlib import Path
from typing import Tuple

import ignite
import numpy as np
import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EpochOutputStore, ModelCheckpoint
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss
from ignite.utils import setup_logger
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from uda import HParams, UNet, UNetConfig
from uda.datasets import CC359, CC359Config
from uda.metrics import dice_score
from uda.utils import reshape_to_volume


def binary_one_hot_output_transform(output: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    y_pred, y = output
    y_pred = y_pred.round().long()
    y_pred = ignite.utils.to_onehot(y_pred, 2)
    return y_pred, y.long()


def run(config_dir: Path, data_dir: Path) -> None:
    # load configuration files
    dataset_conf = CC359Config.from_file(config_dir / "cc359.yml")
    unet_conf = UNetConfig.from_file(config_dir / "unet.yml")
    hparams = HParams.from_file(config_dir / "hparams.yml")

    run = wandb.init(
        project="UDA",
        group="Dev",
        name=f"UNet{unet_conf.dim}D-Source={dataset_conf.vendor}",
        config={
            "hparams": hparams.__dict__,
            "dataset": dataset_conf.__dict__,
            "model": unet_conf.__dict__,
        },
    )

    run.save(str(config_dir / "*"), policy="now")
    # run.save(str(Path(__file__)), policy="now")
    run.log_code()

    cc359 = run.use_artifact("CC359-Skull-stripping:latest")
    data_dir = cc359.download(root=data_dir)

    train_dataset = CC359(data_dir, dataset_conf, train=True)
    val_dataset = CC359(data_dir, dataset_conf, train=False)

    train_loader = DataLoader(train_dataset, batch_size=hparams.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.val_batch_size, shuffle=False)

    # -------------------- model, loss, metrics --------------------

    model = UNet(unet_conf)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# parameters: {n_params:,}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)  # Move model before creating optimizer
    optimizer = hparams.get_optimizer()(model.parameters(), lr=hparams.learning_rate)
    criterion = hparams.get_criterion()

    # metrics
    cm = ConfusionMatrix(num_classes=2, output_transform=binary_one_hot_output_transform)
    metrics = {"dice": DiceCoefficient(cm, ignore_index=0), "loss": Loss(criterion)}

    # -------------------- trainer & evaluators --------------------

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")

    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    final_evaluator = create_supervised_evaluator(model, device=device)
    final_evaluator.logger = setup_logger("Final Evaluator")
    # gather outputs in final evaluation for visualization
    eos = EpochOutputStore()
    eos.attach(final_evaluator, "output")

    # evaluation callback
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine: Engine) -> None:
        train_evaluator.run(train_loader)
        validation_evaluator.run(val_loader)

    # final evaluation
    @trainer.on(Events.COMPLETED)
    def compute_final_metrics(engine: Engine) -> None:
        final_evaluator.run(val_loader)

        y_pred, y_true = [*zip(*final_evaluator.state.output)]

        y_pred = torch.cat(y_pred).round().cpu()
        y_true = torch.cat(y_true).cpu()

        y_pred = reshape_to_volume(y_pred, val_dataset.PADDING_SHAPE, val_dataset.patch_dims)
        y_true = reshape_to_volume(y_true, val_dataset.PADDING_SHAPE, val_dataset.patch_dims)
        data = reshape_to_volume(val_dataset.data, val_dataset.PADDING_SHAPE, val_dataset.patch_dims)

        class_labels = {1: "Skull"}
        slice_index = val_dataset.PADDING_SHAPE[0] // 2

        table = wandb.Table(columns=["ID", "Dice", "Surface Dice", "Image"])

        # iterate over subjects
        subject_data = zip(y_pred, y_true, data, val_dataset.spacing_mm)
        for i, (y_pred_subj, y_true_subj, data_subj, spacing_mm) in tqdm(
            enumerate(subject_data), total=len(y_pred), desc="Final Evaluation Metric Computing", leave=False
        ):
            dice = dice_score(y_pred_subj, y_true_subj)
            surface_dice = compute_surface_dice_at_tolerance(
                compute_surface_distances(y_true_subj.bool().numpy(), y_pred_subj.bool().numpy(), spacing_mm),
                tolerance_mm=hparams.sdice_tolerance,
            )

            # the raw background image as a numpy array
            data_subj = (data_subj[slice_index] * 255).numpy().astype(np.uint8)
            y_pred_subj = y_pred_subj[slice_index].numpy().astype(np.uint8)
            y_true_subj = y_true_subj[slice_index].numpy().astype(np.uint8)

            wandb_img = wandb.Image(
                data_subj,
                masks={
                    "prediction": {"mask_data": y_pred_subj, "class_labels": class_labels},
                    "ground truth": {"mask_data": y_true_subj, "class_labels": class_labels},
                },
            )

            table.add_data(i, dice, surface_dice, wandb_img)

        surface_dice_mean = np.array(table.get_column("Surface Dice")).mean()
        wandb.log({"Segmentation": table, "validation/surface_dice": surface_dice_mean})

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

    for tag, evaluator, metr in [("training", train_evaluator, metrics), ("validation", validation_evaluator, metrics)]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=list(metr.keys()),
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    # -------------------- model checkpoint --------------------

    # score function used for model checkpoint
    def score_function(engine: Engine) -> None:
        if len(engine.state.metrics["dice"]) == 1:
            return engine.state.metrics["dice"].item()
        else:
            return 0

    model_checkpoint = ModelCheckpoint(
        wandb_logger.run.dir,
        n_saved=1,
        score_function=score_function,
        filename_pattern="best_model",
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

    run(config_dir, data_dir)
