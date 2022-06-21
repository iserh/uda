from pathlib import Path
from typing import List, Optional

import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss
from ignite.utils import setup_logger
from torch.utils.data import DataLoader

from uda import HParams, UNet, UNetConfig
from uda.datasets import CC359, CC359Config
from uda.losses import loss_fn, optimizer_cls
from uda.utils import binary_one_hot_output_transform


def run(config_dir: Path, data_dir: Path, project: str, tags: List[str] = [], group: Optional[str] = None) -> None:
    # load configuration files
    dataset_config: CC359Config = CC359Config.from_file(config_dir / "cc359.yaml")
    unet_config: UNetConfig = UNetConfig.from_file(config_dir / "unet.yaml")
    hparams: HParams = HParams.from_file(config_dir / "hparams.yaml")

    run_name = f"UNet-{unet_config.dim}D-{dataset_config.vendor}"
    run = wandb.init(
        project=project,
        tags=tags,
        group=group,
        name=run_name,
        config={
            "hparams": hparams.__dict__,
            "dataset": dataset_config.__dict__,
            "model": unet_config.__dict__,
        },
    )

    run.save(str(config_dir / "*"), base_path=str(config_dir.parent), policy="now")
    run.save(str(Path(__file__)), policy="now")

    cc359 = run.use_artifact("tiser/UDA-Datasets/CC359-Skull-stripping:latest")
    data_dir = cc359.download(root=data_dir)

    train_dataset = CC359(data_dir, dataset_config, train=True)
    val_dataset = CC359(data_dir, dataset_config, train=False)

    train_loader = DataLoader(train_dataset, batch_size=hparams.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams.val_batch_size, shuffle=False)

    # -------------------- model, loss, metrics --------------------

    model = UNet(unet_config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.summary.update({"n_parameters": n_params})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)  # Move model before creating optimizer
    optimizer = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    criterion = loss_fn(hparams.criterion)(**hparams.loss_kwargs)

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

    # evaluation callback
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine: Engine) -> None:
        if train_dataset.fold is not None:
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
        filename_pattern="best_model.pt",
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    # -------------------- early stopping --------------------

    if hparams.early_stopping:
        early_stopping = EarlyStopping(
            patience=hparams.early_stopping_patience,
            score_function=score_function,
            trainer=trainer,
        )

        evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    # -------------------- training --------------------

    # kick everything off
    trainer.run(train_loader, max_epochs=hparams.epochs)

    wandb_logger.close()

    return run.id


if __name__ == "__main__":
    import shutil
    from argparse import ArgumentParser
    from tempfile import TemporaryDirectory

    parser = ArgumentParser()
    parser.add_argument("-t", "--tags", nargs="+", default=[])
    args = parser.parse_args()

    # directories
    data_dir = Path("/tmp/data/CC359")
    config_dir = Path("config")
    project = "UDA-CC359"

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "config").mkdir()

        shutil.copy(config_dir / "cc359.yaml", tmpdir / "config")
        shutil.copy(config_dir / "hparams.yaml", tmpdir / "config")
        shutil.copy(config_dir / "unet.yaml", tmpdir / "config")

        run_id = run(tmpdir / "config", data_dir, project=project, tags=args.tags, group="U-Net")

    from evaluate_run import evaluate_run

    evaluate_run(run_id, project=project, save_predictions=True)

    from cross_evaluate_run import cross_evaluate_run

    cross_evaluate_run(run_id, project=project, save_predictions=True)

    # from delete_model_binaries import delete_model_binaries

    # delete_model_binaries(run_id, project=project)
