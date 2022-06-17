from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, EpochOutputStore, ModelCheckpoint
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss, Metric
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader

from uda import VAE, HParams, VAEConfig
from uda.datasets import CC359, CC359Config
from uda.losses import kl_loss, loss_fn, optimizer_cls
from uda.metrics import dice_score
from uda.utils import (
    binary_one_hot_output_transform,
    distr_from_vae_output,
    pipe,
    pred_from_vae_output,
    reshape_to_volume,
    to_cpu,
    sigmoid_round,
)


def vae_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rec_loss_fn: Union[Callable, torch.nn.Module],
    beta: float = 1,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Engine:
    def update(engine: Engine, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad()
        model.train()

        x = convert_tensor(batch, device, non_blocking)
        x_rec, mean, v_log = model(x)

        rec_l = rec_loss_fn(x_rec, x)
        kl_l = kl_loss(mean, v_log)
        loss = beta * kl_l + rec_l

        loss.backward()
        optimizer.step()

        return rec_l, kl_l

    trainer = Engine(update)
    return trainer


def vae_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Callable:
    metrics = metrics or {}

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        model.eval()
        with torch.no_grad():
            x = convert_tensor(batch, device, non_blocking)
            output = model(x)

        return output, x

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def run(config_dir: Path, data_dir: Path, project: str, tags: List[str] = [], group: Optional[str] = None) -> None:
    # load configuration files
    dataset_config: CC359Config = CC359Config.from_file(config_dir / "cc359.yaml")
    vae_config: VAEConfig = VAEConfig.from_file(config_dir / "vae.yaml")
    hparams: HParams = HParams.from_file(config_dir / "hparams.yaml")

    # run_name = f"VAE-{vae_config.dim}D-{dataset_config.vendor}"
    run_name = f"VAE-{vae_config.dim}D-{hparams.criterion}"
    run = wandb.init(
        project=project,
        tags=tags,
        group=group,
        name=run_name,
        config={
            "hparams": hparams.__dict__,
            "dataset": dataset_config.__dict__,
            "model": vae_config.__dict__,
        },
    )

    run.save(str(config_dir / "*"), base_path=str(config_dir.parent), policy="now")
    run.save(str(Path(__file__)), policy="now")

    cc359 = run.use_artifact("tiser/UDA-Datasets/CC359-Skull-stripping:latest")
    data_dir = cc359.download(root=data_dir)

    train_dataset = CC359(data_dir, dataset_config, train=True)
    val_dataset = CC359(data_dir, dataset_config, train=False)
    train_loader = DataLoader(train_dataset.targets, batch_size=hparams.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset.targets, batch_size=hparams.val_batch_size, shuffle=False)

    # -------------------- model, loss, metrics --------------------

    model = VAE(vae_config)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.summary.update({"n_parameters": n_params})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)  # Move model before creating optimizer
    optimizer = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    rec_criterion = loss_fn(hparams.criterion)(**hparams.loss_kwargs)

    # metrics
    rec_loss_metric = Loss(rec_criterion, output_transform=pred_from_vae_output)
    kl_loss_metric = Loss(kl_loss, output_transform=distr_from_vae_output)
    cm = ConfusionMatrix(num_classes=2, output_transform=pipe(pred_from_vae_output, binary_one_hot_output_transform))
    metrics = {"dice": DiceCoefficient(cm, ignore_index=0), "rec_loss": rec_loss_metric, "kl_loss": kl_loss_metric}

    # -------------------- trainer & evaluators --------------------

    trainer = vae_trainer(model, optimizer, rec_criterion, beta=hparams.vae_beta, device=device)
    trainer.logger = setup_logger("Trainer")

    evaluator = vae_evaluator(model, metrics=metrics, device=device)
    evaluator.logger = setup_logger("Evaluator")
    eos = EpochOutputStore(output_transform=pipe(pred_from_vae_output, sigmoid_round, to_cpu))
    eos.attach(evaluator, "output")

    # evaluation callback
    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine: Engine) -> None:
        evaluator.run(val_loader)

        preds, targets = [*zip(*evaluator.state.output)]

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        preds = reshape_to_volume(preds, val_dataset.imsize, val_dataset.patch_size)
        targets = reshape_to_volume(targets, val_dataset.imsize, val_dataset.patch_size)
        data = reshape_to_volume(val_dataset.data, val_dataset.imsize, val_dataset.patch_size)

        class_labels = {1: "Skull"}
        slice_index = val_dataset.imsize[0] // 2

        table = wandb.Table(columns=["ID", "Name", "Dice", "Surface Dice", "Image"])

        for i in range(5):
            dice = dice_score(preds[i], targets[i])
            # the raw background image as a numpy array
            img = (data[i][slice_index] * 255).astype(np.uint8)
            x_rec = preds[i][slice_index].astype(np.uint8)
            x_true = targets[i][slice_index].astype(np.uint8)

            wandb_img = wandb.Image(
                img,
                masks={
                    "prediction": {"mask_data": x_rec, "class_labels": class_labels},
                    "ground truth": {"mask_data": x_true, "class_labels": class_labels},
                },
            )

            if i < 5:
                table.add_data(i, run.name, dice, "-", wandb_img)

        try:
            old_artifact = run.use_artifact(f"run-{run.id}-validation_results:latest")
            old_artifact.delete(delete_aliases=True)
        except Exception:
            pass

        run.log({"validation_results": table})

    # -------------------- Handlers --------------------
    # -----------------------------------------------------

    # ---------- tqdm ----------
    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(evaluator)

    # -------------------- WandB --------------------

    wandb_logger = WandBLogger(id=run.id)
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda out: {"batch_rec_loss": out[0], "batch_kl_loss": out[1]},
    )

    for tag, evaluator, metr in [("validation", evaluator, metrics)]:
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

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

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
    from argparse import ArgumentParser
    from tempfile import TemporaryDirectory
    import shutil

    parser = ArgumentParser()
    parser.add_argument("-t", "--tags", nargs="+", default=[])
    args = parser.parse_args()

    # directories
    data_dir = Path("/tmp/data/CC359")
    config_dir = Path("config")
    project = "UDA-CC359-VAE"

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "config").mkdir()

        shutil.copy(config_dir / "cc359.yaml", tmpdir / "config")
        shutil.copy(config_dir / "hparams.yaml", tmpdir / "config")
        shutil.copy(config_dir / "vae.yaml", tmpdir / "config")

        run_id = run(tmpdir / "config", data_dir, project=project, tags=args.tags, group=None)

    from evaluate_vae import evaluate_vae

    evaluate_vae(run_id, project=project, save_predictions=True)

    # from delete_model_binaries import delete_model_binaries

    # delete_model_binaries(run_id, project=project)
