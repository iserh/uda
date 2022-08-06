"""Evaluation functions for wandb."""
from pathlib import Path
from tempfile import TemporaryDirectory

import ignite.distributed as idist
import numpy as np
import torch
import wandb
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from uda import HParams, reshape_to_volume
from uda.datasets import CC359
from uda.metrics import dice_score
from uda.models import VAE, UNet, UNetConfig, VAEConfig
from uda.trainer import VaeEvaluator
from uda.utils import reshape_to_volume
from uda_wandb.config import RunConfig

from .download import download_config, download_dataset, download_model

vendors = ["GE_15", "GE_3", "SIEMENS_15", "SIEMENS_3", "PHILIPS_15", "PHILIPS_3"]


def evaluate_vae(run_cfg: RunConfig, table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Evaluating run {run_cfg.run_id}\n")
    run = wandb.init(project=run_cfg.project, id=run_cfg.run_id, resume=True)

    with TemporaryDirectory() as tmpdir:
        cfg_dir = download_config(run_cfg, tmpdir)
        hparams: HParams = HParams.from_file(cfg_dir / "hparams.yaml")
        dataset = CC359.from_preconfigured(cfg_dir / "dataset.yaml")
    with TemporaryDirectory() as tmpdir:
        model_path = download_model(run_cfg, tmpdir)
        model = VAE.from_pretrained(model_path)

    download_dataset(CC359)
    dataset.setup()
    data_loader = dataset.val_dataloader(hparams.val_batch_size)

    model.eval().to(idist.device())
    with torch.no_grad():
        preds, g_truths = [
            *zip(
                *[
                    (model(x.to(idist.device()))[0].sigmoid().round().cpu(), x)
                    for _, x in tqdm(data_loader, desc="Predicting")
                ]
            )
        ]

    preds = torch.cat(preds).numpy()
    g_truths = torch.cat(g_truths).numpy()
    data = dataset.val_split.tensors[0].numpy()

    preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
    g_truths = reshape_to_volume(g_truths, dataset.imsize, dataset.patch_size)
    data = reshape_to_volume(data, dataset.imsize, dataset.patch_size)

    class_labels = {1: "Foreground"}
    slice_index = dataset.imsize[0] // 2

    table = wandb.Table(columns=["ID", "Dim", "Criterion", "Beta", "Model Size", "Dice", "Surface Dice", "Image"])
    all_dice, all_sdice = [], []

    for i, (x_rec, x_true, img, spacing_mm) in tqdm(
        enumerate(zip(preds, g_truths, data, dataset.spacings_mm)),
        total=len(preds),
        desc="Computing Scores",
    ):
        all_dice.append(dice_score(x_rec, x_true))
        all_sdice.append(
            compute_surface_dice_at_tolerance(
                compute_surface_distances(x_true.astype(bool), x_rec.astype(bool), spacing_mm),
                tolerance_mm=hparams.sf_dice_tolerance,
            )
        )

        if table_plot and i < table_size:
            # the raw background image as a numpy array
            img = (img[slice_index] * 255).astype(np.uint8)
            x_rec = x_rec[slice_index].astype(np.uint8)
            x_true = x_true[slice_index].astype(np.uint8)

            wandb_img = wandb.Image(
                img,
                masks={
                    "prediction": {"mask_data": x_rec, "class_labels": class_labels},
                    "ground truth": {"mask_data": x_true, "class_labels": class_labels},
                },
            )

            table.add_data(
                i,
                str(run.config.model["dim"]),
                run.config.hparams["criterion"],
                str(run.config.hparams["vae_beta"]),
                run.summary["model_size"],
                all_dice[-1],
                all_sdice[-1],
                wandb_img,
            )

    dice_mean = np.stack(all_dice).mean()
    sdice_mean = np.array(all_sdice).mean()

    if table_plot:
        try:
            old_artifact = run.use_artifact(f"run-{run.id}-validation_results:latest")
            old_artifact.delete(delete_aliases=True)
        except Exception:
            pass

        run.log({"validation_results": table})

    run.summary["validation/dice"] = dice_mean
    run.summary["validation/surface_dice"] = sdice_mean

    wandb.finish()


def evaluate_unet(run_cfg: RunConfig, table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Evaluating run {run_cfg.run_id}\n")

    with wandb.init(project=run_cfg.project, id=run_cfg.run_id, resume=True) as run:

        with TemporaryDirectory() as tmpdir:
            cfg_dir = download_config(run_cfg, tmpdir)
            hparams: HParams = HParams.from_file(cfg_dir / "hparams.yaml")
            dataset = CC359.from_preconfigured(cfg_dir / "dataset.yaml")
        with TemporaryDirectory() as tmpdir:
            model_path = download_model(run_cfg, tmpdir)
            model = UNet.from_pretrained(model_path)

        download_dataset(CC359)
        dataset.setup()
        data_loader = dataset.val_dataloader(hparams.val_batch_size)

        model.eval().to(idist.device())
        with torch.no_grad():
            preds, targets = [
                *zip(
                    *[
                        (model(x.to(idist.device())).sigmoid().round().cpu(), y)
                        for x, y in tqdm(data_loader, desc="Predicting")
                    ]
                )
            ]

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        data = dataset.val_split.tensors[0].numpy()

        preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
        targets = reshape_to_volume(targets, dataset.imsize, dataset.patch_size)
        data = reshape_to_volume(data, dataset.imsize, dataset.patch_size)

        class_labels = {1: "Skull"}
        slice_index = dataset.imsize[0] // 2

        table = wandb.Table(columns=["ID", "Dim", "Criterion", "Dice", "Surface Dice", "Image"])
        all_dice, all_sdice = [], []

        for i, (y_pred, y_true, img, spacing_mm) in tqdm(
            enumerate(zip(preds, targets, data, dataset.spacings_mm)),
            total=len(preds),
            desc="Computing Scores",
        ):
            all_dice.append(dice_score(y_pred, y_true))
            all_sdice.append(
                compute_surface_dice_at_tolerance(
                    compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                    tolerance_mm=hparams.sf_dice_tolerance,
                )
            )

            if table_plot and i < table_size:
                # the raw background image as a numpy array
                img = (img[slice_index] * 255).astype(np.uint8)
                y_pred = y_pred[slice_index].astype(np.uint8)
                y_true = y_true[slice_index].astype(np.uint8)

                wandb_img = wandb.Image(
                    img,
                    masks={
                        "prediction": {"mask_data": y_pred, "class_labels": class_labels},
                        "ground truth": {"mask_data": y_true, "class_labels": class_labels},
                    },
                )

                table.add_data(
                    i,
                    str(run.config.model["dim"]),
                    run.config.hparams["criterion"],
                    all_dice[-1],
                    all_sdice[-1],
                    wandb_img,
                )

        dice_mean = np.stack(all_dice).mean()
        sdice_mean = np.array(all_sdice).mean()

        if table_plot:
            try:
                old_artifact = run.use_artifact(f"run-{run.id}-validation_results:latest")
                old_artifact.delete(delete_aliases=True)
            except Exception:
                pass

            run.log({"validation_results": table})

        run.summary["validation/dice"] = dice_mean
        run.summary["validation/surface_dice"] = sdice_mean


def cross_evaluate_unet(run_cfg: RunConfig, table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Cross evaluating run {run_cfg.run_id}\n")
    run = wandb.init(project=run_cfg.project, id=run_cfg.run_id, resume=True)

    with TemporaryDirectory() as tmpdir:
        cfg_dir = download_config(run_cfg, tmpdir)
        hparams: HParams = HParams.from_file(cfg_dir / "hparams.yaml")
        dataset = CC359.from_preconfigured(cfg_dir / "dataset.yaml")
    with TemporaryDirectory() as tmpdir:
        model_path = download_model(run_cfg, tmpdir)
        model = UNet.from_pretrained(model_path)

    download_dataset(CC359)

    model.eval().to(idist.device())
    for vendor in vendors:
        print(f"\nEVALUATING VENDOR - {vendor} -\n")
        dataset.config.vendor = vendor

        dataset.setup()
        data_loader = dataset.val_dataloader(hparams.val_batch_size)

        with torch.no_grad():
            preds, targets = [
                *zip(
                    *[
                        (model(x.to(idist.device())).sigmoid().round().cpu(), y)
                        for x, y in tqdm(data_loader, desc="Predicting")
                    ]
                )
            ]

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        data = dataset.val_split.tensors[0].numpy()

        preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
        targets = reshape_to_volume(targets, dataset.imsize, dataset.patch_size)
        data = reshape_to_volume(data, dataset.imsize, dataset.patch_size)

        class_labels = {1: "Skull"}
        slice_index = dataset.imsize[0] // 2

        table = wandb.Table(columns=["ID", "Dim", "Criterion", "Dice", "Surface Dice", "Image"])
        all_dice, all_sdice = [], []

        for i, (y_pred, y_true, img, spacing_mm) in tqdm(
            enumerate(zip(preds, targets, data, dataset.spacings_mm)),
            total=len(preds),
            desc="Computing Scores",
        ):
            all_dice.append(dice_score(y_pred, y_true))
            all_sdice.append(
                compute_surface_dice_at_tolerance(
                    compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                    tolerance_mm=hparams.sf_dice_tolerance,
                )
            )

            if table_plot and i < table_size:
                # the raw background image as a numpy array
                img = (img[slice_index] * 255).astype(np.uint8)
                y_pred = y_pred[slice_index].astype(np.uint8)
                y_true = y_true[slice_index].astype(np.uint8)

                wandb_img = wandb.Image(
                    img,
                    masks={
                        "prediction": {"mask_data": y_pred, "class_labels": class_labels},
                        "ground truth": {"mask_data": y_true, "class_labels": class_labels},
                    },
                )

                table.add_data(
                    i,
                    str(run.config.model["dim"]),
                    run.config.hparams["criterion"],
                    all_dice[-1],
                    all_sdice[-1],
                    wandb_img,
                )

        dice_mean = np.stack(all_dice).mean()
        sdice_mean = np.array(all_sdice).mean()

        if table_plot:
            try:
                old_artifact = run.use_artifact(f"run-{run.id}-{vendor}_results:latest")
                old_artifact.delete(delete_aliases=True)
            except Exception:
                pass

            run.log({f"{vendor}_results": table})

        run.summary[f"{vendor}_dice"] = dice_mean
        run.summary[f"{vendor}_surface_dice"] = sdice_mean

    wandb.finish()


def vae_table_plot(
    evaluator: VaeEvaluator,
    data: torch.Tensor,
    imsize: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    table_size: int = 5,
):
    print("Creating VAE reconstruction table")
    preds, targets = [*zip(*evaluator.state.output)]

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    data = data.numpy()

    preds = reshape_to_volume(preds, imsize, patch_size)
    targets = reshape_to_volume(targets, imsize, patch_size)
    data = reshape_to_volume(data, imsize, patch_size)

    class_labels = {1: "Foreground"}
    slice_index = imsize[0] // 2

    table = wandb.Table(
        columns=[
            "ID",
            "Dim",
            "Criterion",
            "Beta",
            "Model Size",
            "Dice",
            "Surface Dice",
            "Image",
        ]
    )

    for i in range(min(table_size, preds.shape[0])):
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

        table.add_data(
            i,
            str(wandb.config.model["dim"]),
            wandb.config.hparams["criterion"],
            str(wandb.config.hparams["vae_beta"]),
            wandb.summary["model_size"],
            dice,
            "-",
            wandb_img,
        )

    try:
        old_artifact = wandb.use_artifact(f"run-{wandb.run.id}-validation_results:latest")
        old_artifact.delete(delete_aliases=True)
    except Exception:
        pass

    wandb.log({"validation_results": table})


def segmentation_table_plot(
    evaluator: VaeEvaluator,
    imsize: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    table_size: int = 5,
):
    print("Creating Segmentation prediction table")
    preds, targets, data = [*zip(*evaluator.state.output)]

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    data = torch.cat(data).numpy()

    preds = reshape_to_volume(preds, imsize, patch_size)
    targets = reshape_to_volume(targets, imsize, patch_size)
    data = reshape_to_volume(data, imsize, patch_size)

    class_labels = {1: "Foreground"}
    slice_index = imsize[0] // 2

    table = wandb.Table(
        columns=[
            "ID",
            "Dim",
            "Criterion",
            "Dice",
            "Surface Dice",
            "Image",
        ]
    )

    for i in range(min(table_size, preds.shape[0])):
        dice = dice_score(preds[i], targets[i])
        # the raw background image as a numpy array
        img = (data[i][slice_index] * 255).astype(np.uint8)
        y_pred = preds[i][slice_index].astype(np.uint8)
        y_true = targets[i][slice_index].astype(np.uint8)

        wandb_img = wandb.Image(
            img,
            masks={
                "prediction": {"mask_data": y_pred, "class_labels": class_labels},
                "ground truth": {"mask_data": y_true, "class_labels": class_labels},
            },
        )

        table.add_data(
            i,
            str(wandb.config.model["dim"]),
            wandb.config.hparams["criterion"],
            dice,
            "-",
            wandb_img,
        )

    try:
        old_artifact = wandb.use_artifact(f"run-{wandb.run.id}-validation_results:latest")
        old_artifact.delete(delete_aliases=True)
    except Exception:
        pass

    wandb.log({"validation_results": table})