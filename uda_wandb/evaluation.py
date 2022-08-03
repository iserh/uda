"""Evaluation functions for wandb."""
from tempfile import TemporaryDirectory
from uda.trainer import VaeEvaluator
from uda.utils import reshape_to_volume
from uda.metrics import dice_score
import wandb
import numpy as np
import torch
from pathlib import Path

import wandb
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from uda import HParams, reshape_to_volume
from uda.models import VAEConfig, VAE, UNetConfig, UNet
from uda.datasets import CC359
import ignite.distributed as idist

vendors = ["GE_15", "GE_3", "SIEMENS_15", "SIEMENS_3", "PHILIPS_15", "PHILIPS_3"]


def evaluate_vae(run_id: str, project: str, team: str = "iserh", table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Evaluating run {run_id}\n")
    run = wandb.init(project=project, id=run_id, resume=True)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wandb.restore("config/cc359.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/hparams.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/vae.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("best_model.pt", f"{team}/{project}/{run_id}", root=tmpdir)

        hparams: HParams = HParams.from_file(tmpdir / "config/hparams.yaml")
        vae_config: VAEConfig = VAEConfig.from_file(tmpdir / "config/vae.yaml")
        dataset = CC359.from_preconfigured(tmpdir / "config/cc359.yaml")

        model = VAE.from_pretrained(tmpdir / "best_model.pt", vae_config)
        model.eval().to(idist.device())

    dataset.prepare_data()
    dataset.setup()
    data_loader = dataset.val_dataloader(hparams.val_batch_size)

    with torch.no_grad():
        preds, g_truths = [
            *zip(*[(model(x.to(idist.device()))[0].sigmoid().round().cpu(), x) for x, _ in tqdm(data_loader, desc="Predicting")])
        ]

    preds = torch.cat(preds).numpy()
    g_truths = torch.cat(g_truths).numpy()
    data = dataset.val_split.tensors[0].numpy()

    preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
    g_truths = reshape_to_volume(g_truths, dataset.imsize, dataset.patch_size)
    data = reshape_to_volume(data, dataset.imsize, dataset.patch_size)

    class_labels = {1: "Skull"}
    slice_index = dataset.imsize[0] // 2

    table = wandb.Table(columns=["ID", "Dim", "Criterion", "Beta", "Model Size", "Dice", "Surface Dice", "Image"])
    all_dice, all_sdice = [], []

    for i, (x_rec, x_true, img, spacing_mm) in tqdm(
        enumerate(zip(preds, g_truths, data, dataset.spacings_mm)),
        total=len(preds),
        desc="Computing Scores",
    ):
        all_dice.append(dice_score(x_rec, x_true))
        all_sdice.append(compute_surface_dice_at_tolerance(
            compute_surface_distances(x_true.astype(bool), x_rec.astype(bool), spacing_mm),
            tolerance_mm=hparams.sf_dice_tolerance,
        ))

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


def evaluate_unet(run_id: str, project: str, team: str = "iserh", table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Evaluating run {run_id}\n")
    run = wandb.init(project=project, id=run_id, resume=True)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wandb.restore("config/cc359.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/hparams.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/unet.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("best_model.pt", f"{team}/{project}/{run_id}", root=tmpdir)

        hparams: HParams = HParams.from_file(tmpdir / "config/hparams.yaml")
        unet_config: UNetConfig = UNetConfig.from_file(tmpdir / "config/unet.yaml")
        dataset = CC359.from_preconfigured(tmpdir / "config/cc359.yaml")

        model = UNet.from_pretrained(tmpdir / "best_model.pt", unet_config)
        model.eval().to(idist.device())

    dataset.prepare_data()
    dataset.setup()
    data_loader = dataset.val_dataloader(hparams.val_batch_size)

    with torch.no_grad():
        preds, targets = [
            *zip(*[(model(x.to(idist.device())).sigmoid().round().cpu(), y) for x, y in tqdm(data_loader, desc="Predicting")])
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
        all_sdice.append(compute_surface_dice_at_tolerance(
            compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
            tolerance_mm=hparams.sf_dice_tolerance,
        ))

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

    wandb.finish()


def cross_evaluate_unet(run_id: str, project: str, team: str = "iserh", table_plot: bool = True, table_size: int = 5) -> None:
    print(f"Cross evaluating run {run_id}\n")
    run = wandb.init(project=project, id=run_id, resume=True)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wandb.restore("config/cc359.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/hparams.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("config/unet.yaml", f"{team}/{project}/{run_id}", root=tmpdir)
        wandb.restore("best_model.pt", f"{team}/{project}/{run_id}", root=tmpdir)

        hparams: HParams = HParams.from_file(tmpdir / "config/hparams.yaml")
        unet_config: UNetConfig = UNetConfig.from_file(tmpdir / "config/unet.yaml")
        dataset = CC359.from_preconfigured(tmpdir / "config/cc359.yaml")

        model = UNet.from_pretrained(tmpdir / "best_model.pt", unet_config)
        model.eval().to(idist.device())

    dataset.prepare_data()  # downloads the dataset

    for vendor in vendors:
        print(f"\nEVALUATING VENDOR - {vendor} -\n")
        dataset.config.vendor = vendor

        dataset.setup()
        data_loader = dataset.val_dataloader(hparams.val_batch_size)

        with torch.no_grad():
            preds, targets = [
                *zip(*[(model(x.to(idist.device())).sigmoid().round().cpu(), y) for x, y in tqdm(data_loader, desc="Predicting")])
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
            all_sdice.append(compute_surface_dice_at_tolerance(
                compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                tolerance_mm=hparams.sf_dice_tolerance,
            ))

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
