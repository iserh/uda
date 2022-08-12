"""Evaluation functions for wandb."""
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine
from ignite.handlers import EpochOutputStore
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from uda import HParams, get_preds_output_transform, pipe, reshape_to_volume, to_cpu_output_transform
from uda.datasets import UDADataset
from uda.metrics import dice_score
from uda.models import VAE, UNet
from uda_wandb.config import RunConfig

from .download import download_model

vendors = ["GE_15", "GE_3", "SIEMENS_15", "SIEMENS_3", "PHILIPS_15", "PHILIPS_3"]


def evaluate(
    Engine: type[Engine],
    Model: Union[type[VAE], type[UNet]],
    dataset: UDADataset,
    hparams: HParams,
    run_cfg: RunConfig,
    splits: list[str] = ["validation"],
    n_predictions: int = 6,
) -> None:
    print()
    print(f"Evaluating run {run_cfg.run_id}\n")

    with wandb.init(project=run_cfg.project, id=run_cfg.run_id, resume=True) as run:
        with TemporaryDirectory() as tmpdir:
            model_path = download_model(run_cfg, tmpdir)
            model = Model.from_pretrained(model_path)

        for split in splits:
            try:
                dataloader, spacings = dataset.get_split(split, batch_size=hparams.val_batch_size)
            except NotImplementedError:
                print(f"Skipping split '{split}', since it is not available in dataset {dataset.__class__.__name__}.")
                return
            else:
                print(f"Setup of split '{split}' successful.")

            evaluator = Engine(model)
            ProgressBar(desc=f"Eval({split})", persist=True).attach(evaluator)
            eos = EpochOutputStore(
                output_transform=pipe(lambda o: o[:3], get_preds_output_transform, to_cpu_output_transform)
            )
            eos.attach(evaluator, "output")

            evaluator.run(dataloader)

            preds, targets, _ = prediction_image_plot(evaluator, model.config.dim, dataset, split, n_predictions)

            all_dice, all_sdice = [], []
            for y_pred, y_true, spacing_mm in tqdm(
                zip(preds, targets, spacings), total=len(preds), desc="Computing Scores"
            ):
                all_dice.append(dice_score(y_pred, y_true))
                all_sdice.append(
                    compute_surface_dice_at_tolerance(
                        compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                        tolerance_mm=hparams.sf_dice_tolerance,
                    )
                )

            run.summary[f"{split}/dice"] = np.stack(all_dice).mean()
            run.summary[f"{split}/surface_dice"] = np.array(all_sdice).mean()


def cross_evaluate_unet(
    Engine: type[Engine],
    Model: Union[type[VAE], type[UNet]],
    dataset: UDADataset,
    hparams: HParams,
    run_cfg: RunConfig,
    n_predictions: int = 6,
) -> None:
    print()
    print(f"Cross evaluating run {run_cfg.run_id}\n")

    with wandb.init(project=run_cfg.project, id=run_cfg.run_id, resume=True) as run:
        with TemporaryDirectory() as tmpdir:
            model_path = download_model(run_cfg, tmpdir)
            model = Model.from_pretrained(model_path)

        for vendor in vendors:
            dataset.vendor = vendor
            dataset.setup()
            dataloader, spacings = dataset.get_split("validation", hparams.val_batch_size)

            evaluator = Engine(model)
            ProgressBar(desc=f"Eval({vendor})", persist=True).attach(evaluator)
            eos = EpochOutputStore(
                output_transform=pipe(lambda o: o[:3], get_preds_output_transform, to_cpu_output_transform)
            )
            eos.attach(evaluator, "output")

            evaluator.run(dataloader)

            preds, targets, _ = prediction_image_plot(evaluator, model.config.dim, dataset, vendor, n_predictions)

            all_dice, all_sdice = [], []
            for y_pred, y_true, spacing_mm in tqdm(
                zip(preds, targets, spacings), total=len(preds), desc="Computing Scores"
            ):
                all_dice.append(dice_score(y_pred, y_true))
                all_sdice.append(
                    compute_surface_dice_at_tolerance(
                        compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                        tolerance_mm=hparams.sf_dice_tolerance,
                    )
                )

            run.summary[f"{vendor}/dice"] = np.stack(all_dice).mean()
            run.summary[f"{vendor}/surface_dice"] = np.array(all_sdice).mean()


def prediction_image_plot(
    evaluator: Engine,
    dim: int,
    dataset: UDADataset,
    name: str,
    n_predictions: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print()
    print("Plotting prediction images")
    preds, targets, data = [*zip(*evaluator.state.output)]

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    data = torch.cat(data).numpy()

    preds = reshape_to_volume(preds, dim, dataset.imsize, dataset.patch_size)
    targets = reshape_to_volume(targets, dim, dataset.imsize, dataset.patch_size)
    data = reshape_to_volume(data, dim, dataset.imsize, dataset.patch_size)

    slice_index = dataset.imsize[0] // 2

    for i in range(min(n_predictions, preds.shape[0])):
        # the raw background image as a numpy array
        img = (data[i][slice_index] * 255).astype(np.uint8)
        y_pred = preds[i][slice_index].astype(np.uint8)
        y_true = targets[i][slice_index].astype(np.uint8)

        wandb_img = wandb.Image(
            img,
            masks={
                "prediction": {"mask_data": y_pred, "class_labels": dataset.class_labels},
                "ground truth": {"mask_data": y_true, "class_labels": dataset.class_labels},
            },
        )
        wandb.log({f"{name}/predictions/{i}": wandb_img}, commit=False)

    return preds, targets, data
