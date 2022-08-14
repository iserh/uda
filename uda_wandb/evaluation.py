"""Evaluation functions for wandb."""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
import torch
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine
from ignite.handlers import EpochOutputStore
from ignite.utils import to_onehot
from pypatchify.pt import pt

from uda import HParams
from uda.datasets import UDADataset
from uda.metrics import dice_score, surface_dice
from uda.models import VAE, UNet
from uda.trainer import get_preds_output_transform, pipe, to_cpu_output_transform
from uda_wandb.config import RunConfig

from .download import download_model

vendors = ["GE_15", "GE_3", "SIEMENS_15", "SIEMENS_3", "PHILIPS_15", "PHILIPS_3"]


def evaluate(
    Engine: type[Engine],
    Model: Union[type[VAE], type[UNet]],
    dataset: UDADataset,
    hparams: HParams,
    splits: list[str] = ["validation"],
    n_predictions: int = 6,
) -> None:
    print()
    print(f"Evaluating run\n")

    run_dir = Path(wandb.run.dir)
    model = Model.from_pretrained(run_dir / "best_model.pt", run_dir / "config" / "model.yaml")

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

        preds, targets, _ = prediction_image_plot(evaluator, dataset, split, n_predictions)
        preds = to_onehot(preds, num_classes=len(dataset.class_labels))

        dice = dice_score(preds, targets, ignore_index=0)
        sf_dice = surface_dice(preds, targets, spacings, hparams.sf_dice_tolerance, ignore_index=0)

        for i, (dsc, sf_dsc) in enumerate(zip(dice, sf_dice)):
            wandb.run.summary[f"{split}/dice/{i}"] = dsc
            wandb.run.summary[f"{split}/surface_dice/{i}"] = sf_dsc


def cross_evaluate_unet(
    Engine: type[Engine],
    Model: Union[type[VAE], type[UNet]],
    dataset: UDADataset,
    hparams: HParams,
    n_predictions: int = 6,
) -> None:
    print()
    print(f"Cross evaluating run\n")

    run_dir = Path(wandb.run.dir)
    model = Model.from_pretrained(run_dir / "best_model.pt", run_dir / "config" / "model.yaml")

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

        preds, targets, _ = prediction_image_plot(evaluator, dataset, vendor, n_predictions)
        preds = to_onehot(preds, num_classes=len(dataset.class_labels))

        dice = dice_score(preds, targets, ignore_index=0)
        sf_dice = surface_dice(preds, targets, spacings, hparams.sf_dice_tolerance, ignore_index=0)

        for i, (dsc, sf_dsc) in enumerate(zip(dice, sf_dice)):
            wandb.run.summary[f"{vendor}/dice/{i}"] = dsc
            wandb.run.summary[f"{vendor}/surface_dice/{i}"] = sf_dsc


def prediction_image_plot(
    evaluator: Engine,
    dataset: UDADataset,
    name: str,
    n_predictions: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print()
    print("Plotting prediction images")
    preds, targets, data = [*zip(*evaluator.state.output)]

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    data = torch.cat(data).squeeze(1)  # we don't need channel dim

    # preds/targets shape: (N, ...) - indices of predicted/gt class
    if dataset.flatten:
        # uncollapse Z dim from batch dim
        preds = preds.reshape(-1, dataset.imsize[0], *preds.shape[-2:])
        targets = targets.reshape(-1, dataset.imsize[0], *targets.shape[-2:])
        data = data.reshape(-1, dataset.imsize[0], *data.shape[-2:])

    preds = pt.unpatchify_from_batches(preds, dataset.imsize)
    targets = pt.unpatchify_from_batches(targets, dataset.imsize)
    data = pt.unpatchify_from_batches(data, dataset.imsize)

    slice_index = dataset.imsize[0] // 2
    class_labels = {k: v for k, v in dataset.class_labels.items() if v != "Background"}

    for i in range(min(n_predictions, preds.shape[0])):
        wandb_img = wandb.Image(
            data[i][slice_index].numpy(),
            masks={
                "prediction": {"mask_data": preds[i][slice_index].numpy(), "class_labels": class_labels},
                "ground truth": {"mask_data": targets[i][slice_index].numpy(), "class_labels": class_labels},
            },
        )
        wandb.log({f"{name}/predictions/{i}": wandb_img}, commit=False)

    return preds, targets, data
