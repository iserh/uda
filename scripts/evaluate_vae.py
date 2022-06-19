from pathlib import Path

import numpy as np
import torch
import wandb
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from uda import CC359, CC359Config, HParams, VAEConfig
from uda.metrics import dice_score
from uda.models.modeling_vae import VAE
from uda.utils import reshape_to_volume


def evaluate_vae(
    run_id: str,
    project: str,
    data_dir: Path = Path("/tmp/data/CC359"),
    files_dir: Path = Path("/tmp/files"),
    save_predictions: bool = True,
    n_predictions: int = 5,
) -> None:
    run = wandb.init(project=project, id=run_id, resume=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.restore("config/cc359.yaml", f"tiser/{project}/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/hparams.yaml", f"tiser/{project}/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/vae.yaml", f"tiser/{project}/{run_id}", root=files_dir, replace=True)
    wandb.restore("best_model.pt", f"tiser/{project}/{run_id}", root=files_dir, replace=True)

    vae_config: VAEConfig = VAEConfig.from_file(files_dir / "config/vae.yaml")
    dataset_config: CC359Config = CC359Config.from_file(files_dir / "config/cc359.yaml")
    hparams: HParams = HParams.from_file(files_dir / "config/hparams.yaml")

    print(f"Evaluating run {run_id}\n")

    dataset = CC359(data_dir, dataset_config)
    data_loader = DataLoader(dataset.targets, batch_size=hparams.val_batch_size, shuffle=False)

    model = VAE.from_pretrained(files_dir / "best_model.pt", vae_config)
    model.eval().to(device)

    with torch.no_grad():
        preds, targets = [
            *zip(*[(model(x.to(device))[0].sigmoid().round().cpu(), x) for x in tqdm(data_loader, desc="Predicting")])
        ]

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
    targets = reshape_to_volume(targets, dataset.imsize, dataset.patch_size)
    data = reshape_to_volume(dataset.data, dataset.imsize, dataset.patch_size)

    class_labels = {1: "Skull"}
    slice_index = dataset.imsize[0] // 2

    table = wandb.Table(columns=["ID", "Name", "Dice", "Surface Dice", "Image"])

    for i, (x_rec, x_true, img, spacing_mm) in tqdm(
        enumerate(zip(preds, targets, data, dataset.spacings_mm)),
        total=len(preds),
        desc="Building Table",
        leave=False,
    ):
        dice = dice_score(x_rec, x_true)
        surface_dice = compute_surface_dice_at_tolerance(
            compute_surface_distances(x_true.astype(bool), x_rec.astype(bool), spacing_mm),
            tolerance_mm=hparams.sf_dice_tolerance,
        )

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

        if i < n_predictions:
            table.add_data(i, run.name, dice, surface_dice, wandb_img)

    dice_mean = np.array(table.get_column("Dice")).mean()
    surface_dice_mean = np.array(table.get_column("Surface Dice")).mean()

    if save_predictions:
        run.log({"validation_results": table})

    run.summary["validation/dice"] = dice_mean
    run.summary["validation/surface_dice"] = surface_dice_mean

    wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    evaluate_vae(args.run_id, project=args.project, save_predictions=True)
