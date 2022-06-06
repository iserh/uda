from pathlib import Path

import numpy as np
import torch
import wandb
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm import tqdm

from uda import CC359, CC359Config, HParams, UNet, UNetConfig
from uda.metrics import dice_score
from uda.utils import reshape_to_volume

vendors = ["PHILIPS_3", "PHILIPS_15", "SIEMENS_3", "SIEMENS_15", "GE_3", "GE_15"]


def cross_evaluate_run(
    run_id: str, data_dir: Path = Path("/tmp/data/CC359"), files_dir: Path = Path("/tmp/files")
) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.restore("config/cc359.yaml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/hparams.yaml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/unet.yaml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("best_model", f"tiser/UDA/{run_id}", root=files_dir, replace=True)

    unet_config: UNetConfig = UNetConfig.from_file(files_dir / "config/unet.yaml")
    dataset_config: CC359Config = CC359Config.from_file(files_dir / "config/cc359.yaml")
    hparams: HParams = HParams.from_file(files_dir / "config/hparams.yaml")

    dataset_config.fold = None

    for vendor in vendors:
        print()
        print(f"EVALUATING VENDOR - {vendor} -")
        print()

        dataset_config.vendor = vendor

        dataset = CC359(data_dir, dataset_config)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams.val_batch_size, shuffle=False)

        model = UNet.from_pretrained(files_dir / "best_model", unet_config)

        model.eval().to(device)

        with torch.no_grad():
            preds, targets = [*zip(*[(model(x.to(device)).cpu(), y_true) for x, y_true in tqdm(data_loader)])]

        preds = torch.cat(preds).round().numpy()
        targets = torch.cat(targets).numpy()

        preds = reshape_to_volume(preds, dataset.imsize, dataset.patch_size)
        targets = reshape_to_volume(targets, dataset.imsize, dataset.patch_size)
        data = reshape_to_volume(dataset.data, dataset.imsize, dataset.patch_size)

        model.cpu()

        class_labels = {1: "Skull"}
        slice_index = dataset.imsize[0] // 2

        table = wandb.Table(columns=["ID", "Dice", "Surface Dice", "Image"])

        for i, (y_pred, y_true, x, spacing_mm) in tqdm(
            enumerate(zip(preds, targets, data, dataset.spacings_mm)),
            total=len(preds),
            desc="Final Evaluation Metric Computing",
            leave=False,
        ):
            dice = dice_score(y_pred, y_true)
            surface_dice = compute_surface_dice_at_tolerance(
                compute_surface_distances(y_true.astype(bool), y_pred.astype(bool), spacing_mm),
                tolerance_mm=hparams.sf_dice_tolerance,
            )

            # the raw background image as a numpy array
            x = (x[slice_index] * 255).astype(np.uint8)
            y_pred = y_pred[slice_index].astype(np.uint8)
            y_true = y_true[slice_index].astype(np.uint8)

            wandb_img = wandb.Image(
                x,
                masks={
                    "prediction": {"mask_data": y_pred, "class_labels": class_labels},
                    "ground truth": {"mask_data": y_true, "class_labels": class_labels},
                },
            )

            table.add_data(i, dice, surface_dice, wandb_img)

        dice_mean = np.array(table.get_column("Dice")).mean()
        surface_dice_mean = np.array(table.get_column("Surface Dice")).mean()

        run = wandb.init(project="UDA", id=run_id, resume=True)

        run.log({f"{dataset.vendor}_results": table})
        run.summary[f"{dataset.vendor}_dice"] = dice_mean
        run.summary[f"{dataset.vendor}_surface_dice"] = surface_dice_mean

        wandb.finish()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    cross_evaluate_run(args.run_id)