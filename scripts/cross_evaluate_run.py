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

    wandb.restore("config/cc359.yml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/hparams.yml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("config/unet.yml", f"tiser/UDA/{run_id}", root=files_dir, replace=True)
    wandb.restore("best_model", f"tiser/UDA/{run_id}", root=files_dir, replace=True)

    unet_conf = UNetConfig.from_file(files_dir / "config/unet.yml")
    dataset_conf = CC359Config.from_file(files_dir / "config/cc359.yml")
    hparams = HParams.from_file(files_dir / "config/hparams.yml")

    dataset_conf.fold = None

    for vendor in vendors:
        print()
        print(f"EVALUATING VENDOR - {vendor} -")
        print()

        dataset_conf.vendor = vendor

        dataset = CC359(data_dir, dataset_conf)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams.val_batch_size, shuffle=False)

        model = UNet.from_pretrained(files_dir / "best_model", unet_conf)

        model.eval().to(device)

        with torch.no_grad():
            preds, targets = [*zip(*[(model(x.to(device)).cpu(), y_true) for x, y_true in tqdm(data_loader)])]

        preds = reshape_to_volume(torch.cat(preds).round(), dataset.PADDING_SHAPE, dataset.patch_dims)
        targets = reshape_to_volume(torch.cat(targets), dataset.PADDING_SHAPE, dataset.patch_dims)
        inputs = reshape_to_volume(dataset.data, dataset.PADDING_SHAPE, dataset.patch_dims)

        model.cpu()

        preds.shape

        class_labels = {1: "Skull"}
        slice_index = dataset.PADDING_SHAPE[0] // 2

        table = wandb.Table(columns=["ID", "Dice", "Surface Dice", "Image"])

        # iterate over subjects
        subject_data = zip(preds, targets, inputs, dataset.spacing_mm)
        for i, (y_pred, y_true, data, spacing_mm) in tqdm(
            enumerate(subject_data), total=len(preds), desc="Final Evaluation Metric Computing", leave=False
        ):
            dice = dice_score(y_pred, y_true)
            surface_dice = compute_surface_dice_at_tolerance(
                compute_surface_distances(y_true.bool().numpy(), y_pred.bool().numpy(), spacing_mm),
                tolerance_mm=hparams.sdice_tolerance,
            )

            # the raw background image as a numpy array
            data = (data[slice_index] * 255).numpy().astype(np.uint8)
            y_pred = y_pred[slice_index].numpy().astype(np.uint8)
            y_true = y_true[slice_index].numpy().astype(np.uint8)

            # rotate images & masks
            data = np.rot90(data, k=2)
            y_pred = np.rot90(y_pred, k=2)
            y_true = np.rot90(y_true, k=2)

            wandb_img = wandb.Image(
                data,
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
