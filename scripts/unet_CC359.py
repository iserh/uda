from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch
from mlflow import log_artifact, log_metrics, log_params
from tqdm import tqdm

from uda import HParamsConfig, UNet, UNetConfig
from uda.datasets import CC359, CC359Config
from uda.metrics import dice_score

# directories
data_dir = Path("/tmp/data/CC359")
config_dir = Path("config")

# setup mlflow
mlflow.set_tracking_uri("http://localhost:5000")

experiment_name = "U-Net Training"
print(f"Running in Experiment: '{experiment_name}'")
if (experiment := mlflow.get_experiment_by_name(experiment_name)) is None:
    experiment_id = mlflow.create_experiment(name=experiment_name)
else:
    experiment_id = experiment.experiment_id

# configure the device used for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load configuration files
ds_config = CC359Config.from_file(config_dir / "cc359.json")
unet_config = UNetConfig.from_file(config_dir / "unet.json")
hparams = HParamsConfig.from_file(config_dir / "hparams.json")

# load dataset
train_dataset = CC359(data_dir, ds_config, train=True)
test_dataset = CC359(data_dir, ds_config, train=False)

print(train_dataset.data.shape)
print(train_dataset.label.shape)
print(train_dataset.voxel_dim.shape)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

model = UNet(unet_config)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# parameters: {n_params:,}")

model = model.to(device)
optim = hparams.get_optim()(model.parameters(), lr=hparams.learning_rate)
criterion = hparams.get_criterion()


# ------------------------------
# ----- Training
# ------------------------------
run_name = f"{model.__class__.__name__}_{train_dataset.__class__.__name__}"

with mlflow.start_run(experiment_id=experiment_id, run_name=run_name), TemporaryDirectory() as tempdir:
    tempdir = Path(tempdir)
    # log params
    log_params(
        {
            "n_params": n_params,
            "optim": optim.__class__.__name__,
            "criterion": criterion.__name__,
            **ds_config.__dict__,
            **hparams.__dict__,
        }
    )

    # log artifacts
    log_artifact(config_dir / "cc359.json")
    log_artifact(config_dir / "unet.json")
    log_artifact(config_dir / "hparams.json")

    i = 0
    for e in range(1, hparams.epochs + 1):
        for x, y_true in (pbar := tqdm(train_loader, desc=f"Training Epoch {e}/{hparams.epochs}")):
            i += 1

            x = x.to(device)
            y_true = y_true.to(device)

            optim.zero_grad()
            y_pred = model(x)
            train_loss = criterion(y_pred, y_true)

            train_loss.backward()
            optim.step()

            train_dsc = dice_score(y_pred.round(), y_true)
            log_metrics({"train_loss": train_loss, "train_dsc": train_dsc}, step=i)

            if i % hparams.test_interval == 0:
                # Evaluate on test set
                with torch.no_grad():
                    preds, targets = [
                        *zip(
                            *[
                                (model(x.to(device)).cpu(), y_true)
                                for x, y_true in tqdm(test_loader, desc="Testing", leave=False)
                            ]
                        )
                    ]
                preds = torch.cat(preds)
                targets = torch.cat(targets)

                test_loss = criterion(preds, targets)
                test_dsc = dice_score(preds.round(), targets)

                log_metrics({"test_loss": test_loss, "test_dsc": test_dsc}, step=i)

        pbar.set_postfix(
            {
                "loss": train_loss,
                "train_dsc": train_dsc,
                "test_dsc": test_dsc if "test_dsc" in locals() else 0,
            }
        )

    model.cpu()
    model.save(tempdir / "unet.pt")
    log_artifact(tempdir / "unet.pt")
