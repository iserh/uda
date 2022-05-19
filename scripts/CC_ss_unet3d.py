from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch
from mlflow import log_artifact, log_metrics, log_params
from tqdm import tqdm

from uda import UNet, UNetConfig
from uda.calgary_campinas_dataset import CalgaryCampinasDataset
from uda.losses import dice_loss
from uda.metrics import dice_score

data_dir = Path("/tmp/data/CC359")
mlflow.set_tracking_uri("http://localhost:5000")
if (experiment := mlflow.get_experiment_by_name("U-Net Training")) is None:
    experiment_id = mlflow.create_experiment(name="U-Net Training")
else:
    experiment_id = experiment.experiment_id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = CalgaryCampinasDataset(
    data_dir, vendor="GE_3", fold=1, train=True, patchify=(64, 256, 256), flatten_patches=True
)
test_dataset = CalgaryCampinasDataset(
    data_dir, vendor="GE_3", fold=1, train=False, patchify=(64, 256, 256), flatten_patches=True
)

print(train_dataset.data.shape)
print(train_dataset.label.shape)
print(train_dataset.voxel_dim.shape)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# This configuration is similar to the preprint, but it uses smaller u-net blocks
# to compensate for CPU limitations
config = UNetConfig(
    out_channels=1,
    dim=3,
    encoder_blocks=(
        (1, 8, 8),
        (8, 16, 16),
        (16, 32, 32),
        (32, 64, 64),
        (64, 128, 128),
        (128, 256, 256),
    ),
    decoder_blocks=(
        (256, 128, 128),
        (128, 64, 64),
        (64, 32, 32),
        (32, 16, 16),
        (16, 8, 8),
    ),
)

model = UNet(config)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# parameters: {n_params:,}")

model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = dice_loss

MAX_EPOCHS = 1
TEST_INTERVAL = 20

train_losses, test_losses = [], []
train_dscs, test_dscs = [], []

with mlflow.start_run(experiment_id=experiment_id, run_name="unet3d"), TemporaryDirectory() as tempdir:
    tempdir = Path(tempdir)
    # log params
    log_params(
        {
            "n_params": n_params,
            "optim": optim.__class__.__name__,
            "criterion": criterion.__name__,
        }
    )

    # log artifacts
    config.save(tempdir / "unet2d_config.json")
    log_artifact(tempdir / "unet2d_config.json")

    i = 0
    for e in range(1, MAX_EPOCHS + 1):
        for x, y_true in (pbar := tqdm(train_loader, desc=f"Training Epoch {e}/{MAX_EPOCHS}")):
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

            if i % TEST_INTERVAL == 0:
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
    model.save(tempdir / "unet2d_model.pt")
    log_artifact(tempdir / "unet2d_model.pt")
