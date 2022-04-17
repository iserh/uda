from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

from uda import UNet, UNetConfig
from uda.calgary_campinas_dataset import CalgaryCampinasDataset

data_dir = Path("/home/iailab36/iser/uda-data")
output_dir = Path("/home/iailab36/iser/uda-data/output")
output_dir.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="darkgrid")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")

train_dataset = CalgaryCampinasDataset(
    data_dir, vendor="GE_3", fold=1, train=True, patchify=(64, 128, 128), squash_patches=True
)
test_dataset = CalgaryCampinasDataset(
    data_dir, vendor="GE_3", fold=1, train=False, patchify=(64, 128, 128), squash_patches=True
)

print(train_dataset.data.shape)
print(train_dataset.label.shape)
print(train_dataset.voxel_dim.shape)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

config = UNetConfig(
    n_classes=1,
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


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum())


model = UNet(config).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

TEST_INTERVAL = 100
MAX_STEPS = 5_000

train_losses, test_losses = [], []
train_dscs, test_dscs = [], []

i = 0
dsc_test = 0
with tqdm(total=MAX_STEPS, desc="Training") as pbar:
    while i < MAX_STEPS:
        for x, y_true in train_loader:
            i += 1
            if i > MAX_STEPS:
                break

            x = x.to(device)
            y_true = y_true.to(device)

            optim.zero_grad()
            y_pred = model(x)

            loss = F.binary_cross_entropy(y_pred, y_true)

            loss.backward()
            optim.step()

            pred_mask = (y_pred.detach() > 0.5).float()
            train_losses.append(loss.item())
            train_dscs.append(dice_score(pred_mask, y_true).item())

            if i % TEST_INTERVAL == 0:
                preds, targets = [], []
                with torch.no_grad():
                    preds, targets = [*zip(*[(model(x.to(device)).cpu(), y_true) for x, y_true in test_loader])]
                    preds = torch.cat(preds)
                    targets = torch.cat(targets)

                    pred_mask = (preds > 0.5).float()
                    test_losses.append(F.binary_cross_entropy(preds, targets).item())
                    test_dscs.append(dice_score(pred_mask, targets).item())

            pbar.set_postfix(
                {
                    "loss": sum(train_losses[-5:]) / 5,
                    "dsc_train": sum(train_dscs[-5:]) / 5,
                    "dsc_test": test_dscs[-1] if test_dscs != [] else 0,
                }
            )
            pbar.update()

model.cpu()

_, ax = plt.subplots(1, 2, figsize=(10, 4))

sns.lineplot(x=range(len(train_losses)), y=train_losses, label="train", ax=ax[0])
sns.lineplot(x=range(TEST_INTERVAL, len(train_losses) + 1, TEST_INTERVAL), y=test_losses, label="test", ax=ax[0])

sns.lineplot(x=range(len(train_dscs)), y=train_dscs, label="train", ax=ax[1])
sns.lineplot(x=range(TEST_INTERVAL, len(train_dscs) + 1, TEST_INTERVAL), y=test_dscs, label="test", ax=ax[1])

plt.legend()
plt.savefig(output_dir / "metrics.pdf")

config.save(output_dir / "unet3d_config.json")
model.save(output_dir / "unet3d_model.pt")
