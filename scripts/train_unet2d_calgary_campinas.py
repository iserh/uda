"""Train U-Net 2D model on Calgary images."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torchsummaryX import summary
from tqdm import tqdm

from uda import UNet, UNetConfig
from uda.calgary_campinas_dataset import CalgaryCampinasDataset

sns.set_theme(style="darkgrid")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Using device: {device}")

data_dir = Path("/home/iailab36/iser/uda-data")
output_dir = Path("/home/iailab36/iser/uda-data/output")


train_dataset = CalgaryCampinasDataset(data_dir, vendor="GE_3", fold=1, train=True, rotate=True, flatten=True)
test_dataset = CalgaryCampinasDataset(data_dir, vendor="GE_3", fold=1, train=False, rotate=True, flatten=True)

print(train_dataset.data.shape)
print(train_dataset.label.shape)
print(train_dataset.voxel_dim.shape)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


# We use one encoder block less than the original U-Net, since our input is of shape (256, 256)
config = UNetConfig(
    n_classes=1,
    encoder_blocks=(
        (1, 64, 64),
        (64, 128, 128),
        (128, 256, 256),
        (256, 512, 512),
    ),
    decoder_blocks=(
        (512, 256, 256),
        (256, 128, 128),
        (128, 64, 64),
    ),
)


model = UNet(config)
summary(model, x=torch.randn(1, 1, 512, 512))


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum())


model = UNet(config).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

TEST_INTERVAL = 100
MAX_STEPS = 2_000

train_losses, test_losses = [], []
train_dscs, test_dscs = [], []

i = 0
dsc_test = 0
with tqdm(total=MAX_STEPS, desc="Training") as pbar:
    while i < MAX_STEPS:
        for x, y_true, _ in train_loader:
            i += 1
            if i > MAX_STEPS:
                break

            x = x.to(device)
            y_true = y_true.to(device)

            optim.zero_grad()
            y_pred = model(x).sigmoid()
            loss = F.binary_cross_entropy(y_pred, y_true)

            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            train_dscs.append(dice_score(y_true.cpu(), y_pred.detach().cpu()).item())

            del loss, y_pred, x, y_true

            if i % TEST_INTERVAL == 0:
                r_loss, r_dsc = 0, 0
                with torch.no_grad():
                    for x, y_true, _ in tqdm(test_loader, desc="Testing", leave=False):
                        x = x.to(device)
                        y_true = y_true.to(device)

                        y_pred = model(x).sigmoid()
                        r_loss += F.binary_cross_entropy(y_pred, y_true).item()
                        r_dsc += dice_score(y_true.cpu(), y_pred.cpu()).item()

                del y_pred, x, y_true

                test_losses.append(r_loss / len(test_loader))
                test_dscs.append(r_loss / len(test_loader))

            pbar.set_postfix(
                {
                    "loss": sum(train_losses[-10:]) / 10,
                    "dsc_train": sum(train_dscs[-10:]) / 10,
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
plt.savefig(output_dir / "metrics.png")


config.save(output_dir / "unet_config.json")
model.save(output_dir / "unet_model.pt")
