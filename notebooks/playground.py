"""Test the Unet model."""
import torch
from torchsummaryX import summary

from uda import UNet, UNetConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# StackedConvolutions U-Net configuration (like paper)
config = UNetConfig(
    n_classes=2,
    dim=3,
    encoder_blocks=(
        (1, 8, 8, 8),
        (8, 8, 16, 16, 16),
        (16, 16, 32, 32, 32),
        (32, 32, 64, 64, 64),
        (64, 64, 128, 128, 128),
        (128, 128, 256, 256, 256),
    ),
    decoder_blocks=(
        (256, 128, 128, 128),
        (128, 64, 64, 64),
        (64, 32, 32, 32),
        (32, 16, 16, 16),
        (16, 8, 8, 8),
    ),
)

# # ResNet UNet configuration
# config = UNetConfig(
#     n_classes=2,
#     dim=3,
#     encoder_blocks=(
#         (1, 8, 8),
#         (8, 16, 16),
#         (16, 32, 32),
#         (32, 64, 64),
#         (64, 128, 128),
#         (128, 256, 256),
#     ),
#     decoder_blocks=(
#         (256, 128),
#         (128, 64),
#         (64, 32),
#         (32, 16),
#         (16, 8),
#     ),
#     encoder_backbone=UNetBackbones.ResNet,
#     decoder_backbone=UNetBackbones.ResNet,
#     batch_norm_after_encoder=False,
# )

model = UNet(config)
model = model.to(device)
x = torch.randn(1, 1, 64, 64, 64).to(device)

# test model
y = model(x).detach().cpu()
print("Works! Output Shape: ", y.shape)
print()

# model summary
df = summary(model, x).reset_index()
# print only convolutional layers
print(
    df[df["Layer"].apply(lambda layer: ("Conv3d" in layer or "Conv2d" in layer) and "conv_shortcut" not in layer)]
)
