#!/usr/bin/env python
import shutil
from pathlib import Path
from typing import Union

import torch
import wandb
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from uda.models import VAE, UNet, UNetConfig, VAEConfig


def download_configuration(run_id: str, project: str, path: str = "config", team: str = "iserh") -> None:
    path = Path(path)

    print(f"Downloading configuration from run {run_id}")

    wandb.restore("config/cc359.yaml", f"{team}/{project}/{run_id}", root=path.parent, replace=True)
    wandb.restore("config/hparams.yaml", f"{team}/{project}/{run_id}", root=path.parent, replace=True)
    try:
        wandb.restore("config/unet.yaml", f"{team}/{project}/{run_id}", root=path.parent, replace=True)
    except Exception:
        pass
    try:
        wandb.restore("config/vae.yaml", f"{team}/{project}/{run_id}", root=path.parent, replace=True)
    except Exception:
        pass


def download_dataset(dataset: type, path: str = "/tmp/data/CC359-Skull-stripping") -> None:
    api = wandb.Api()
    artifact = api.artifact(dataset.artifact_name)
    artifact.download(root=path)


def download_model(run_id: str, project: str, team: str = "iserh", path: Union[Path, str] = "/tmp/models/model.pt"):
    path = Path(path)
    print(f"Downloading model from run {run_id}")

    wandb.restore("best_model.pt", f"{team}/{project}/{run_id}", root=path.parent).close()
    shutil.move(path.parent / "best_model.pt", path)


def download_old_model(run_id: str, project: str, team: str = "iserh", root: str = "/tmp/models/model"):
    root = Path(root)
    print(f"Downloading OLD model from run {run_id}")

    try:
        wandb.restore("config/unet.yaml", f"{team}/{project}/{run_id}", root=root, replace=True).close()
        ModelClass = UNet
        ConfigClass = UNetConfig
        config_name = "unet.yaml"
    except Exception:
        pass
    try:
        wandb.restore("config/vae.yaml", f"{team}/{project}/{run_id}", root=root, replace=True).close()
        ModelClass = VAE
        ConfigClass = VAEConfig
        config_name = "vae.yaml"
    except Exception:
        pass

    shutil.move(root / "config" / config_name, root / "config.yaml")
    shutil.rmtree(root / "config")

    wandb.restore("best_model.pt", f"{team}/{project}/{run_id}", root=root).close()

    print(f"Converting OLD model (class={ModelClass.__name__})")

    config = ConfigClass.from_file(root / f"config.yaml")
    model = ModelClass(config)
    # begin workaround
    for m in model.modules():
        if isinstance(m, BatchNorm1d) or isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm3d):
            m.__init__(m.num_features, track_running_stats=True)
    # end workaround
    model.load_state_dict(torch.load(root / "best_model.pt"))

    # begin workaround
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
            del m.running_mean
            del m.running_var
            del m.num_batches_tracked
    # end workaround

    model.save(root.parent / f"{root.name}.pt")
    shutil.rmtree(root)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from uda.datasets import CC359

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    parser.add_argument("-o", "--object", type=str, default="config")
    parser.add_argument("--path", type=Path, default="config")
    args = parser.parse_args()

    if args.object == "old-model":
        download_old_model(args.run_id, project=args.project, root=args.path)
    elif args.object == "model":
        download_model(args.run_id, project=args.project, path=args.path)
    elif args.object == "dataset":
        download_dataset(CC359)
    elif args.object == "config":
        download_configuration(args.run_id, project=args.project, path=args.path)
