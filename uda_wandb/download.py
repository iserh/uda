#!/usr/bin/env python
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import torch
import wandb
import yaml
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from uda.models import VAE, UNet, UNetConfig, VAEConfig
from uda_wandb.config import RunConfig


def _move_all_files(src: Union[Path, str], dest: Union[Path, str]) -> None:
    dest = Path(dest)
    src = Path(src)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for file_path in src.iterdir():
        shutil.move(file_path, dest)


def download_dataset(dataset: type, path: Union[Path, str] = "/tmp/data/CC359-Skull-stripping") -> Path:
    path = Path(path)

    api = wandb.Api()
    artifact = api.artifact(dataset.artifact_name)
    path = artifact.download(root=path)
    return path


def download_config(run_cfg: RunConfig, path: Union[Path, str] = "config", old: bool = False) -> Path:
    path = Path(path)

    if old:
        return download_old_config(run_cfg, path)

    print(f"Downloading config from run {run_cfg.run_id}")
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        wandb.restore("config/dataset.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
        wandb.restore("config/hparams.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
        wandb.restore("config/model.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
        _move_all_files(tmpdir / "config", dest=path)

    run_cfg.save(path / "run_config.yaml")
    return path


def download_old_config(
    run_cfg: RunConfig, path: Union[Path, str] = "config", ret_model_type: bool = False
) -> Union[tuple[Path, str], Path]:
    path = Path(path)
    print(f"Downloading OLD config from run {run_cfg.run_id}")

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # dataset
        wandb.restore("config/cc359.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
        shutil.move(tmpdir / "config" / "cc359.yaml", tmpdir / "config" / "dataset.yaml")
        # hyperparameters
        with wandb.restore("config/hparams.yaml", run_cfg.run_path, root=tmpdir, replace=True) as f:
            # remove old key 'early_stopping'
            hparams = yaml.load(f, Loader=yaml.SafeLoader)
            if "early_stopping" in hparams.keys():
                del hparams["early_stopping"]
        with open(tmpdir / "config" / "hparams.yaml", "w") as f:
            yaml.dump(hparams, f)
        # model
        try:
            wandb.restore("config/unet.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
            shutil.move(tmpdir / "config" / "unet.yaml", tmpdir / "config" / "model.yaml")
            model_type = "UNet"
        except Exception:
            pass
        try:
            wandb.restore("config/vae.yaml", run_cfg.run_path, root=tmpdir, replace=True).close()
            shutil.move(tmpdir / "config" / "vae.yaml", tmpdir / "config" / "model.yaml")
            model_type = "VAE"
        except Exception:
            pass

        _move_all_files(tmpdir / "config", dest=path)

    run_cfg.save(path / "run_config.yaml")
    return path, model_type if ret_model_type else path


def download_model(run_cfg: RunConfig, path: Union[Path, str] = "/tmp/models/model", old: bool = False) -> Path:
    path = Path(path)

    if old:
        return download_old_model(run_cfg, path)

    download_config(run_cfg, path)
    print(f"Downloading model from run {run_cfg.run_id}")
    wandb.restore("best_model.pt", run_cfg.run_path, root=path).close()

    run_cfg.save(path / "run_config.yaml")
    return path / "best_model.pt"


def download_old_model(run_cfg: RunConfig, path: Union[Path, str] = "/tmp/models/model") -> Path:
    path = Path(path)

    _, model_type = download_old_config(run_cfg, path, ret_model_type=True)

    ModelClass = UNet if model_type == "UNet" else VAE
    ConfigClass = UNetConfig if model_type == "UNet" else VAEConfig

    print(f"Downloading OLD model from run {run_cfg.run_id}")
    wandb.restore("best_model.pt", run_cfg.run_path, root=path).close()

    print(f"Converting OLD model (class={ModelClass.__name__})")

    config = ConfigClass.from_file(path / f"model.yaml")
    model = ModelClass(config)
    # begin workaround
    for m in model.modules():
        if isinstance(m, BatchNorm1d) or isinstance(m, BatchNorm2d) or isinstance(m, BatchNorm3d):
            m.__init__(m.num_features, track_running_stats=True)
    # end workaround
    model.load_state_dict(torch.load(path / "best_model.pt"))

    # begin workaround
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
            del m.running_mean
            del m.running_var
            del m.num_batches_tracked
    # end workaround

    model.save(path / "best_model.pt")
    return path / "best_model.pt"


if __name__ == "__main__":
    from argparse import ArgumentParser

    from uda.datasets import CC359

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    parser.add_argument("--object", type=str, default="config")
    parser.add_argument("--path", type=Path, default="config")
    parser.add_argument("-o", "--old", action="store_true")
    args = parser.parse_args()

    run_cfg = RunConfig(args.run_id, args.project)

    if args.object == "config":
        download_config(run_cfg, path=args.path, old=args.old)
    elif args.object == "model":
        download_model(run_cfg, path=args.path, old=args.old)
    elif args.object == "dataset":
        download_dataset(CC359)