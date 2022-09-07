#!/usr/bin/env python
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import torch
import wandb
import yaml
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

from uda.datasets import DatasetType, UDADataset
from uda.models import VAE, UNet, UNetConfig, VAEConfig
from wandb_utils.config import RunConfig


def _move_all_files(src: str, dest: str) -> None:
    dest: Path = Path(dest)
    src: Path = Path(src)

    dest.mkdir(parents=True, exist_ok=True)

    for file_path in src.iterdir():
        if (dest / file_path.name).exists():
            os.remove(dest / file_path.name)

        shutil.move(file_path, dest)


def download_dataset(dataset: Union[type[UDADataset], UDADataset], root: str = "/tmp/data") -> Path:
    root = Path(root)
    if isinstance(dataset, UDADataset):
        dataset = dataset.__class__

    api = wandb.Api()
    artifact = api.artifact(dataset.artifact_name)
    path = artifact.download(root=root / dataset.__name__)
    return path


def download_config(run_cfg: RunConfig, path: str = "config", old: bool = False) -> Path:
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

    import yaml

    with open(path / "model.yaml", "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if "model_name" in model_cfg.keys():
        print("model_name found")
        del model_cfg["model_name"]
    if "track_running_stats" in model_cfg.keys():
        print("track_running_stats found")
        del model_cfg["track_running_stats"]
    with open(path / "model.yaml", "w") as f:
        model_cfg = yaml.dump(model_cfg, f)

    run_cfg.save(path / "run_config.yaml")
    return path


def download_old_config(
    run_cfg: RunConfig, path: str = "config", ret_model_type: bool = False
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


def download_model(run_cfg: RunConfig, path: str = "/tmp/models/model", old: bool = False) -> Path:
    path = Path(path)

    if old:
        return download_old_model(run_cfg, path)

    download_config(run_cfg, path)
    print(f"Downloading model from run {run_cfg.run_id}")
    wandb.restore("best_model.pt", run_cfg.run_path, root=path).close()

    run_cfg.save(path / "run_config.yaml")
    return path / "best_model.pt"


def download_old_model(run_cfg: RunConfig, path: str = "/tmp/models/model") -> Path:
    path = Path(path)

    _, model_type = download_old_config(run_cfg, path, ret_model_type=True)

    ModelClass = UNet if model_type == "UNet" else VAE
    ConfigClass = UNetConfig if model_type == "UNet" else VAEConfig

    print(f"Downloading OLD model from run {run_cfg.run_id}")
    wandb.restore("best_model.pt", run_cfg.run_path, root=path).close()

    print(f"Converting OLD model (class={ModelClass.__name__})")

    config = ConfigClass.from_file(path / "model.yaml")
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

    parser = ArgumentParser()
    parser.add_argument("-p", "--project", type=str, default="")
    parser.add_argument("-r", "--run", type=str, default="")
    parser.add_argument("--dataset", type=DatasetType, default="CC359")
    parser.add_argument("--object", type=str, default="config")
    parser.add_argument("--path", type=Path, default="config")
    parser.add_argument("-o", "--old", action="store_true")
    args = parser.parse_args()

    run_cfg = RunConfig(args.run, args.project)

    if args.object == "config":
        download_config(run_cfg, path=args.path, old=args.old)
    elif args.object == "model":
        download_model(run_cfg, path=args.path, old=args.old)
    elif args.object == "dataset":
        download_dataset(args.dataset)
