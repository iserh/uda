import wandb
from pathlib import Path


def download_configuration(run_id: str, project: str, config_dir: str = "config", team: str = "iserh") -> None:
    config_dir = Path(config_dir)

    print(f"Downloading configuration from run {run_id}")

    wandb.restore("config/cc359.yaml", f"{team}/{project}/{run_id}", root=config_dir.parent, replace=True)
    wandb.restore("config/hparams.yaml", f"{team}/{project}/{run_id}", root=config_dir.parent, replace=True)
    try:
        wandb.restore("config/unet.yaml", f"{team}/{project}/{run_id}", root=config_dir.parent, replace=True)
    except Exception:
        pass
    try:
        wandb.restore("config/vae.yaml", f"{team}/{project}/{run_id}", root=config_dir.parent, replace=True)
    except Exception:
        pass


def download_dataset(dataset: type, path: str = "/tmp/data/CC359-Skull-stripping") -> None:
    api = wandb.Api()
    artifact = api.artifact(dataset.artifact_name)
    artifact.download(root=path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    download_configuration(args.run_id, project=args.project, config_dir="config")
