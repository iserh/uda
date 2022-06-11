from pathlib import Path

import wandb


def restore_config(run_id: str, project: str, root: Path = Path.cwd()) -> None:
    print(f"Downloading configuration from run {run_id}")

    wandb.restore("config/cc359.yaml", f"tiser/{project}/{run_id}", root=root, replace=True)
    wandb.restore("config/hparams.yaml", f"tiser/{project}/{run_id}", root=root, replace=True)
    try:
        wandb.restore("config/unet.yaml", f"tiser/{project}/{run_id}", root=root, replace=True)
    except Exception:
        pass
    try:
        wandb.restore("config/vae.yaml", f"tiser/{project}/{run_id}", root=root, replace=True)
    except Exception:
        pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    restore_config(args.run_id, project=args.project, root=Path.cwd())
