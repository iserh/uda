from pathlib import Path

import wandb


def restore_config(run_id: str, root: Path) -> None:
    print(f"Downloading configuration from run {run_id}")

    wandb.restore("config/cc359.yaml", f"tiser/UDA/{run_id}", root=root, replace=True)
    wandb.restore("config/hparams.yaml", f"tiser/UDA/{run_id}", root=root, replace=True)
    wandb.restore("config/unet.yaml", f"tiser/UDA/{run_id}", root=root, replace=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    restore_config(args.run_id, Path.cwd())
