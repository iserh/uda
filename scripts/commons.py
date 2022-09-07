from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from uda import Config
from uda.datasets import DatasetType
from wandb_utils import RunConfig, download_model


@dataclass
class LaunchConfig(Config):
    dataset: DatasetType
    vendors: Optional[list[str]] = None
    vae: Optional[str] = None
    teacher: Optional[str] = None
    download_model: bool = None
    data_root: Path = Path("/tmp/data")
    config_dir: Path = Path("config")
    project: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    group: Optional[str] = None
    wandb: bool = False
    evaluate: bool = False
    evaluate_vendors: bool = False
    store_model: bool = True

    def __post_init__(self) -> None:
        self.dataset = DatasetType(self.dataset)
        self.data_root = Path(self.data_root)
        self.config_dir = Path(self.config_dir)
        self.config_dir = Path(self.config_dir)


def get_model_path(path: str, download: bool) -> Path:
    if download:
        run: RunConfig = RunConfig.parse_path(path)
        return download_model(run, path=f"/tmp/models/{run.run_id}").parent
    else:
        return Path(path)


def get_launch_config() -> LaunchConfig:
    parser = ArgumentParser()
    parser.add_argument("launch_path", nargs="?", type=Path, default="config/launch.yaml")
    args = parser.parse_args()

    return LaunchConfig.from_file(args.launch_path)


if __name__ == "__main__":
    launch_cfg = LaunchConfig.from_file("config/launch.yaml")
    print(launch_cfg.__dict__)

    launch_cfg.save("/tmp/launch.yaml")
