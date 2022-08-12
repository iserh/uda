from argparse import ArgumentParser, Namespace
from pathlib import Path

from uda.datasets import DatasetType


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=DatasetType, required=True)

    parser.add_argument("--vae-path", type=str, default="/tmp/models/vae")
    parser.add_argument("--teacher-path", type=str, default="/tmp/models/teacher")

    parser.add_argument("--dl-model", action="store_true")
    parser.add_argument("--data-root", type=Path, default="/tmp/data")
    parser.add_argument("--config-dir", type=Path, default="config")

    parser.add_argument("--project", type=str, default="Test")
    parser.add_argument("-t", "--tags", type=str, default=[], nargs="+")
    parser.add_argument("-g", "--group", type=str, default=None)
    parser.add_argument("-w", "--wandb", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-x", "--cross-eval", action="store_true")
    parser.add_argument("-s", "--store", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args._get_kwargs())
