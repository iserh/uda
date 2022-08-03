from argparse import ArgumentParser, Namespace
from pathlib import Path


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", type=Path, default="/tmp/data/CC359-Skull-stripping")
    parser.add_argument("-c", "--config", type=Path, default="config")
    parser.add_argument("-w", "--wandb", action="store_true")
    parser.add_argument("-p", "--project", type=str, default="Test")
    parser.add_argument("-t", "--tags", type=str, default=[], nargs="+")
    parser.add_argument("-g", "--group", type=str, default=None)
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("-x", "--cross-eval", action="store_true")
    parser.add_argument("-s", "--store", action="store_true")
    return parser.parse_args()
