"""Unsupervised Domain Adaptation."""
from importlib.metadata import version

from .hparams import HParams, LossCriterion, Optimizer  # noqa: F401
from .config import Config  # noqa: F401
from .losses import get_criterion, optimizer_cls  # noqa: F401
from .utils import (  # noqa: F401
    binary_one_hot_output_transform,
    flatten_output_transform,
    pipe,
    reshape_to_volume,
    sigmoid_round_output_transform,
    to_cpu_output_transform,
)

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
