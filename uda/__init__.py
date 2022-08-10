"""Unsupervised Domain Adaptation."""
from importlib.metadata import version

from .config import Config  # noqa: F401
from .hparams import HParams, LossCriterion, Optimizer  # noqa: F401
from .losses import get_criterion, optimizer_cls  # noqa: F401
from .utils import (  # noqa: F401
    flatten_output_transform,
    get_preds_output_transform,
    one_hot_output_transform,
    pipe,
    reshape_to_volume,
    to_cpu_output_transform,
)

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
