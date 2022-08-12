"""Unsupervised Domain Adaptation."""
from importlib.metadata import version

from .config import Config  # noqa: F401
from .hparams import HParams, get_loss_cls, get_optimizer_cls  # noqa: F401

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
