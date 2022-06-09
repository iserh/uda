"""Unsupervised Domain Adaptation."""
from importlib.metadata import version

from .datasets import CC359, CC359Config  # noqa: F401
from .hparams import HParams, LossCriterion, Optimizer  # noqa: F401
from .models import VAE, UNet, UNetConfig, VAEConfig  # noqa: F401

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
