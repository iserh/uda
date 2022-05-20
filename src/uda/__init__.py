"""Unsupervised Domain Adaptation."""
from importlib.metadata import version

from .config import HParamsConfig  # noqa: F401
from .unet import UNet, UNetBackbones, UNetConfig  # noqa: F401

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
