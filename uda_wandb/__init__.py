"""Unsupervised Domain Adaptation - WandB integrations."""
from importlib.metadata import version

from .config import RunConfig  # noqa: F401
from .delete import delete_model_binaries  # noqa: F401
from .download import download_config, download_dataset, download_model  # noqa: F401
from .evaluation import cross_evaluate, evaluate, prediction_image_plot  # noqa: F401

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
