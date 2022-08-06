"""Unsupervised Domain Adaptation - WandB integrations."""
from importlib.metadata import version

from .config import RunConfig
from .delete import delete_model_binaries  # noqa: F401
from .download import download_config, download_dataset  # noqa: F401
from .evaluation import (  # noqa: F401
    cross_evaluate_unet,
    evaluate_unet,
    evaluate_vae,
    segmentation_table_plot,
    vae_table_plot,
)

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
