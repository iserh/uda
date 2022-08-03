"""Unsupervised Domain Adaptation - WandB integrations."""
from importlib.metadata import version

from .evaluation import vae_table_plot, evaluate_vae, evaluate_unet, cross_evaluate_unet  # noqa: F401
from .delete import delete_model_binaries  # noqa: F401
from .download import download_configuration  # noqa: F401

try:
    __version__ = version(__name__)
except Exception:
    __version__ = "unknown"
