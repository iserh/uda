from dataclasses import dataclass
from typing import Optional

from ..config import Config


@dataclass
class MAndMsConfig(Config):
    """Configuration for Calgary Campinas Dataset.

    Args:
    """

    vendor: str
    fold: Optional[int] = None
    phases: tuple[str] = ("ED", "ES")
    flatten: bool = False
    imsize: tuple[int, int, int] = (12, 256, 256)
    patch_size: Optional[tuple[int, int, int]] = None
    clip_intensities: Optional[tuple[int, int]] = None
    limit: Optional[int] = None
    random_state: Optional[int] = None
