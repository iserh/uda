from dataclasses import dataclass
from typing import ClassVar, Optional

from ..config import Config


@dataclass
class MAndMsConfig(Config):
    """Configuration for Calgary Campinas Dataset.

    Args:
    """

    phases: tuple[str] = ("ED", "ES")
    unlabeled: bool = False
    flatten: bool = False
    imsize: tuple[int, int, int] = (12, 256, 256)
    offset: tuple[int, int, int] = (0, 0, 0)
    patch_size: Optional[tuple[int, int, int]] = None
    clip_intensities: Optional[tuple[int, int]] = None
    limit: Optional[int] = None
