from dataclasses import dataclass
from typing import Optional, Tuple

from uda.config import Config


@dataclass
class CC359Config(Config):
    """Configuration for Calgary Campinas Dataset.

    `data_path` : Dataset location
    `vendor` : vendor
    `fold` : Fold index for cross-validation
    `rotate` : Rotate the images
    `flatten` : Flatten 3D volumes to 2D images
    `patch_size` : Patch dimensions for slicing the images
    `random_state` : Random state for cross-validation
    """

    vendor: str
    fold: Optional[int] = None
    rotate: Optional[int] = 1
    flatten: bool = False
    patch_size: Optional[Tuple[int, int, int]] = None
    clip_intensities: Optional[Tuple[int, int]] = None
    random_state: Optional[int] = None
