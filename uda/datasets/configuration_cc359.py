from dataclasses import dataclass
from typing import Optional

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
    imsize: tuple[int, int, int] = (192, 256, 256)
    patch_size: Optional[tuple[int, int, int]] = None
    clip_intensities: Optional[tuple[int, int]] = None
    random_state: Optional[int] = None
