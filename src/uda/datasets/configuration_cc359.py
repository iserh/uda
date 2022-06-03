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
    `flatten` : Flatten the Z dimension
    `patch_dims` : Patch dimensions for slicing the images
    `flatten_patches` : Flatten the patches
    `random_state` : Random state for cross-validation
    """

    vendor: str
    fold: Optional[int] = None
    rotate: bool = True
    flatten: bool = True
    patch_dims: Optional[Tuple[int, int, int]] = None
    flatten_patches: bool = True
    clip_intensities: Optional[Tuple[int, int]] = None
    random_state: Optional[int] = None
