from dataclasses import dataclass
from typing import ClassVar, Optional

from uda.config import Config


@dataclass
class CC359Config(Config):
    """Configuration for Calgary Campinas Dataset.

    Args:
        vendor (str): Vendor of the scanner.
        fold (str, optional): Fold of cv to select. Must be either 0 / 1 / 2. Defaults to None (No splitting).
        rotate (int, optional): Order of rotation of the images. Defaults to None (no rotation)
        flatten (bool): Flatten 3d volumes into 2d images. Defaults to False.
        imsize (tuple[int, int, int]): Size the loaded volumes will get padded/cropped to. Defaults to (192, 256, 256).
        patch_size (tuple[int, int, int], optional): Size of volume patches. Defaults to None (No patchification).
        clip_intensities (tuple[int, int], optional): Clip the intensities of the scanner images.
        Defaults to None (No clipping).
        random_state (int, optional): Random state for reproducibility (e.g. of kfold splits).
    """

    name: ClassVar[str] = "CC359"

    vendor: str
    fold: Optional[int] = None
    rotate: Optional[int] = 1
    flatten: bool = False
    imsize: tuple[int, int, int] = (192, 256, 256)
    patch_size: Optional[tuple[int, int, int]] = None
    clip_intensities: Optional[tuple[int, int]] = None
    limit: Optional[int] = None
    random_state: Optional[int] = None
