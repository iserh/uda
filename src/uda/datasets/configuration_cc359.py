from typing import Optional, Tuple

from uda.config import Config


class CC359Config(Config):
    """Configuration for Calgary Campinas Dataset."""

    def __init__(
        self,
        vendor: str,
        fold: Optional[int] = None,
        rotate: bool = True,
        flatten: bool = False,
        patchify: Optional[Tuple[int]] = None,
        flatten_patches: bool = True,
        clip_intensities: Optional[Tuple[int]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Args:
        `data_path` : Dataset location
        `vendor` : vendor
        `fold` : Fold index for cross-validation
        `rotate` : Rotate the images
        `flatten` : Flatten the Z dimension
        `patchify` : Patchify the images
        `flatten_patches` : Flatten the patches
        `random_state` : Random state for cross-validation
        """
        self.vendor = vendor
        self.fold = fold
        self.rotate = rotate
        self.flatten = flatten
        self.patchify = patchify
        self.flatten_patches = flatten_patches
        self.clip_intensities = clip_intensities
        self.random_state = random_state
