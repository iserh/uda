from .base import UDADataset  # noqa: F401
from .configuration_cc359 import CC359Config  # noqa: F401
from .configuration_mms import MAndMsConfig  # noqa: F401
from .dataset_cc359 import CC359  # noqa: F401
from .dataset_mms import MAndMs  # noqa: F401
from .dataset_teacher import TeacherData  # noqa: F401


class DatasetType:
    def __new__(cls, name: str) -> None:
        if name == CC359.__name__:
            return CC359
        elif name == MAndMs.__name__:
            return MAndMs
        else:
            raise ValueError(f"Dataset '{name}' does not exist.")
