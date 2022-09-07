from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from yaml import Node


class EnumDumper(yaml.SafeDumper):
    def represent_data(self, data: Any) -> Node:
        if isinstance(data, Enum):
            return self.represent_data(data.name)

        if isinstance(data, Path):
            return self.represent_data(str(data))

        if isinstance(data, type):
            return self.represent_data(data.__name__)

        return super(EnumDumper, self).represent_data(data)


@dataclass
class Config:
    """Configuration Class."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            dict_wo_nones = {k: v for k, v in self.__dict__.items() if v is not None}
            yaml.dump(dict_wo_nones, f, Dumper=EnumDumper)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = cls(**yaml.load(f, Loader=yaml.SafeLoader))
        return config
