from dataclasses import dataclass
from enum import Enum
from typing import Any

import yaml
from yaml import Node


class EnumDumper(yaml.SafeDumper):
    def represent_data(self, data: Any) -> Node:
        if isinstance(data, Enum):
            return self.represent_data(data.name)
        return super(EnumDumper, self).represent_data(data)


@dataclass
class Config:
    """Configuration Class."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, Dumper=EnumDumper)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as f:
            config = cls(**yaml.load(f, Loader=yaml.SafeLoader))
        return config
