from uda import Config
from dataclasses import dataclass


@dataclass
class RunConfig(Config):
    run_id: str
    project: str
    team: str = "iserh"

    @property
    def run_path(self) -> str:
        return f"{self.team}/{self.project}/{self.run_id}"
