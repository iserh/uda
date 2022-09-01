from dataclasses import dataclass

from uda import Config


@dataclass
class RunConfig(Config):
    run_id: str
    project: str
    team: str = "iserh"

    @property
    def run_path(self) -> str:
        return f"{self.team}/{self.project}/{self.run_id}"

    @classmethod
    def parse_path(cls, run_path: str) -> "RunConfig":
        components = run_path.split("/")
        return cls(*reversed(components))
