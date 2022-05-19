import shutil
from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlparse

import mlflow
from mlflow.entities.view_type import ViewType
from mlflow.tracking import MlflowClient

parser = ArgumentParser()
parser.add_argument("--backend-store-uri", type=str)

args = parser.parse_args()

backend_store = Path(urlparse(args.backend_store_uri).path)
mlflow.set_tracking_uri("http://localhost:5000")


def remove_run_dir(run_id: str) -> None:
    for hit in backend_store.glob("**/" + run_id):
        shutil.rmtree(hit, ignore_errors=True)


def remove_artifact_dir(artifacts_uri: str) -> None:
    artifact_dir = Path(urlparse(artifacts_uri).path).parent
    shutil.rmtree(artifact_dir, ignore_errors=True)


client = MlflowClient()
client.list_experiments()

for exp in client.list_experiments(ViewType.ALL):
    print(f"Experiment {exp.experiment_id}: {exp.name}")
    for run in client.search_runs(exp.experiment_id, run_view_type=ViewType.DELETED_ONLY):
        print(f"\tRemoving run {run.info.run_id}")
        remove_artifact_dir(run.info.artifact_uri)
        remove_run_dir(run.info.run_id)
