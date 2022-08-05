import wandb

from uda_wandb.config import RunConfig


def delete_model_binaries(run_cfg: RunConfig) -> None:
    api = wandb.Api()

    run = api.run(run_cfg.run_path)

    model_file = run.file("best_model.pt")
    if model_file.size != 0:
        print(f"Deleted model in run {run.id}")
        model_file.delete()
    else:
        print(f"No model found in run {run.id}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("project", type=str)
    parser.add_argument("run_id", type=str)
    args = parser.parse_args()

    run_cfg = RunConfig(args.run_id, args.project)
    delete_model_binaries(run_cfg, config_dir="config")
