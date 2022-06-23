import wandb


def delete_model_binaries(run_id: str, project: str) -> None:
    api = wandb.Api()

    run = api.run(f"iserh/{project}/{run_id}")

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

    delete_model_binaries(args.run_id, project=args.project)
