#!/usr/bin/env python
from pathlib import Path

from uda import HParams
from uda.datasets import UDADataset
from uda.models import UNet
from tqdm import tqdm
import ignite.distributed as idist


def run(teacher: UNet, dataset: UDADataset, hparams: HParams, use_wandb: bool = False) -> None:
    if use_wandb:
        import wandb

    dataset.setup()
    train_loader = dataset.train_dataloader(hparams.train_batch_size)

    model = UNet(teacher.config)
    model.load_state_dict(teacher.state_dict())  # copy weights
    model = model.train().to(idist.device())

    cache_dir = Path(wandb.run.dir if use_wandb else "/tmp/models/student")
    cache_dir.mkdir(exist_ok=True, parents=True)

    for x, _ in tqdm(train_loader, desc="Predicting train data"):
        model(x.to(idist.device()))

    model = model.cpu().eval()
    model.save(cache_dir / "best_model.pt")


if __name__ == "__main__":
    from commons import get_launch_config, get_model_path

    launch = get_launch_config()

    # load configuration
    hparams = HParams.from_file(launch.config_dir / "hparams.yaml")
    dataset: UDADataset = launch.dataset(launch.config_dir / "dataset.yaml", root=launch.data_root)
    # information about teacher / vae
    teacher_path = get_model_path(launch.teacher, launch.download_model)
    if launch.wandb:
        from wandb_utils import RunConfig

        teacher_run = RunConfig.from_file(teacher_path / "run_config.yaml")
        teacher_dataset = launch.dataset(teacher_path / "dataset.yaml")
    # load models
    teacher = UNet.from_pretrained(teacher_path / "best_model.pt")

    # finetune on all given vendors
    for vendor in launch.vendors:
        dataset.config.vendor = vendor
        dataset.vendor = vendor

        if launch.wandb:
            import wandb
            from evaluation import evaluate, evaluate_vendors

            from uda.trainer import SegEvaluator
            from wandb_utils import delete_model_binaries, download_dataset

            with wandb.init(
                project=launch.project,
                tags=launch.tags,
                group=launch.group,
                config={
                    "info": "BatchNorm-Adaption",
                    "hparams": hparams.__dict__,
                    "dataset": dataset.config.__dict__,
                    "model": teacher.config.__dict__,
                    "teacher_dataset": teacher_dataset.config.__dict__,
                    "teacher_run": teacher_run.__dict__,
                },
            ) as r:
                run_cfg = RunConfig(r.id, r.project)
                r.log_code()
                # save the configuration to the wandb run dir
                cfg_dir = Path(r.dir) / "config"
                cfg_dir.mkdir()
                launch.save(cfg_dir / "launch.yaml")
                hparams.save(cfg_dir / "hparams.yaml")
                teacher.config.save(cfg_dir / "model.yaml")
                dataset.config.save(cfg_dir / "dataset.yaml")

                download_dataset(dataset)
                run(teacher, dataset, hparams, use_wandb=True)

                if launch.evaluate:
                    evaluate(SegEvaluator, UNet, dataset, hparams, splits=["validation", "testing"])
                if launch.evaluate_vendors:
                    evaluate_vendors(SegEvaluator, UNet, dataset, hparams, vendors=[dataset.vendor])

            if not launch.store_model:
                delete_model_binaries(run_cfg)
        else:
            run(teacher, dataset, hparams, use_wandb=False)
