#!/usr/bin/env python
from pathlib import Path

import ignite.distributed as idist
import torch.nn as nn
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore
from torch.optim.lr_scheduler import LinearLR

from uda import HParams, get_criterion, optimizer_cls, pipe, sigmoid_round_output_transform, to_cpu_output_transform
from uda.datasets import CC359
from uda.datasets.dataset_teacher import TeacherData
from uda.models import UNet, UNetConfig
from uda.trainer import JointTrainer, joint_standard_metrics
from uda_wandb import segmentation_table_plot


def run(dataset: CC359, teacher: nn.Module, vae: nn.Module, hparams: HParams, model_config: UNetConfig) -> None:
    teacher_data = TeacherData(teacher, dataset)
    teacher_data.setup(hparams.val_batch_size)

    model = UNet(model_config).to(idist.device())
    model.load_state_dict(teacher.state_dict())  # copy weights
    optim = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    loss_fn = get_criterion(hparams.criterion)(**hparams.loss_kwargs)

    trainer = JointTrainer(
        model=model,
        vae=vae,
        optim=optim,
        loss_fn=loss_fn,
        lambd=hparams.vae_lamdb,
        train_loader=teacher_data.train_dataloader(hparams.val_batch_size),
        val_loader=teacher_data.val_dataloader(hparams.val_batch_size),
        test_loader=dataset.val_dataloader(hparams.val_batch_size),
        patience=hparams.early_stopping_patience,
        metrics=joint_standard_metrics(loss_fn),
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Train(Eval)", persist=True).attach(trainer.train_evaluator)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)
    ProgressBar(desc="Test", persist=True).attach(trainer.test_evaluator)

    trainer.run(teacher_data.train_dataloader(hparams.train_batch_size), max_epochs=hparams.epochs)


def run_with_wandb(dataset: CC359, hparams: HParams, model_config: UNetConfig) -> None:
    teacher_data = TeacherData(teacher, dataset)
    teacher_data.setup(hparams.val_batch_size)
    train_loader = teacher_data.train_dataloader(hparams.val_batch_size)

    model = UNet(model_config).to(idist.device())
    model.load_state_dict(teacher.state_dict())  # copy weights
    optim = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    schedule = LinearLR(optim, 0.01, 1.0, len(train_loader))
    loss_fn = get_criterion(hparams.criterion)(**hparams.loss_kwargs)

    # log model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.summary.update({"n_parameters": n_params})

    trainer = JointTrainer(
        model=model,
        vae=vae,
        optim=optim,
        schedule=schedule,
        loss_fn=loss_fn,
        lambd=hparams.vae_lamdb,
        train_loader=train_loader,
        val_loader=teacher_data.val_dataloader(hparams.val_batch_size),
        test_loader=teacher_data.dataset.val_dataloader(hparams.val_batch_size),
        patience=hparams.early_stopping_patience,
        metrics=joint_standard_metrics(loss_fn),
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Train(Eval)", persist=True).attach(trainer.train_evaluator)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)
    ProgressBar(desc="Test", persist=True).attach(trainer.test_evaluator)

    # wandb logger
    wandb_logger = WandBLogger(id=wandb.run.id)
    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda o: {"pseudo_batchloss": o[0], "rec_batchloss": o[1]},
    )
    # wandb table evaluation
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=segmentation_table_plot,
        evaluator=trainer.val_evaluator,
        data=dataset.val_split.tensors[0],
        imsize=dataset.imsize,
        patch_size=dataset.patch_size,
    )
    # table evaluation functions needs predictions from validation set
    eos = EpochOutputStore(
        output_transform=pipe(lambda o: (o[2], o[1]), sigmoid_round_output_transform, to_cpu_output_transform)
    )
    eos.attach(trainer.val_evaluator, "output")

    for tag, evaluator in [
        ("training", trainer.train_evaluator),
        ("validation", trainer.val_evaluator),
        ("test", trainer.test_evaluator),
    ]:
        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=list(trainer.metrics.keys()),
            global_step_transform=lambda *_: trainer.state.iteration,
        )

    trainer.run(train_loader, max_epochs=hparams.epochs)

    wandb_logger.close()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from commons import get_args

    from uda.models import VAE, UNet
    from uda_wandb import cross_evaluate_unet, delete_model_binaries, download_dataset, evaluate_unet

    args = get_args()

    # load configuration
    hparams = HParams.from_file(args.config / "hparams.yaml")
    model_config = UNetConfig.from_file(args.config / "unet.yaml")
    dataset = CC359.from_preconfigured(args.config / "cc359.yaml", root=args.data)
    download_dataset(dataset)

    vae = VAE.from_pretrained(args.vae_path)
    teacher = UNet.from_pretrained(args.teacher_path)

    if args.wandb:
        r = wandb.init(
            project=args.project,
            tags=args.tags,
            group=args.group,
            config={
                "hparams": hparams.__dict__,
                "dataset": dataset.config.__dict__,
                "model": model_config.__dict__,
            },
        )

        # we have to copy the config files to a tmp dir, because the original files might change during wandb syncing
        with TemporaryDirectory() as tmpdir:
            cfg_dir = Path(tmpdir) / "config"
            cfg_dir.mkdir(parents=True)

            hparams.save(cfg_dir / "hparams.yaml")
            model_config.save(cfg_dir / "unet.yaml")
            dataset.config.save(cfg_dir / "cc359.yaml")

            wandb.save(str(Path(__file__)), policy="now")
            wandb.save(str(cfg_dir / "*"), base_path=str(cfg_dir.parent), policy="now")

            run_with_wandb(dataset, hparams, model_config)

        if args.evaluate:
            evaluate_unet(r.id, args.project, table_plot=True)
        if args.cross_eval:
            cross_evaluate_unet(r.id, args.project, table_plot=True)
        if not args.store:
            delete_model_binaries(r.id, args.project)
    else:
        run(dataset, teacher, vae, hparams, model_config)
