#!/usr/bin/env python
from pathlib import Path

import ignite.distributed as idist
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore
from torch.optim.lr_scheduler import LinearLR

from uda import HParams, get_criterion, optimizer_cls, pipe, sigmoid_round_output_transform, to_cpu_output_transform
from uda.datasets import CC359
from uda.datasets.dataset_teacher import TeacherData
from uda.models import UNet, UNetConfig, VAE
from uda.trainer import JointTrainer, joint_standard_metrics
from uda_wandb import segmentation_table_plot


def run(teacher: UNet, vae: VAE, dataset: CC359, hparams: HParams, use_wandb: bool = False) -> None:
    if use_wandb: import wandb

    teacher_data = TeacherData(teacher, dataset)
    teacher_data.setup(hparams.val_batch_size)

    train_loader = teacher_data.train_dataloader(hparams.val_batch_size)
    val_loader = teacher_data.val_dataloader(hparams.val_batch_size)  # pseudo labels
    true_val_loader = dataset.val_dataloader(hparams.val_batch_size)  # real labels

    model = UNet(teacher.config).to(idist.device())
    model.load_state_dict(teacher.state_dict())  # copy weights

    optim = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    schedule = LinearLR(optim, 0.01, 1.0, len(train_loader))
    loss_fn = get_criterion(hparams.criterion)(**hparams.loss_kwargs)

    trainer = JointTrainer(
        model=model,
        vae=vae,
        optim=optim,
        schedule=schedule,
        loss_fn=loss_fn,
        lambd=hparams.vae_lamdb,
        train_loader=train_loader,
        val_loader=val_loader,  # pseudo labels
        true_val_loader=true_val_loader,  # real labels
        patience=hparams.early_stopping_patience,
        metrics=joint_standard_metrics(loss_fn),
        cache_dir=wandb.run.dir if use_wandb else "/tmp/models/student"
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Train(Eval)", persist=True).attach(trainer.train_evaluator)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)
    ProgressBar(desc="Val(True)", persist=True).attach(trainer.true_val_evaluator)

    if use_wandb:
        # log model size
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.summary.update({"n_parameters": n_params})

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
            evaluator=trainer.true_val_evaluator,
            imsize=dataset.imsize,
            patch_size=dataset.patch_size,
        )
        # table evaluation functions needs predictions from validation set
        eos = EpochOutputStore(
            output_transform=pipe(lambda o: (*o[:2], o[4]), sigmoid_round_output_transform, to_cpu_output_transform)
        )
        eos.attach(trainer.true_val_evaluator, "output")

        for tag, evaluator in [
            ("training", trainer.train_evaluator),
            ("validation", trainer.val_evaluator),
            ("true_validation", trainer.true_val_evaluator),
        ]:
            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=list(trainer.metrics.keys()),
                global_step_transform=lambda *_: trainer.state.iteration,
            )

    # kick everything off
    trainer.run(dataset.train_dataloader(hparams.train_batch_size), max_epochs=hparams.epochs)


if __name__ == "__main__":
    from commons import get_args
    from uda.datasets import CC359Config
    from uda_wandb import cross_evaluate_unet, delete_model_binaries, download_dataset, evaluate_unet, RunConfig

    args = get_args()

    # load configuration
    hparams = HParams.from_file(args.config / "hparams.yaml")
    model_config = UNetConfig.from_file(args.config / "model.yaml")
    dataset = CC359.from_preconfigured(args.config / "dataset.yaml", root=args.data)

    vae = VAE.from_pretrained(args.vae_path / "best_model.pt")
    vae_run = RunConfig.from_file(args.vae_path / "run_config.yaml")
    teacher = UNet.from_pretrained(args.teacher_path / "best_model.pt")
    teacher_ds_cfg = CC359Config.from_file(args.teacher_path / "dataset.yaml")
    teacher_run = RunConfig.from_file(args.teacher_path / "run_config.yaml")

    if args.wandb:
        import wandb

        with wandb.init(
            project=args.project,
            tags=args.tags,
            group=args.group,
            config={
                "hparams": hparams.__dict__,
                "dataset": dataset.config.__dict__,
                "model": model_config.__dict__,
                "teacher": teacher.config.__dict__,
                "vae": vae.config.__dict__,
                "teacher_dataset": teacher_ds_cfg.__dict__,
                "teacher_run": teacher_run.__dict__,
                "vae_run": vae_run.__dict__,
            },
        ) as r:
            run_cfg = RunConfig(r.id, r.project)
            r.log_code()
            # save the configuration to the wandb run dir
            cfg_dir = Path(r.dir) / "config"
            cfg_dir.mkdir()
            hparams.save(cfg_dir / "hparams.yaml")
            model_config.save(cfg_dir / "model.yaml")
            dataset.config.save(cfg_dir / "dataset.yaml")

            download_dataset(dataset)
            run(teacher, vae, dataset, hparams, use_wandb=True)

        if args.evaluate:
            evaluate_unet(run_cfg, table_plot=True)
        if args.cross_eval:
            cross_evaluate_unet(run_cfg, table_plot=True)
        if not args.store:
            delete_model_binaries(run_cfg)
    else:
        run(teacher, vae, dataset, hparams, use_wandb=False)
