#!/usr/bin/env python
from pathlib import Path

from evaluation import prediction_image_plot
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore
from torch.optim.lr_scheduler import LinearLR

from uda import HParams, get_loss_cls, get_optimizer_cls
from uda.datasets import UDADataset
from uda.datasets.dataset_teacher import TeacherData
from uda.models import VAE, UNet
from uda.trainer import JointTrainer, get_preds_output_transform, joint_standard_metrics, pipe, to_cpu_output_transform


def run(teacher: UNet, vae: VAE, dataset: UDADataset, hparams: HParams, use_wandb: bool = False) -> None:
    if use_wandb:
        import wandb

    teacher_data = TeacherData(teacher, dataset)
    teacher_data.setup(hparams.val_batch_size)

    train_loader = teacher_data.train_dataloader(hparams.train_batch_size)
    n_epochs = hparams.max_steps // len(train_loader)

    model = UNet(teacher.config)
    model.load_state_dict(teacher.state_dict())  # copy weights

    optim = get_optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    schedule = LinearLR(optim, 0.01, 1.0, len(train_loader))
    loss_fn = get_loss_cls(hparams.criterion)(**hparams.loss_kwargs)

    trainer = JointTrainer(
        model=model,
        vae=vae,
        vae_input_size=vae.config.input_size,
        optim=optim,
        schedule=schedule,
        loss_fn=loss_fn,
        lambd=hparams.vae_lambd,
        train_loader=teacher_data.train_dataloader(hparams.val_batch_size),
        pseudo_val_loader=teacher_data.val_dataloader(hparams.val_batch_size),  # pseudo labels
        val_loader=dataset.val_dataloader(hparams.val_batch_size),  # real labels
        test_loader=dataset.test_dataloader(hparams.val_batch_size) if dataset.has_split("testing") else None,
        patience=hparams.early_stopping_patience,
        metrics=joint_standard_metrics(loss_fn, len(dataset.class_labels), hparams.vae_lambd),
        cache_dir=wandb.run.dir if use_wandb else "/tmp/models/student",
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Train(Eval)", persist=True).attach(trainer.train_evaluator)
    ProgressBar(desc="Val(Pseudo)", persist=True).attach(trainer.pseudo_val_evaluator)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)
    if hasattr(trainer, "test_evaluator"):
        ProgressBar(desc="Test", persist=True).attach(trainer.test_evaluator)

    if use_wandb:
        # log model size
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.summary.update({"n_parameters": n_params})
        wandb.summary.update({"n_epochs": n_epochs})

        # wandb logger
        wandb_logger = WandBLogger(id=wandb.run.id)
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda o: {"pseudo_batchloss": o[0], "rec_batchloss": o[1]},
        )

        evaluators = [
            ("training", trainer.train_evaluator),
            ("pseudo_validation", trainer.pseudo_val_evaluator),
            ("validation", trainer.val_evaluator),
        ]
        if hasattr(trainer, "test_evaluator"):
            evaluators.append(("testing", trainer.test_evaluator))

        for tag, evaluator in evaluators:
            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=list(trainer.metrics.keys()),
                global_step_transform=lambda *_: trainer.state.iteration,
            )

        # wandb table evaluation
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED(every=max(n_epochs // 10, 1)),
            handler=prediction_image_plot,
            evaluator=trainer.val_evaluator,
            dataset=dataset,
            name="validation",
        )
        # table evaluation functions needs predictions from validation set
        eos = EpochOutputStore(
            output_transform=pipe(lambda o: o[:3], get_preds_output_transform, to_cpu_output_transform)
        )
        eos.attach(trainer.val_evaluator, "output")

    # kick everything off
    trainer.run(train_loader, max_epochs=n_epochs)


if __name__ == "__main__":
    from commons import get_launch_config, get_model_path

    launch = get_launch_config()

    # load configuration
    hparams = HParams.from_file(launch.config_dir / "hparams.yaml")
    dataset: UDADataset = launch.dataset(launch.config_dir / "dataset.yaml", root=launch.data_root)
    # information about teacher / vae
    vae_path = get_model_path(launch.vae, launch.download_model)
    teacher_path = get_model_path(launch.teacher, launch.download_model)
    if launch.wandb:
        from wandb_utils import RunConfig

        vae_run = RunConfig.from_file(vae_path / "run_config.yaml")
        teacher_run = RunConfig.from_file(teacher_path / "run_config.yaml")
        teacher_dataset = launch.dataset(teacher_path / "dataset.yaml")
    # load models
    vae = VAE.from_pretrained(vae_path / "best_model.pt")
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
                    "hparams": hparams.__dict__,
                    "dataset": dataset.config.__dict__,
                    "model": teacher.config.__dict__,
                    "vae": vae.config.__dict__,
                    "teacher_dataset": teacher_dataset.config.__dict__,
                    "teacher_run": teacher_run.__dict__,
                    "vae_run": vae_run.__dict__,
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
                run(teacher, vae, dataset, hparams, use_wandb=True)

                if launch.evaluate:
                    evaluate(SegEvaluator, UNet, dataset, hparams, splits=["validation", "testing"])
                if launch.evaluate_vendors:
                    evaluate_vendors(SegEvaluator, UNet, dataset, hparams, vendors=[dataset.vendor])

            if not launch.store_model:
                delete_model_binaries(run_cfg)
        else:
            run(teacher, vae, dataset, hparams, use_wandb=False)
