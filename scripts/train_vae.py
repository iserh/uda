#!/usr/bin/env python
from pathlib import Path

from evaluation import prediction_image_plot
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore

from uda import HParams, get_loss_cls, get_optimizer_cls
from uda.datasets import UDADataset
from uda.models import VAE, VAEConfig
from uda.trainer import VaeTrainer, get_preds_output_transform, pipe, to_cpu_output_transform, vae_standard_metrics


def run(dataset: UDADataset, hparams: HParams, model_config: VAEConfig, use_wandb: bool = False) -> None:
    if use_wandb:
        import wandb

    dataset.setup()
    train_loader = dataset.train_dataloader(hparams.train_batch_size)
    n_epochs = hparams.max_steps // len(train_loader)

    model = VAE(model_config)
    optim = get_optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    loss_fn = get_loss_cls(hparams.criterion)(**hparams.loss_kwargs)

    trainer = VaeTrainer(
        model=model,
        optim=optim,
        val_loader=dataset.val_dataloader(hparams.val_batch_size),
        loss_fn=loss_fn,
        beta=hparams.vae_beta,
        patience=hparams.early_stopping_patience,
        metrics=vae_standard_metrics(loss_fn, len(dataset.class_labels), hparams.vae_beta),
        cache_dir=wandb.run.dir if use_wandb else "/tmp/models/student",
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)

    if use_wandb:
        # log model size
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.summary.update({"n_parameters": n_params})
        wandb.summary.update({"model_size": "Large" if len(model_config.encoder_blocks) == 6 else "Small"})
        wandb.summary.update({"n_epochs": n_epochs})

        # wandb logger
        wandb_logger = WandBLogger(id=wandb.run.id)
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="training",
            output_transform=lambda o: {"rec_batchloss": o[0], "kl_batchloss": o[1]},
        )
        wandb_logger.attach_output_handler(
            trainer.val_evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="validation",
            metric_names=list(trainer.metrics.keys()),
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        # wandb table evaluation
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED(every=n_epochs // 10),
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

    trainer.run(train_loader, max_epochs=n_epochs)


if __name__ == "__main__":
    from commons import get_launch_config

    launch = get_launch_config()

    # load configuration
    hparams = HParams.from_file(launch.config_dir / "hparams.yaml")
    model_config = VAEConfig.from_file(launch.config_dir / "model.yaml")
    dataset: UDADataset = launch.dataset(launch.config_dir / "dataset.yaml", root=launch.data_root)

    if launch.wandb:
        import wandb
        from evaluation import evaluate, evaluate_vendors

        from uda.trainer import VaeEvaluator
        from wandb_utils import RunConfig, delete_model_binaries, download_dataset

        with wandb.init(
            project=launch.project,
            tags=launch.tags,
            group=launch.group,
            config={
                "hparams": hparams.__dict__,
                "dataset": dataset.config.__dict__,
                "model": model_config.__dict__,
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
            run(dataset, hparams, model_config, use_wandb=True)

            if launch.evaluate:
                evaluate(VaeEvaluator, VAE, dataset, hparams, splits=["validation", "testing"])
            if launch.evaluate_vendors:
                evaluate_vendors(VaeEvaluator, VAE, dataset, hparams, dataset.vendors)

        if not launch.store_model:
            delete_model_binaries(run_cfg)
    else:
        run(dataset, hparams, model_config, use_wandb=False)
