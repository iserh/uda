#!/usr/bin/env python
from pathlib import Path

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore

from uda import HParams, get_loss_cls, get_optimizer_cls
from uda.datasets import UDADataset
from uda.models import UNet, UNetConfig
from uda.trainer import (
    SegTrainer,
    get_preds_output_transform,
    pipe,
    segmentation_standard_metrics,
    to_cpu_output_transform,
)
from uda_wandb.evaluation import prediction_image_plot


def run(dataset: UDADataset, hparams: HParams, model_config: UNetConfig, use_wandb: bool = True) -> None:
    if use_wandb:
        import wandb

    dataset.setup()

    model = UNet(model_config)
    optim = get_optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    loss_fn = get_loss_cls(hparams.criterion)(**hparams.loss_kwargs)

    trainer = SegTrainer(
        model=model,
        optim=optim,
        train_loader=dataset.train_dataloader(hparams.val_batch_size),
        val_loader=dataset.val_dataloader(hparams.val_batch_size),
        loss_fn=loss_fn,
        patience=hparams.early_stopping_patience,
        metrics=segmentation_standard_metrics(loss_fn, model.config.out_channels),
        cache_dir=wandb.run.dir if use_wandb else "/tmp/models/student",
    )

    ProgressBar(desc="Train", persist=True).attach(trainer)
    ProgressBar(desc="Train(Eval)", persist=True).attach(trainer.train_evaluator)
    ProgressBar(desc="Val", persist=True).attach(trainer.val_evaluator)

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
            output_transform=lambda loss: {"batchloss": loss},
        )
        # wandb table evaluation
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=prediction_image_plot,
            evaluator=trainer.val_evaluator,
            dim=model.config.dim,
            dataset=dataset,
            name="validation",
        )
        # table evaluation functions needs predictions from validation set
        eos = EpochOutputStore(
            output_transform=pipe(lambda o: o[:3], get_preds_output_transform, to_cpu_output_transform)
        )
        eos.attach(trainer.val_evaluator, "output")

        for tag, evaluator in [("training", trainer.train_evaluator), ("validation", trainer.val_evaluator)]:
            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=list(trainer.metrics.keys()),
                global_step_transform=lambda *_: trainer.state.iteration,
            )

    trainer.run(dataset.train_dataloader(hparams.train_batch_size), max_epochs=hparams.epochs)


if __name__ == "__main__":
    from commons import get_args

    args = get_args()

    # load configuration
    hparams = HParams.from_file(args.config / "hparams.yaml")
    model_config = UNetConfig.from_file(args.config / "model.yaml")
    dataset: UDADataset = args.dataset(args.config / "dataset.yaml", root=args.data)

    if args.wandb:
        import wandb

        from uda.trainer import SegEvaluator
        from uda_wandb import RunConfig, cross_evaluate_unet, delete_model_binaries, download_dataset, evaluate

        with wandb.init(
            project=args.project,
            tags=args.tags,
            group=args.group,
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

        if args.evaluate:
            evaluate(SegEvaluator, UNet, dataset, hparams, run_cfg, splits=["validation", "testing"])
        if args.cross_eval:
            cross_evaluate_unet(run_cfg, table_plot=True)
        if not args.store:
            delete_model_binaries(run_cfg)
    else:
        run(dataset, hparams, model_config, use_wandb=False)
