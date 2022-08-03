#!/usr/bin/env python
from pathlib import Path

import ignite.distributed as idist
import wandb
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import WandBLogger
from ignite.engine import Events
from ignite.handlers import EpochOutputStore

from uda import HParams, get_criterion, optimizer_cls, pipe, sigmoid_round_output_transform, to_cpu_output_transform
from uda.datasets import CC359
from uda.models import VAE, VAEConfig
from uda.trainer import VaeTrainer, vae_standard_metrics
from uda_wandb import vae_table_plot


def run(dataset: CC359, hparams: HParams, model_config: VAEConfig):
    dataset.prepare_data()
    dataset.setup()

    model = VAE(model_config).to(idist.device())
    optim = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    loss_fn = get_criterion(hparams.criterion)(**hparams.loss_kwargs)

    trainer = VaeTrainer(
        model=model,
        optim=optim,
        val_loader=dataset.val_dataloader(hparams.val_batch_size),
        loss_fn=loss_fn,
        beta=hparams.vae_beta,
        patience=hparams.early_stopping_patience,
        metrics=vae_standard_metrics(loss_fn),
    )

    ProgressBar().attach(trainer)
    ProgressBar().attach(trainer.val_evaluator)

    trainer.run(dataset.train_dataloader(hparams.train_batch_size), max_epochs=hparams.epochs)


def run_with_wandb(dataset: CC359, hparams: HParams, model_config: VAEConfig):
    dataset.prepare_data()
    dataset.setup()

    model = VAE(model_config).to(idist.device())
    optim = optimizer_cls(hparams.optimizer)(model.parameters(), lr=hparams.learning_rate)
    loss_fn = get_criterion(hparams.criterion)(**hparams.loss_kwargs)

    # log model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.summary.update({"n_parameters": n_params})
    wandb.summary.update({"model_size": "Large" if len(model_config.encoder_blocks) == 6 else "Small"})

    trainer = VaeTrainer(
        model=model,
        optim=optim,
        val_loader=dataset.val_dataloader(hparams.val_batch_size),
        loss_fn=loss_fn,
        beta=hparams.vae_beta,
        patience=hparams.early_stopping_patience,
        metrics=vae_standard_metrics(loss_fn),
        cache_dir=wandb.run.dir,
    )

    ProgressBar().attach(trainer)
    ProgressBar().attach(trainer.val_evaluator)

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
        event_name=Events.COMPLETED,
        handler=vae_table_plot,
        evaluator=trainer.val_evaluator,
        data=dataset.val_split.tensors[0],
        imsize=dataset.imsize,
        patch_size=dataset.patch_size,
    )
    # table evaluation functions needs predictions from validation set
    eos = EpochOutputStore(
        output_transform=pipe(lambda o: (o[1], o[0]), sigmoid_round_output_transform, to_cpu_output_transform)
    )
    eos.attach(trainer.val_evaluator, "output")

    trainer.run(dataset.train_dataloader(hparams.train_batch_size), max_epochs=hparams.epochs)

    wandb_logger.close()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from commons import get_args

    from uda_wandb import delete_model_binaries, evaluate_vae

    args = get_args()

    # load configuration
    hparams = HParams.from_file(args.config / "hparams.yaml")
    model_config = VAEConfig.from_file(args.config / "vae.yaml")
    dataset = CC359.from_preconfigured(args.config / "cc359.yaml", root=args.data)

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
            model_config.save(cfg_dir / "vae.yaml")
            dataset.config.save(cfg_dir / "cc359.yaml")

            wandb.save(str(Path(__file__)), policy="now")
            wandb.save(str(cfg_dir / "*"), base_path=str(cfg_dir.parent), policy="now")

            run_with_wandb(dataset, hparams, model_config)

        if args.evaluate:
            evaluate_vae(r.id, project=args.project, table_plot=True)
        if not args.store:
            delete_model_binaries(r.id, project=args.project)
    else:
        run(dataset, hparams, model_config)
