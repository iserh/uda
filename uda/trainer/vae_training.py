from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss, Metric
from ignite.utils import convert_tensor, setup_logger
from torch.utils.data import DataLoader

from uda.losses import kl_loss
from uda.models import center_pad_crop
from uda.trainer.base import BaseEvaluator, dice_score_fn
from uda.utils import binary_one_hot_output_transform, pipe


class VaeEvaluator(BaseEvaluator):
    def __init__(self, model: nn.Module):
        super(VaeEvaluator, self).__init__()
        self.model = model.to(idist.device())

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        self.model.eval()

        with torch.no_grad():
            x = convert_tensor(batch[1], idist.device())
            x_rec, mean, v_log = self.model(x)

        x = center_pad_crop(x, x_rec.shape[1:])

        return x_rec, x, mean, v_log


class VaeTrainer(BaseEvaluator):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss,
        beta: float = 1.0,
        val_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        metrics: Optional[dict[str, Metric]] = None,
        score_function: Optional[Callable[[Engine], float]] = dice_score_fn,
        cache_dir: Path = Path("/tmp/models"),
    ):
        super(VaeTrainer, self).__init__()
        self.model = model.to(idist.device())
        self.optim = optim
        self.loss_fn = loss_fn
        self.beta = beta

        # Evaluation on validation data
        if val_loader is not None:
            self.val_evaluator = VaeEvaluator(model)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.val_evaluator.run(val_loader))

        # metrics
        self.metrics = metrics if metrics != {} else None
        if metrics is not None:
            for name, metric in self.metrics.items():
                if val_loader is not None:
                    metric.attach(self.val_evaluator, name)

        # checkpointing
        if metrics is not None and score_function is not None and val_loader is not None:
            model_checkpoint = ModelCheckpoint(
                cache_dir,
                n_saved=1,
                score_function=score_function,
                filename_pattern="best_model.pt",
                require_empty=False,
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

        # early stopping
        if patience is not None and score_function is not None and val_loader is not None:
            self.stopper = EarlyStopping(
                patience=patience,
                score_function=score_function,
                trainer=self,
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, self.stopper)

    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        self.optim.zero_grad()
        self.model.train()

        x = convert_tensor(batch[1], idist.device())
        x_rec, mean, v_log = self.model(x)

        x = center_pad_crop(x, x_rec.shape[1:])

        rec_l = self.loss_fn(x_rec, x)
        kl_l = kl_loss(mean, v_log) * self.beta
        loss = kl_l + rec_l

        loss.backward()
        self.optim.step()

        return rec_l, kl_l


def vae_standard_metrics(loss_fn: nn.Module, beta: float) -> dict[str, Metric]:
    return {
        "rec_loss": Loss(loss_fn, output_transform=lambda o: o[:2]),
        "kl_loss": Loss(lambda *args: kl_loss(*args) * beta, output_transform=lambda o: o[2:]),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=2,
                output_transform=pipe(lambda o: o[:2], binary_one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
