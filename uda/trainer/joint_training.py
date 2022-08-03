from collections.abc import Callable
from pathlib import Path
from typing import Optional

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss, Metric
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from uda.trainer.base import BaseEvaluator, BaseTrainer, dice_score_fn
from uda.utils import binary_one_hot_output_transform, pipe


class JointEvaluator(BaseEvaluator):
    def __init__(self, model: nn.Module, vae: nn.Module):
        super().__init__(model)
        self.vae = vae.to(idist.device())

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.model.eval()
        self.vae.eval()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)
        y_rec, _, _ = self.vae(x)

        return x, y_true, y_pred, y_rec


class JointTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        optim: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss,
        lambd: float = 1.0,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        metrics: Optional[dict[str, Metric]] = None,
        score_function: Optional[Callable[[Engine], float]] = dice_score_fn,
        cache_dir: Path = Path("/tmp/model-cache"),
    ):
        super(JointTrainer, self).__init__(
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            patience=patience,
            metrics=metrics,
            score_function=score_function,
            cache_dir=cache_dir,
            Evaluator=JointEvaluator,
            evaluator_args=[vae],
        )
        self.schedule = schedule
        self.vae = vae.to(idist.device())
        self.lambd = lambd

        # Evaluation on validation data
        if test_loader is not None:
            self.test_evaluator = JointEvaluator(model, vae)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.test_evaluator.run(test_loader))
            # metrics
            if self.metrics is not None:
                for name, metric in self.metrics.items():
                    metric.attach(self.test_evaluator, name)

    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.optim.zero_grad()
        self.model.train()
        self.vae.eval()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)
        with torch.no_grad():
            y_rec, _, _ = self.vae(x)

        pseudo_loss = self.loss_fn(y_pred, y_true)
        rec_loss = self.loss_fn(y_pred, y_rec)
        loss = pseudo_loss + self.lambd * rec_loss

        loss.backward()
        self.optim.step()
        self.schedule.step()

        return pseudo_loss, rec_loss


def joint_standard_metrics(loss_fn: nn.Module) -> dict[str, Metric]:
    return {
        "pseudo_loss": Loss(loss_fn, output_transform=lambda o: (o[2], o[1])),
        "rec_loss": Loss(loss_fn, output_transform=lambda o: (o[2], o[3])),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=2,
                output_transform=pipe(lambda o: (o[2], o[1]), binary_one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
