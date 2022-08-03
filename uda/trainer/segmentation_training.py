from collections.abc import Callable
from pathlib import Path
from typing import Optional

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss, Metric
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from uda.trainer.base import BaseEvaluator, BaseTrainer, dice_score_fn
from uda.utils import binary_one_hot_output_transform, pipe


class SegEvaluator(BaseEvaluator):
    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.model.eval()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)

        return x, y_true, y_pred


class SegTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        metrics: Optional[dict[str, Metric]] = None,
        score_function: Optional[Callable[[Engine], float]] = dice_score_fn,
        cache_dir: Path = Path("/tmp/model-cache"),
    ):
        super(SegTrainer, self).__init__(
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            patience=patience,
            metrics=metrics,
            score_function=score_function,
            cache_dir=cache_dir,
            Evaluator=SegEvaluator,
        )

    def step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.optim.zero_grad()
        self.model.train()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)

        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        self.optim.step()

        return loss


def segmentation_standard_metrics(loss_fn: nn.Module) -> dict[str, Metric]:
    return {
        "loss": Loss(loss_fn, output_transform=lambda o: (o[2], o[1])),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=2,
                output_transform=pipe(lambda o: (o[2], o[1]), binary_one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
