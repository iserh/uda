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

from uda.losses import kl_loss
from uda.trainer.base import BaseEvaluator, BaseTrainer, dice_score_fn
from uda.utils import binary_one_hot_output_transform, pipe


class VaeEvaluator(BaseEvaluator):
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        self.model.eval()

        with torch.no_grad():
            x = convert_tensor(batch[0], idist.device())
            x_rec, mean, v_log = self.model(x)

        return x, x_rec, mean, v_log


class VaeTrainer(BaseTrainer):
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
        cache_dir: Path = Path("/tmp/model-cache"),
    ):
        super(VaeTrainer, self).__init__(
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            train_loader=None,
            val_loader=val_loader,
            patience=patience,
            metrics=metrics,
            score_function=score_function,
            cache_dir=cache_dir,
            Evaluator=VaeEvaluator,
        )
        self.beta = beta

    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        self.optim.zero_grad()
        self.model.train()

        x = convert_tensor(batch[0], idist.device())
        x_rec, mean, v_log = self.model(x)

        rec_l = self.loss_fn(x_rec, x)
        kl_l = kl_loss(mean, v_log)
        loss = self.beta * kl_l + rec_l

        loss.backward()
        self.optim.step()

        return rec_l, kl_l


def vae_standard_metrics(loss_fn: nn.Module) -> dict[str, Metric]:
    return {
        "rec_loss": Loss(loss_fn, output_transform=lambda o: (o[1], o[0])),
        "kl_loss": Loss(kl_loss, output_transform=lambda o: o[2:]),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=2,
                output_transform=pipe(lambda o: (o[1], o[0]), binary_one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
