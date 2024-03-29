from collections.abc import Callable
from typing import Optional

import ignite.distributed as idist
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import ConfusionMatrix, DiceCoefficient, Loss, Metric
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from ..losses import kl_std_div
from ..transforms import center_pad
from .base import BaseEvaluator, dice_score_fn
from .output_transforms import one_hot_output_transform, pipe


class VaeEvaluator(BaseEvaluator):
    def __init__(self, model: nn.Module) -> None:
        super(VaeEvaluator, self).__init__()
        self.model = model.to(idist.device())

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        self.model.eval()

        with torch.no_grad():
            x = convert_tensor(batch[0], idist.device())
            y = convert_tensor(batch[1], idist.device())
            y_rec, mean, v_log = self.model(y)

        x = center_pad(x, y_rec.shape[2:])
        y = center_pad(y, y_rec.shape[2:])

        return y_rec, y, x, mean, v_log


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
        cache_dir: str = "/tmp/models",
    ) -> None:
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

        x = center_pad(x, x_rec.shape[2:])

        rec_l = self.loss_fn(x_rec, x)
        kl_l = kl_std_div(mean, v_log) * self.beta
        loss = kl_l + rec_l

        loss.backward()
        self.optim.step()

        return rec_l, kl_l


def vae_standard_metrics(loss_fn: nn.Module, num_classes: int, beta: float) -> dict[str, Metric]:
    return {
        "rec_loss": Loss(loss_fn, output_transform=lambda o: o[:2]),
        "kl_loss": Loss(lambda *args: kl_std_div(*args) * beta, output_transform=lambda o: o[3:5]),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=num_classes,
                output_transform=pipe(lambda o: o[:2], one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
