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

from ..transforms import CenterPad, center_pad, binarize_prediction
from .base import BaseEvaluator, dice_score_fn
from .output_transforms import one_hot_output_transform, pipe


class JointEvaluator(BaseEvaluator):
    def __init__(self, model: nn.Module, vae: nn.Module, vae_input_size: tuple[int, ...]) -> None:
        super(JointEvaluator, self).__init__()
        self.model = model.to(idist.device())
        self.vae = vae.to(idist.device())
        self.vae_cropping = CenterPad(*vae_input_size)

    @torch.no_grad()
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        self.model.eval()
        self.vae.eval()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)

        y_vae = self.vae_cropping(y_pred)
        y_rec, _, _ = self.vae(y_vae)
        y_rec = binarize_prediction(y_rec)

        y_true = center_pad(y_true, y_pred.shape[2:])
        y_vae = center_pad(y_vae, y_rec.shape[2:])

        x = center_pad(x, y_pred.shape[2:])

        return y_pred, y_true, x, y_vae, y_rec


class JointTrainer(BaseEvaluator):
    def __init__(  # noqa: C901 - too complex
        self,
        model: nn.Module,
        vae: nn.Module,
        vae_input_size: tuple[int, ...],
        optim: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module = nn.BCEWithLogitsLoss,
        lambd: float = 1.0,
        train_loader: Optional[DataLoader] = None,
        pseudo_val_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        metrics: Optional[dict[str, Metric]] = None,
        score_function: Optional[Callable[[Engine], float]] = dice_score_fn,
        cache_dir: str = "/tmp/models",
    ) -> None:
        super(JointTrainer, self).__init__()
        self.model = model.to(idist.device())
        self.vae = vae.to(idist.device())
        self.vae_cropping = CenterPad(*vae.config.input_size)
        self.optim = optim
        self.schedule = schedule
        self.loss_fn = loss_fn
        self.lambd = lambd

        # Evaluation on train data
        if train_loader is not None:
            self.train_evaluator = JointEvaluator(model, vae, vae_input_size)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.train_evaluator.run(train_loader))

        # Evaluation on validation data
        if pseudo_val_loader is not None:
            self.pseudo_val_evaluator = JointEvaluator(model, vae, vae_input_size)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.pseudo_val_evaluator.run(pseudo_val_loader))

        # Evaluation on validation data
        if val_loader is not None:
            self.val_evaluator = JointEvaluator(model, vae, vae_input_size)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.val_evaluator.run(val_loader))

        # Evaluation on testing data
        if test_loader is not None:
            self.test_evaluator = JointEvaluator(model, vae, vae_input_size)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.test_evaluator.run(test_loader))

        # metrics
        self.metrics = metrics if metrics != {} else None
        if metrics is not None:
            for name, metric in self.metrics.items():
                if train_loader is not None:
                    metric.attach(self.train_evaluator, name)
                if pseudo_val_loader is not None:
                    metric.attach(self.pseudo_val_evaluator, name)
                if val_loader is not None:
                    metric.attach(self.val_evaluator, name)
                if test_loader is not None:
                    metric.attach(self.test_evaluator, name)

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
        self.vae.eval()

        x = convert_tensor(batch[0], idist.device())
        y_true = convert_tensor(batch[1], idist.device())
        y_pred = self.model(x)
        with torch.no_grad():
            y_vae = self.vae_cropping(y_pred)
            y_rec, _, _ = self.vae(y_vae)
            y_rec = binarize_prediction(y_rec)

        y_true = center_pad(y_true, y_pred.shape[2:])
        y_vae = center_pad(y_vae, y_rec.shape[2:])

        pseudo_loss = self.loss_fn(y_pred, y_true)
        rec_loss = self.loss_fn(y_vae, y_rec) * self.lambd
        loss = pseudo_loss + rec_loss

        loss.backward()
        self.optim.step()
        self.schedule.step()

        return pseudo_loss, rec_loss


def joint_standard_metrics(loss_fn: nn.Module, num_classes: int, lambd: float) -> dict[str, Metric]:
    return {
        "pseudo_loss": Loss(loss_fn, output_transform=lambda o: o[:2]),
        "rec_loss": Loss(lambda *args: loss_fn(*args) * lambd, output_transform=lambda o: o[3:5]),
        "dice": DiceCoefficient(
            ConfusionMatrix(
                num_classes=num_classes,
                output_transform=pipe(lambda o: o[:2], one_hot_output_transform),
            ),
            ignore_index=0,
        ),
    }
