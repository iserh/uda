from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Metric
from ignite.utils import setup_logger
from torch.utils.data import DataLoader


class BaseEvaluator(Engine, ABC):
    def __init__(self, model: nn.Module):
        super(BaseEvaluator, self).__init__(type(self).step)
        self.logger = setup_logger(self.__class__.__name__)
        self.model = model

    @abstractmethod
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        ...


class BaseTrainer(Engine, ABC):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
        metrics: Optional[dict[str, Metric]] = None,
        score_function: Optional[Callable[[Engine], float]] = None,
        cache_dir: Path = Path("/tmp/model-cache"),
        Evaluator: type[BaseEvaluator] = BaseEvaluator,
    ):
        super(BaseTrainer, self).__init__(type(self).step)
        self.logger = setup_logger(self.__class__.__name__)
        self.loss_fn = loss_fn
        self.model = model
        self.optim = optim

        # Evaluation on train data
        if train_loader is not None:
            self.train_evaluator = Evaluator(model)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.train_evaluator.run(train_loader))

        # Evaluation on validation data
        if val_loader is not None:
            self.val_evaluator = Evaluator(model)
            self.add_event_handler(Events.EPOCH_COMPLETED, lambda: self.val_evaluator.run(val_loader))

        # metrics
        self.metrics = metrics if metrics != {} else None
        if metrics is not None:
            for name, metric in self.metrics.items():
                if train_loader is not None:
                    metric.attach(self.train_evaluator, name)
                if val_loader is not None:
                    metric.attach(self.val_evaluator, name)

        # checkpointing
        if metrics is not None and score_function is not None:
            model_checkpoint = ModelCheckpoint(
                cache_dir,
                n_saved=1,
                score_function=score_function,
                filename_pattern="best_model.pt",
                require_empty=False,
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

        # early stopping
        if patience is not None:
            self.stopper = EarlyStopping(
                patience=patience,
                score_function=score_function,
                trainer=self,
            )
            self.val_evaluator.add_event_handler(Events.COMPLETED, self.stopper)

    @abstractmethod
    def step(self, batch: tuple[torch.Tensor, ...]) -> Any:
        ...


def dice_score_fn(engine: Engine) -> float:
    return engine.state.metrics["dice"][0].item()
