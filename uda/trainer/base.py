from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Metric
from ignite.utils import setup_logger
from torch.utils.data import DataLoader


class BaseEvaluator(Engine, ABC):
    def __init__(self):
        super(BaseEvaluator, self).__init__(type(self).step)
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        ...


def dice_score_fn(engine: Engine) -> float:
    return engine.state.metrics["dice"][0].item()
