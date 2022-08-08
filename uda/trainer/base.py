from abc import ABC, abstractmethod

import torch
from ignite.engine import Engine
from ignite.utils import setup_logger


class BaseEvaluator(Engine, ABC):
    def __init__(self) -> None:
        super(BaseEvaluator, self).__init__(type(self).step)
        self.logger = setup_logger(self.__class__.__name__)

    @abstractmethod
    def step(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        ...


def dice_score_fn(engine: Engine) -> float:
    return engine.state.metrics["dice"][0].item()
