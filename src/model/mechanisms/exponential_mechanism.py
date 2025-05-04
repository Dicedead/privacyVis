from typing import Any

import numpy as np

from definitions import TradeOffFunction, ScoreFunction
from mechanism import Mechanism


class ExponentialMechanism(Mechanism):

    def __init__(self, eps: float, score: ScoreFunction):
        super().__init__(eps, 0)
        self._score = score

    def tradeoff_function(self) -> TradeOffFunction:
        pass

    def tv(self) -> float:
        pass

    def apply(self, x: np.ndarray, *args, **kwargs) -> Any:
        pass