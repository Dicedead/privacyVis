from abc import ABC
from typing import Callable, Any

import numpy as np


class Query(ABC, Callable[[np.ndarray], Any]):
    def __init__(self, f: Callable[[np.ndarray], Any]):
        self._func = f

    def __call__(self, x: np.ndarray) -> Any:
        return self._func(x)

