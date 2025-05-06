import numpy as np

from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Tuple, List


class Query(ABC, Callable[[np.ndarray], Any]):
    @abstractmethod
    def apply(self, x: np.ndarray) -> Any:
        pass

    def __call__(self, x: np.ndarray) -> Any:
        return self.apply(x)


class DPQuery(Query):
    def __init__(self, eps: float, delta: float):
        self._eps = eps
        self._delta = delta

    @abstractmethod
    def utility(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def utility_func(*args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def pretty_name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def params() -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_labels() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_kwargs() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        pass
