import numpy as np
from abc import ABC, abstractmethod

from differential_privacy import region_from_f_dp, region_from_dp_tv_params


class AdditiveMechanism(ABC):
    def __init__(self, eps: float, delta: float):
        self._eps = eps
        self._delta = delta

    @abstractmethod
    def quantile(self, alpha: float | np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cdf(self, alpha: float | np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def tradeoff_function(self):
        pass

    @abstractmethod
    def tv(self):
        pass

    def region_exact(self):
        return region_from_f_dp(self.tradeoff_function())

    def region_tv(self):
        return region_from_dp_tv_params(self._eps, self._delta, self.tv())