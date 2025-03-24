from abc import ABC, abstractmethod

from definitions import TradeOffFunction, Region
from differential_privacy import region_from_f_dp, region_from_dp_tv_params


class Mechanism(ABC):
    def __init__(self, eps: float, delta: float):
        """
        Store eps and delta differential privacy parameters.

        :param eps: float
        :param delta: float
        """
        self._eps = eps
        self._delta = delta

    @abstractmethod
    def tradeoff_function(self) -> TradeOffFunction:
        pass

    @abstractmethod
    def tv(self) -> float:
        """
        Total variation of the mechanism.

        :return: float
        """
        pass

    def region_exact(self) -> Region:
        """
        Define the tradeoff-function induced (exact) region.
        :return: Region
        """
        return region_from_f_dp(self.tradeoff_function())

    def region_tv(self) -> Region:
        """
        Define the total variation induced region.
        :return: Region
        """
        return region_from_dp_tv_params(self._eps, self._delta, self.tv())