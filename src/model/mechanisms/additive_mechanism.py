import numpy as np
from abc import ABC, abstractmethod

from definitions import TradeOffFunction, Region
from differential_privacy import region_from_f_dp, region_from_dp_tv_params


class AdditiveMechanism(ABC):
    """
    Represent differentially private mechanisms of the form M(X) = f(X) + N where N is noise.
    """

    def __init__(self, eps: float, delta: float):
        """
        Store eps and delta differential privacy parameters.

        :param eps: float
        :param delta: float
        """
        self._eps = eps
        self._delta = delta

    @abstractmethod
    def quantile(self, alpha: float | np.ndarray) -> np.ndarray:
        """
        Quantile function of the additive noise N.

        :param alpha: float | np.ndarray
                Point(s) where to compute the quantile function of N.

        :return: np.ndarray
                Computed quantile function at given point(s).
        """
        pass

    @abstractmethod
    def cdf(self, alpha: float | np.ndarray) -> np.ndarray:
        """
        CDF of the additive noise N.

        :param alpha: float | np.ndarray
                Point(s) where to compute the cumulative distribution function of N.

        :return: np.ndarray
                Computed cumulative distribution function at given point(s).
        """
        pass

    @abstractmethod
    def _shift(self) -> float:
        """
        Shift defining the tradeoff-function, of the form:
        T(a) = cdf(quantile(1-a)-shift)

        :return: float
        """
        pass


    @abstractmethod
    def tv(self) -> float:
        """
        Total variation of the mechanism.

        :return: float
        """
        pass

    def tradeoff_function(self) -> TradeOffFunction:
        """
        Define the tradeoff function of the mechanism, which is generally of the form:
        T(a) = cdf(quantile(1-a)-shift), where a is the false positive rate.

        :return: TradeoffFunction
        """
        return lambda fp: self.cdf(self.quantile(1 - fp) - self._shift())

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