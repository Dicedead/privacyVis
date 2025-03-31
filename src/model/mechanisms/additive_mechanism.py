import numpy as np
from abc import abstractmethod

from definitions import TradeOffFunction, Region
from mechanism import Mechanism
from typing import Any


class AdditiveMechanism(Mechanism):
    """
    Represent differentially private mechanisms of the form M(X) = f(X) + N where N is noise.
    """

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
    def noise_scale(self) -> float:
        """
        Notion of noise scale (variance or other measure if appropriate).

        :return: float
        """
        pass

    @abstractmethod
    def generate_noise(self, size) -> np.ndarray:
        """
        Generate random noise according to the prescribed distribution.

        :param size: shape of random variables to generate

        :return: np.ndarray
                Random samples from the  prescribed distribution.
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

    def tradeoff_function(self) -> TradeOffFunction:
        """
        Define the tradeoff function of the mechanism, which is generally of the form:
        T(a) = cdf(quantile(1-a)-shift), where a is the false positive rate.

        :return: TradeoffFunction
        """
        return lambda fp: self.cdf(self.quantile(1 - fp) - self._shift())

    def apply(self, x: np.ndarray, *args, **kwargs) -> Any:
        """
        Apply additive noise.

        :param x: np.ndarray
                Array to privatize.

        :param args: unused

        :param kwargs: unused

        :return: Result.
        """
        return x + self.generate_noise(x.shape)

