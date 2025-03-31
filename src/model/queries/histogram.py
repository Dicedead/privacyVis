from typing import Any

import numpy as np

from laplace_mechanism import LaplaceMechanism
from sensitivities import L1Sensitivity
from query import Query,DPQuery


class Histogram(Query, L1Sensitivity):
    """
    Define  mechanism the histogram query.
    """

    def __init__(self, num_bins: int):
        """
        Construct a histogram query with a given number of bins.

        :param num_bins: int
        """
        self._num_bins = num_bins

    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the histogram of given data.

        :param x: np.ndarray
                Given data.

        :return: np.ndarray
                Histogram of the given data.
        """
        return np.histogram(x, self._num_bins)[0]

    def l1_sens(self) -> float:
        return 2.

class DPHistogram(DPQuery):
    def __init__(self, eps: float, num_bins: int):
        super().__init__(eps, 0)
        self._hist = Histogram(num_bins)
        self._laplace = LaplaceMechanism(eps, self._hist.l1_sens())

    def utility(self) -> float:
        """
        Return Laplacian noise scale.

        :return: float
        """
        return self._laplace.noise_scale()

    def apply(self, x: np.ndarray) -> Any:
        return self._laplace(self._hist.apply(x))
