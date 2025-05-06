from typing import Any, Dict, Tuple, List

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

    def utility(self) -> float: # mse
        return DPHistogram.utility_func(hist_eps=self._eps, hist_l1_sens=self._hist.l1_sens())

    @staticmethod
    def utility_func(*args, **kwargs): # mse
        # TODO replace L1 sens by diameter over dataset size, adding 2 kwargs
        return 2 * (LaplaceMechanism.noise_scale_func(kwargs["hist_eps"], kwargs["hist_l1_sens"]) ** 2)

    @staticmethod
    def pretty_name() -> str:
        return "histogram"

    def apply(self, x: np.ndarray) -> Any:
        return self._laplace(self._hist.apply(x))

    @staticmethod
    def params() -> List[str]:
        return ["eps", "l1_sens"]

    @staticmethod
    def params_to_labels() -> Dict[str, str]:
        return {
            "eps": "$\\epsilon$",
            "l1_sens": "$\\Delta_1$"
        }

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            "eps": "hist_eps",
            "l1_sens": "hist_l1_sens",
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            "eps": (1e-5,1e1),
            "l1_sens": (0.5, 10),
        }
