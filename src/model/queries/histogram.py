from typing import Any, Dict, Tuple, List

import numpy as np

from laplace_mechanism import LaplaceMechanism
from sensitivities import L1Sensitivity
from query import Query, DPQuery


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
        self._num_bins = num_bins
        self._laplace = LaplaceMechanism(eps, self._hist.l1_sens())

    @staticmethod
    def utility_func(*args, **kwargs): # mse
        return 2 * kwargs["hist_num_bins"] * (LaplaceMechanism.noise_scale_func(kwargs["hist_eps"], 2) ** 2)

    @staticmethod
    def window_title() -> str:
        return ("Privacy/utility trade-off of the histogram query privatized by the Laplace mechanism versus the mean "
                "squared error metric")

    def apply(self, x: np.ndarray) -> Any:
        return self._laplace(self._hist.apply(x))

    def privacy_region(self, *args, **kwargs):
        return self._laplace.privacy_region()

    @staticmethod
    def params() -> List[str]:
        return ["eps", "num_bins"]

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            "eps": "log(ε)",
            "num_bins": "Number of bins"
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            "eps": "$\\epsilon$",
            "num_bins": "Number of bins"
        }

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            "eps": "hist_eps",
            "num_bins": "hist_num_bins",
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            "eps": (-3, 1),
            "num_bins": (1, 100),
        }

    @staticmethod
    def params_are_in_logscale() -> Dict[str, bool]:
        return {
            "eps": True,
            "num_bins": False
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            "eps": False,
            "num_bins": True
        }

    @staticmethod
    def params_change_privacy() -> Dict[str, bool]:
        return {
            "eps": True,
            "num_bins": False
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            "eps": np.log10(0.32),
            "num_bins": 10
        }

    @staticmethod
    def utility_label() -> str:
        return "Mean squared error"

    @staticmethod
    def privacy_plot_title() -> str:
        return "Laplace mechanism privacy region"

