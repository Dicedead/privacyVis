import numpy as np

from regions import region_from_dp_params
from sensitivities import L1Sensitivity
from query import Query, DPQuery
from typing import Any, Dict, Tuple, List


class Median(Query, L1Sensitivity):

    def __init__(self, alphabet_diameter: float):
        super().__init__(lambda x: np.median(x))
        self._l1_sens = alphabet_diameter

    def apply(self, x: np.ndarray) -> np.floating[Any]:
        return np.median(x)

    def l1_sens(self) -> float:
        return self._l1_sens


class FiniteAlphabetMedian(Median):
    def __init__(self, alphabet: np.ndarray):
        super().__init__(np.max(alphabet) - np.min(alphabet))


class DPMedian(DPQuery):
    """
    Represent median queries over datasets included with elements in {1, ..., alphabet_size}.
    """

    def __init__(self, eps: float, alphabet_size: float, t: float):

        def score_func(data: np.ndarray) -> np.ndarray:
            return np.exp(-(eps/4) * np.abs(np.sum(np.sign(data-np.arange(1, alphabet_size+1)))))

        self._exp_q = score_func
        self._alphabet_size = alphabet_size
        super().__init__(eps, 0)


    def apply(self, x: np.ndarray) -> Any:
        return np.random.choice(x, 1, p=self._exp_q(x)/self._exp_q(np.arange(1, self._alphabet_size+1)).sum())[0]

    def privacy_region(self, *args, **kwargs):
        return region_from_dp_params(self._eps, 0)

    @staticmethod
    def utility_func(*args, **kwargs):
        return np.minimum(1, kwargs["m"] * np.exp(-kwargs["eps"] * kwargs["t"]/4))

    @staticmethod
    def window_title() -> str:
        return ("DP guarantee of the exponential mechanism for the median mechanism versus a probability bound on how off"
                " the privatized median is from the true median")

    @staticmethod
    def params() -> List[str]:
        return ["eps", "alphabet_size", "t"]

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            "eps": "$\\epsilon$",
            "alphabet_size": "$m$",
            "t": "$t$"
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            "eps": "log(ε)",
            "alphabet_size": "Alphabet size (m)",
            "t": "t"
        }

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            "eps": "eps",
            "alphabet_size": "m",
            "t": "t"
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            "eps": (-3, 1),
            "alphabet_size": (2, 100),
            "t": (0, 250)
        }

    @staticmethod
    def params_are_in_logscale() -> Dict[str, bool]:
        return {
            "eps": True,
            "alphabet_size": False,
            "t": False
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            "eps": False,
            "alphabet_size": True,
            "t": True
        }

    @staticmethod
    def params_change_privacy() -> Dict[str, bool]:
        return {
            "eps": True,
            "alphabet_size": False,
            "t": False
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            "eps": np.log10(0.5),
            "alphabet_size": 30,
            "t": 100
        }

    @staticmethod
    def utility_label() -> str:
        return "Bound on $\\mathbb{P}\\left(\\left|position(M(X),X)-\\frac{n}{2}\\right| \\geq t\\right)$"

    @staticmethod
    def privacy_plot_title() -> str:
        return "$(\\epsilon, 0)-$DP exponential mechanism"
