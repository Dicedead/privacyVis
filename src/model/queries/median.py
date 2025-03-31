import numpy as np

from definitions import ScoreFunction
from exponential_mechanism import ExponentialMechanism
from mechanism import Mechanism
from sensitivities import L1Sensitivity
from query import Query, DPQuery
from typing import Any


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

    def __init__(self, alphabet_size: float, eps: float):

        def score_func(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
            return -np.abs(np.sum(np.sign(data-labels)))

        self._median = Median(alphabet_size)
        self._expmech = ExponentialMechanism(eps, score_func)
        super().__init__(eps, 0)

    def utility(self):
        # TODO finish median
        pass

    def apply(self, x: np.ndarray) -> Any:
        pass