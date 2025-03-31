import numpy as np

from laplace_mechanism import LaplaceMechanism
from sensitivities import L1Sensitivity
from query import Query, DPQuery
from typing import Any


class Mean(Query, L1Sensitivity):

    def __init__(self, alphabet_diameter: float, dataset_size: int):
        self._l1_sens = alphabet_diameter/dataset_size

    def apply(self, x: np.ndarray) -> np.floating[Any]:
        return np.mean(x)

    def l1_sens(self) -> float:
        return self._l1_sens


class FiniteAlphabetMean(Mean):
    def __init__(self, alphabet: np.ndarray, dataset_size: int):
        super().__init__(np.max(alphabet) - np.min(alphabet), dataset_size)


class DPMean(DPQuery):
    def __init__(self, alphabet: np.ndarray, dataset_size: int, eps: float):
        self._mean = Mean(np.max(alphabet) - np.min(alphabet), dataset_size)
        self._laplace = LaplaceMechanism(eps, self._mean.l1_sens())
        super().__init__(eps, 0)

    def utility(self):
        return self._laplace.noise_scale()

    def apply(self, x: np.ndarray) -> Any:
        return self._laplace(np.array(self._mean.apply(x)))

