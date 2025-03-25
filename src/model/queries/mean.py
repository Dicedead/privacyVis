import numpy as np

from sensitivities import L1Sensitivity
from query import Query
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
