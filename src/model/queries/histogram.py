import numpy as np

from sensitivities import L1Sensitivity
from query import Query


class Histogram(Query, L1Sensitivity):

    def __init__(self, num_bins: int):
        self._num_bins = num_bins

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.histogram(x, self._num_bins)[0]

    def l1_sens(self) -> float:
        return 2.