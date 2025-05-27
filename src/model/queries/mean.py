import numpy as np

from laplace_mechanism import LaplaceMechanism
from sensitivities import L1Sensitivity
from query import Query, DPQuery
from typing import Any, Dict, Tuple, List


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

    # TODO implement these for the mean
    def privacy_region(self, *args, **kwargs):
        pass

    @staticmethod
    def pretty_name() -> str:
        pass

    @staticmethod
    def params() -> List[str]:
        pass

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        pass

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        pass

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        pass

    @staticmethod
    def params_to_log() -> Dict[str, bool]:
        pass

    def __init__(self, alphabet: np.ndarray, dataset_size: int, eps: float):
        self._mean = Mean(np.max(alphabet) - np.min(alphabet), dataset_size)
        self._laplace = LaplaceMechanism(eps, self._mean.l1_sens())
        super().__init__(eps, 0)

    @staticmethod
    def utility_func(*args, **kwargs):
        return 2 * (LaplaceMechanism.noise_scale_func(kwargs["mean_eps"], kwargs["mean_l1_sens"]) ** 2)

    def apply(self, x: np.ndarray) -> Any:
        return self._laplace(np.array(self._mean.apply(x)))

