import numpy as np

from definitions import SLIDER_RESOLUTION_NON_INTEGER
from gaussian_mechanism import GaussianMechanism
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

    def l2_sens(self) -> float:
        return self._l1_sens


class FiniteAlphabetMean(Mean):
    def __init__(self, alphabet: np.ndarray, dataset_size: int):
        super().__init__(np.max(alphabet) - np.min(alphabet), dataset_size)


class DPMean(DPQuery):

    def __init__(self, eps: float, delta: float, dataset_diameter: float, dataset_size: int, dimensions: int):
        self._mean = Mean(dataset_diameter, dataset_size)
        self._gaussian_mech = GaussianMechanism(eps, delta, self._mean.l2_sens())
        super().__init__(eps, 0)

    @staticmethod
    def utility_func(*args, **kwargs):
        return (2 * kwargs["mean_dimensions"] *
                (GaussianMechanism.noise_scale_func(kwargs["mean_eps"],kwargs["mean_delta"],
                 kwargs["mean_dataset_diameter"]/kwargs["mean_dataset_size"])))

    def apply(self, x: np.ndarray) -> Any:
        return self._gaussian_mech(np.array(self._mean.apply(x)))

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            "eps" : "log(ε)",
            "delta" : "δ",
            "dataset_size" : "Dataset size",
            "dataset_diameter" : "Data diameter",
            "dimensions": "Number of dimensions"
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            "eps" : False,
            "delta" : False,
            "dataset_size" : True,
            "dataset_diameter" : False,
            "dimensions": True
        }

    @staticmethod
    def params_change_privacy() -> Dict[str, bool]:
        return {
            "eps": True,
            "delta": True,
            "dataset_size": False,
            "dataset_diameter": False,
            "dimensions": False
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            "eps": 0.3,
            "delta": 0.1,
            "dataset_size": 50,
            "dataset_diameter": 10,
            "dimensions": 1
        }

    @staticmethod
    def utility_label() -> str:
        return "Mean squared error"

    def privacy_region(self, *args, **kwargs):
        return self._gaussian_mech.privacy_region()

    @staticmethod
    def window_title() -> str:
        return ("Privacy/utility trade-off of the mean query privatized by the Gaussian mechanism versus the mean "
                "squared error metric")

    @staticmethod
    def params() -> List[str]:
        return ["eps", "delta", "dataset_diameter", "dataset_size", "dimensions"]

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            "eps": "$\\epsilon$",
            "delta": "$\\delta$",
            "dataset_size": DPMean.params_to_slider_labels()["dataset_size"],
            "dataset_diameter": DPMean.params_to_slider_labels()["dataset_diameter"],
            "dimensions": DPMean.params_to_slider_labels()["dimensions"]
        }

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            "eps": "mean_eps",
            "delta": "mean_delta",
            "dataset_size": "mean_dataset_size",
            "dataset_diameter": "mean_dataset_diameter",
            "dimensions": "mean_dimensions"
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            "eps": (-3, 1),
            "delta": (SLIDER_RESOLUTION_NON_INTEGER, 1),
            "dataset_size": (1, 200),
            "dataset_diameter": (SLIDER_RESOLUTION_NON_INTEGER, 20),
            "dimensions": (1, 20)
        }

    @staticmethod
    def params_are_in_logscale() -> Dict[str, bool]:
        return {
            "eps": True,
            "delta": False,
            "dataset_size": False,
            "dataset_diameter": False,
            "dimensions": False
        }

    @staticmethod
    def privacy_plot_title() -> str:
        return "Gaussian mechanism privacy region"
