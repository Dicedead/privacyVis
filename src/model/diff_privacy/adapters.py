from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Tuple

from definitions import Region
from regions import *


class AdaptedRegionComputer(ABC):

    @staticmethod
    @abstractmethod
    def region_computation(*args, **kwargs) -> Region:
        pass

    @staticmethod
    @abstractmethod
    def region_name(*args, **kwargs) -> str:
        pass

    @staticmethod
    @abstractmethod
    def params() -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_kwargs() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_log() -> Dict[str, bool]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_graph_labels() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_default_vals() -> Dict[str, float]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        pass

    @staticmethod
    @abstractmethod
    def adder_label() -> str:
        pass


class DPRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_params(kwargs['eps'], kwargs['delta'])

    @staticmethod
    def region_name(*args, **kwargs) -> str:
        return f"({kwargs['eps'], kwargs['delta']})-DP"

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta'
        }

    @staticmethod
    def params_to_log():
        return {
            'eps': True,
            'delta': False
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': 'log(epsilon)',
            'delta': 'delta'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'delta': 0.1
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (0.0, 1),
            'delta': (0.0, 1.0)
        }

    @staticmethod
    def adder_label() -> str:
        return "DP region"

