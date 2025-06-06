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
    def params() -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_kwargs() -> Dict[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def params_are_integers() -> Dict[str, bool]:
        pass

    @staticmethod
    @abstractmethod
    def params_are_logscale() -> Dict[str, bool]:
        pass

    @staticmethod
    @abstractmethod
    def params_to_slider_labels() -> Dict[str, str]:
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

    @staticmethod
    @abstractmethod
    def region_graph_name() -> str:
        pass


class DPRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_params(kwargs['eps'], kwargs['delta'])

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
    def params_are_logscale():
        return {
            'eps': True,
            'delta': False
        }

    @staticmethod
    def params_are_integers():
        return {
            'eps': False,
            'delta': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$'
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
            'eps': (-3, 1),
            'delta': (0.0, 1.0)
        }

    @staticmethod
    def adder_label() -> str:
        return "DP region"

    @staticmethod
    def region_graph_name() -> str:
        return "DP"


class DPExactCompositionRegion(AdaptedRegionComputer):

    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_composition_exact(kwargs['eps'], kwargs['delta'], kwargs['k'])

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta', 'k']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta',
            'k': 'k'
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'eps': False,
            'delta': False,
            'k': True
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'eps': True,
            'delta': False,
            'k': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ',
            'k': 'Number of mechanisms (k)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$',
            'k': '$k$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'delta': 0.1,
            'k': 2,
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'delta': (0.0, 1.0),
            'k': (1, 100)
        }

    @staticmethod
    def adder_label() -> str:
        return 'DP exact composition region'

    @staticmethod
    def region_graph_name() -> str:
        return "DP exact comp."

class DPBasicCompositionRegion(AdaptedRegionComputer):

    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_composition_basic(kwargs['eps'], kwargs['delta'], kwargs['k'])

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta', 'k']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta',
            'k': 'k'
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'eps': False,
            'delta': False,
            'k': True
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'eps': True,
            'delta': False,
            'k': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ',
            'k': 'Number of mechanisms (k)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$',
            'k': '$k$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'delta': 0.1,
            'k': 2,
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'delta': (0.0, 1.0),
            'k': (1, 100)
        }

    @staticmethod
    def adder_label() -> str:
        return 'DP basic composition region'

    @staticmethod
    def region_graph_name() -> str:
        return "DP basic comp."