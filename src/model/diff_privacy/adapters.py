from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

from definitions import Region, SLIDER_RESOLUTION_NON_INTEGER
from regions import *

from mechanisms import laplace_mechanism, gaussian_mechanism, randomized_response_mechanism


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


def intersected_regions(regions: List[Region], name: str) -> Type[AdaptedRegionComputer]:
    class IntersectedRegions(AdaptedRegionComputer):

        @staticmethod
        def region_computation(*args, **kwargs) -> Region:
            return intersect_regions(regions)

        @staticmethod
        def params() -> List[str]:
            return []

        @staticmethod
        def params_to_kwargs() -> Dict[str, str]:
            return {}

        @staticmethod
        def params_are_integers() -> Dict[str, bool]:
            return {}

        @staticmethod
        def params_are_logscale() -> Dict[str, bool]:
            return {}

        @staticmethod
        def params_to_slider_labels() -> Dict[str, str]:
            return {}

        @staticmethod
        def params_to_graph_labels() -> Dict[str, str]:
            return {}

        @staticmethod
        def params_to_default_vals() -> Dict[str, float]:
            return {}

        @staticmethod
        def params_to_limits() -> Dict[str, Tuple[float, float]]:
            return {}

        @staticmethod
        def adder_label() -> str:
            return {}

        @staticmethod
        def region_graph_name() -> str:
            return name

    return IntersectedRegions

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


class DPSimplifiedCompositionRegion(AdaptedRegionComputer):

    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_composition_simplified(
            [kwargs['eps']] * kwargs['k'], [kwargs['delta']] * kwargs['k'], kwargs['delta_s']
        )

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta', 'k', 'delta_s']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta',
            'k': 'k',
            'delta_s': 'delta_s'
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'eps': False,
            'delta': False,
            'k': True,
            'delta_s': False
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'eps': True,
            'delta': False,
            'k': False,
            'delta_s': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ',
            'k': 'Number of mechanisms (k)',
            'delta_s': 'δ slack'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$',
            'k': '$k$',
            'delta_s': '$\\tilde{\\delta}$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'delta': 0.1,
            'k': 2,
            'delta_s': 3 * SLIDER_RESOLUTION_NON_INTEGER
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'delta': (0.0, 1.0),
            'k': (1, 100),
            'delta_s': (SLIDER_RESOLUTION_NON_INTEGER, 1.0)
        }

    @staticmethod
    def adder_label() -> str:
        return 'DP simplified composition region'

    @staticmethod
    def region_graph_name() -> str:
        return "DP simplified comp."

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

class DPTVRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_tv_params(kwargs['eps'], kwargs['delta'], kwargs['eta'])

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta', 'eta']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta',
            'eta': 'eta'
        }

    @staticmethod
    def params_are_logscale():
        return {
            'eps': True,
            'delta': False,
            'eta': False
        }

    @staticmethod
    def params_are_integers():
        return {
            'eps': False,
            'delta': False,
            'eta': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ',
            'eta': 'Total variation (η)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$',
            'eta': '$\\eta$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'delta': 0.15,
            'eta': 0.25
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'delta': (0.0, 1.0),
            'eta': (0.0, 1.0)
        }

    @staticmethod
    def adder_label() -> str:
        return "DP-TV region"

    @staticmethod
    def region_graph_name() -> str:
        return "DP-TV"


class DPTVCompositionRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_dp_composition_exact_total_var(kwargs['eps'], kwargs['delta'], kwargs['eta'], kwargs['k'])

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'delta', 'eta', 'k']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'delta': 'delta',
            'eta': 'eta',
            'k': 'k'
        }

    @staticmethod
    def params_are_logscale():
        return {
            'eps': True,
            'delta': False,
            'eta': False,
            'k': False
        }

    @staticmethod
    def params_are_integers():
        return {
            'eps': False,
            'delta': False,
            'eta': False,
            'k': True
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'delta': 'δ',
            'eta': 'Total variation (η)',
            'k': 'Number of mechanisms (k)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'delta': '$\\delta$',
            'eta': '$\\eta$',
            'k': '$k$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.3),
            'delta': 0.05,
            'eta': 0.2,
            'k': 4,
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'delta': (0.0, 1.0),
            'eta': (0.0, 1.0),
            'k': (1, 100)
        }

    @staticmethod
    def adder_label() -> str:
        return "DP-TV composition region"

    @staticmethod
    def region_graph_name() -> str:
        return "DP-TV comp."

class GaussianDPRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_gaussian_dp(kwargs['mu'])

    @staticmethod
    def params() -> List[str]:
        return ['mu']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'mu': 'mu'
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'mu': False
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'mu': False
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'mu': 'μ'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'mu': '$\\mu$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'mu': 1.5
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'mu': (0, 6)
        }

    @staticmethod
    def adder_label() -> str:
        return 'Gaussian DP region'

    @staticmethod
    def region_graph_name() -> str:
        return 'Gaussian DP'

class GaussianDPCompositionRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return region_from_gaussian_dp_composition([kwargs['mu']] * kwargs['k'])

    @staticmethod
    def params() -> List[str]:
        return ['mu', 'k']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'mu': 'mu',
            'k': 'k'
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'mu': False,
            'k': False
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'mu': False,
            'k': True
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'mu': 'μ',
            'k': 'Number of mechanisms (k)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'mu': '$\\mu$',
            'k': '$k$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'mu': 1.5,
            'k': 3
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'mu': (0, 6),
            'k': (1, 100)
        }

    @staticmethod
    def adder_label() -> str:
        return 'Gaussian DP composition region'

    @staticmethod
    def region_graph_name() -> str:
        return 'Gaussian DP comp.'

class LaplaceMechanismRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return laplace_mechanism.LaplaceMechanism(kwargs['eps'], 1).privacy_region()

    @staticmethod
    def params() -> List[str]:
        return ['eps']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps'
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'eps': False
        }

    @staticmethod
    def params_are_logscale() -> Dict[str, bool]:
        return {
            'eps': True
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6)
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1)
        }

    @staticmethod
    def adder_label() -> str:
        return "Laplace mechanism"

    @staticmethod
    def region_graph_name() -> str:
        return "Laplace mech."

class GaussianMechanismRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return gaussian_mechanism.GaussianMechanism(kwargs['eps'], kwargs['delta'], 1).privacy_region()

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
            'delta': (SLIDER_RESOLUTION_NON_INTEGER, 1.0)
        }

    @staticmethod
    def adder_label() -> str:
        return "Gaussian mechanism"

    @staticmethod
    def region_graph_name() -> str:
        return "Gaussian mech."

class RandomizedResponseRegion(AdaptedRegionComputer):
    @staticmethod
    def region_computation(*args, **kwargs) -> Region:
        return randomized_response_mechanism.RandomizedResponseMech(kwargs['eps'], kwargs['m']).privacy_region()

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'm']

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'm': 'm'
        }

    @staticmethod
    def params_are_logscale():
        return {
            'eps': True,
            'm': False
        }

    @staticmethod
    def params_are_integers():
        return {
            'eps': False,
            'm': True
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'm': 'Alphabet size (m)'
        }

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'm': '$m$'
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.6),
            'm': 5
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'm': (2, 100)
        }

    @staticmethod
    def adder_label() -> str:
        return "Randomized response mechanism"

    @staticmethod
    def region_graph_name() -> str:
        return "RR mech."
