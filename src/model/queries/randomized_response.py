from typing import Any, Dict, Tuple, List

import numpy as np

from definitions import TradeOffFunction
from query import DPQuery
from regions import tradeoff_eps_delta_dp_total_var
from mechanism import Mechanism
from mechanisms.randomized_response_mechanism import RandomizedResponseMech


class RandomizedResponse(DPQuery):
    def __init__(self, eps: float, alphabet_size: int):
        super().__init__(eps, 0)
        self._rr = RandomizedResponseMech(eps, alphabet_size)

    def apply(self, x: np.ndarray) -> Any:
        return self._rr(x)

    def privacy_region(self, *args, **kwargs):
        return self._rr.privacy_region()

    @staticmethod
    def utility_func(*args, **kwargs):
        return RandomizedResponseMech.compute_randomized_response_proba(kwargs['eps'], kwargs['alphabet_size'])

    @staticmethod
    def window_title() -> str:
        return "DP region of randomized response versus the probability of a uniformly random output"

    @staticmethod
    def params() -> List[str]:
        return ['eps', 'alphabet_size']

    @staticmethod
    def params_to_graph_labels() -> Dict[str, str]:
        return {
            'eps': '$\\epsilon$',
            'alphabet_size': '$m$'
        }

    @staticmethod
    def params_to_slider_labels() -> Dict[str, str]:
        return {
            'eps': 'log(ε)',
            'alphabet_size': 'Alphabet size (m)'

        }

    @staticmethod
    def params_to_kwargs() -> Dict[str, str]:
        return {
            'eps': 'eps',
            'alphabet_size': 'alphabet_size'
        }

    @staticmethod
    def params_to_limits() -> Dict[str, Tuple[float, float]]:
        return {
            'eps': (-3, 1),
            'alphabet_size': (2, 100)
        }

    @staticmethod
    def params_are_in_logscale() -> Dict[str, bool]:
        return {
            'eps': True,
            'alphabet_size': False
        }

    @staticmethod
    def params_are_integers() -> Dict[str, bool]:
        return {
            'eps': False,
            'alphabet_size': True
        }

    @staticmethod
    def params_change_privacy() -> Dict[str, bool]:
        return {
            'eps': True,
            'alphabet_size': True
        }

    @staticmethod
    def params_to_default_vals() -> Dict[str, float]:
        return {
            'eps': np.log10(0.5),
            'alphabet_size': 5
        }

    @staticmethod
    def utility_label() -> str:
        return 'Probability of a uniform choice'

    @staticmethod
    def privacy_plot_title() -> str:
        return "Randomized response privacy region"