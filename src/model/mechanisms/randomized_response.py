from typing import Any

import numpy as np

from definitions import TradeOffFunction
from differential_privacy import tradeoff_eps_delta_dp_total_var
from mechanism import Mechanism


class RandomizedResponse(Mechanism):
    """
    Define the privacy region and the utility proxy of randomized response.
    """

    def __init__(self, eps: float, alphabet_size: int):
        """
        Instantiate the randomized response parameters.

        :param eps: float
                Desired epsilon parameter for (eps, delta) differential privacy.

        :param alphabet_size: int
                Size of randomized input alphabet.
        """
        super().__init__(eps, 0)
        exp_eps = np.exp(eps)
        self._denom = 1/(exp_eps + alphabet_size - 1)
        self._alphabet_size = alphabet_size
        self._p_eps = (exp_eps - 1) * self._denom
        self._total_var = self._p_eps

    def switch_probability(self) -> float:
        """
        Utility proxy: probability of a random choice.

        :return: float
        """
        return 1 - self._p_eps

    @staticmethod
    def compute_randomized_response_proba(eps: float, alphabet_size: int) -> float:
        # TODO
        pass

    @staticmethod
    def compute_randomized_response_epsilon(p: float, alphabet_size: int) -> float:
        """
        Given a random choice probability p and an alphabet size, compute the corresponding epsilon parameter
        for (eps, 0) differential privacy.

        :param p: float

        :param alphabet_size: int

        :return: float
        """
        p = 1-p
        return np.log((p + (1-p)/alphabet_size) / ((1-p)/alphabet_size))

    def apply(self, x: np.ndarray, *args, **kwargs) -> Any:
        y = np.copy(x)

        for i in range(len(x)):
            if np.random.random() > self._p_eps:
                y[i] = np.random.uniform(low=1, high=self._alphabet_size+1)

        return y

    def tradeoff_function(self) -> TradeOffFunction:
        return tradeoff_eps_delta_dp_total_var(self._eps, 0, self._total_var)

    def tv(self) -> float:
        return self._total_var