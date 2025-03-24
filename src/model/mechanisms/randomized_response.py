import numpy as np

from definitions import Region, TradeOffFunction
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
        self._p_eps = (exp_eps - 1)/(exp_eps + alphabet_size - 1)

    def switch_probability(self) -> float:
        """
        Utility proxy: probability of a random choice.

        :return: float
        """
        return 1 - self._p_eps

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

    def tradeoff_function(self) -> TradeOffFunction:
        # TODO define tradeoff function
        pass

    def tv(self) -> float:
        # TODO define tv or return nan
        pass