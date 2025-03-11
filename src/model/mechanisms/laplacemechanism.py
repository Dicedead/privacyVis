from additive_mechanism import AdditiveMechanism
from differential_privacy import *


class LaplaceMechanism(AdditiveMechanism):
    """
    Laplace mechanism definition.
    """

    def __init__(self, eps: float):
        """
        Construct the mechanism.

        :param eps: float
                Epsilon parameter of the differentially private mechanism.

        For the Laplace mechanism, delta = 0.
        """
        super().__init__(eps, 0)
        self._mu = 0
        self._scale = 1
        self._eps = eps

    def quantile(self, alpha: float | np.ndarray) -> np.ndarray:
        alpha = np.array(alpha)
        log_arg = 1 - 2 * np.abs(alpha - 0.5)
        ret = self._mu + np.zeros_like(alpha)
        ret[log_arg > 0] -= self._scale * np.sign(alpha[log_arg != 0] - 0.5) * np.log(log_arg[log_arg != 0])
        return ret

    def cdf(self, alpha: float | np.ndarray) -> np.ndarray:
        alpha = np.array(alpha - self._mu)
        return 0.5 + 0.5 * np.sign(alpha) * (1 - np.exp(-np.abs(alpha)/self._scale))

    def _shift(self) -> float:
        return self._eps

    def tv(self) -> float:
        return 1-np.exp(-self._eps/2)
