from additive_mechanism import AdditiveMechanism
from differential_privacy import *


class GaussianMechanism(AdditiveMechanism):
    """
    Gaussian mechanism definition.
    """

    def __init__(self, eps: float, delta: float, l2_sens: float):
        """
        Construct the mechanism.

        :param eps: float
                Epsilon parameter of the differentially private mechanism.

        :param delta: float
                Delta parameter of the differentially private mechanism.
        """
        super().__init__(eps, delta)
        self._mu = 0
        self._sigma = 1
        self._shiftval = eps/np.sqrt(2 * np.log(5/(4 * delta)))
        self._l2_sens = l2_sens

    def quantile(self, alpha: float | np.ndarray) -> np.ndarray:
        return self._sigma * stats.norm.ppf(alpha) + self._mu

    def cdf(self, alpha: float | np.ndarray) -> np.ndarray:
        return stats.norm.cdf((alpha - self._mu) / self._sigma)

    def _shift(self) -> float:
        return self._shiftval

    def tv(self):
        return 2 * stats.norm.cdf(self._mu / 2) - 1

    def noise_scale(self) -> float:
        return np.sqrt(2 * np.log(5/(4 * self._delta)) * ((self._l2_sens / self._eps) ** 2))

    def generate_noise(self, size) -> np.ndarray:
        return np.random.normal(loc=0, scale=self.noise_scale(), size=size)
