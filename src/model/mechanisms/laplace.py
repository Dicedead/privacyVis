from additive_mechanism import AdditiveMechanism
from differential_privacy import *


class Laplace(AdditiveMechanism):

    def __init__(self, l1_sens: float, eps: float):
        super().__init__(eps, 0)
        self._mu = 0
        self._scale = l1_sens / eps
        self._sens = l1_sens

    def quantile(self, alpha: float | np.ndarray):
        alpha = np.array(alpha)
        return self._mu - self._scale * np.sign(alpha - 0.5) * np.minimum(0, np.log(1 - 2 * np.abs(alpha - 0.5)))

    def cdf(self, alpha: float | np.ndarray):
        alpha = np.array(alpha - self._mu)
        return 0.5 + 0.5 * np.sign(alpha) * (1 - np.exp(-np.abs(alpha)/self._scale))

    def tradeoff_function(self) -> TradeOffFunction:
        return lambda fp: self.cdf(self.quantile(1 - fp) - self._sens)

    def tv(self):
        return 1-np.exp(-self._eps/2)
