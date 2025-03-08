import numpy as np

from differential_privacy import *


def laplace_quantile(alpha: float | np.ndarray, mu: float, scale: float):
    alpha = np.array(alpha)
    return mu - scale * np.sign(alpha - 0.5) * np.minimum(0, np.log(1 - 2 * np.abs(alpha - 0.5)))

def laplace_cdf(alpha: float | np.ndarray, mu: float, scale: float):
    alpha = np.array(alpha - mu)
    return 0.5 + 0.5 * np.sign(alpha) * (1 - np.exp(-np.abs(alpha)/scale))

def region_laplace_mechanism_exact(sens: float, eps: float):
    return region_from_f_dp(lambda fp: laplace_cdf(laplace_quantile(1 - fp, 0, sens / eps) - sens, 0, sens / eps))

def region_laplace_mechanism_tv(sens: float, eps: float):
    return region_from_dp_tv_params(eps, 0, 1-np.exp(-eps/2))
