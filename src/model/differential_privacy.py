from math import ceil

import scipy.special as sps
import scipy.stats as stats
import numpy as np

from definitions import Region, TradeOffFunction
from typing import List


def intersect_regions(regions: List[Region]) -> Region:
    """
    Intersect regions and output the intersection.

    :param regions: List[Region]
            List of regions.

    :return: Region
            Intersection of the given regions.
    """
    ret = []
    for region in regions:
        ret.extend(region)
    return ret


def region_from_dp_params(eps: float, delta: float) -> Region:
    """
    Define the privacy region corresponding to (eps, delta)-differential privacy.

    :param eps: float
            Epsilon parameter of the differential privacy region.

    :param delta: float
            Delta parameter of the differential privacy region.

    :return: Region
            List of constraints defining the privacy region.
    """
    assert eps >= 0
    assert 0 <= delta <= 1

    exp = np.exp(eps)
    ineq = lambda fp, fn: fp + exp * fn >= 1 - delta
    reverse_ineq = lambda fp, fn: ineq(fn, fp)
    sum_line = lambda fp, fn: fp + fn <= 1

    return [ineq, reverse_ineq, sum_line]

def region_from_dp_tv_params(eps: float, delta: float, eta: float):
    tv_constraint = lambda fp, fn: fp + fn >= 1 - eta
    return intersect_regions([region_from_dp_params(eps, delta), [tv_constraint]])

def region_from_dp_composition_basic(eps: float, delta: float, k: int) -> Region:
    """
    Compute the differential privacy basic composition region corresponding to the basic result for the composition
    of differentially private mechanisms.

    :param eps: float
            Epsilon parameter of the differentially private mechanisms being composed.

    :param delta: float
            Delta parameter of the differentially private mechanisms being composed.

    :param k: int
            Number of composed mechanisms.

    :return: Region
            List of constraints defining the privacy region.
    """
    assert eps >= 0
    assert 0 <= delta <= 1
    assert k >= 0

    return region_from_dp_params(k * eps, min(1., k * delta))


def region_from_dp_composition_exact(eps: float, delta: float, k: int) -> Region:
    """
    Compute the differential privacy composition region corresponding to the improved result for the composition
    of differentially private mechanisms.

    :param eps: float
            Epsilon parameter of the differentially private mechanisms being composed.

    :param delta: float
            Delta parameter of the differentially private mechanisms being composed.

    :param k: int
            Number of composed mechanisms.

    :return: Region
            List of constraints defining the privacy region.
    """
    assert eps >= 0
    assert 0 <= delta <= 1
    assert k >= 0

    constraints = []
    for i in range(np.floor(k/2)+1):
        eps_prime = (k - 2 * i) * eps
        delta_tmp = sum([sps.comb(k, l) * (np.exp((k-l) * eps) - np.exp((k-2*i+l) * eps)) for l in range(i)])
        delta_tmp /= (1+np.exp(eps)) ** k
        delta_prime = 1 - ((1 - delta) ** k) * (1 - delta_tmp)
        constraints.append(region_from_dp_params(eps_prime, delta_prime))

    return intersect_regions(constraints)

def region_from_dp_composition_simplified(
        eps_ls: List[float] | np.ndarray,
        delta_ls: List[float] | np.ndarray,
        delta_slack: float
) -> Region:
    """
    Compute the differential privacy composition region corresponding to the improved then simplified result for the composition
    of differentially private mechanisms.

    :param eps_ls: List[float] | np.ndarray
            Epsilon parameters of the differentially private mechanisms being composed.

    :param delta_ls: List[float] | np.ndarray
            Delta parameters of the differentially private mechanisms being composed.

    :param delta_slack: float
            Additional delta slackness.

    :return: Region
            List of constraints defining the privacy region.
    """
    sum_eps = np.array(eps_ls).sum()
    sum_eps_sq = (np.array(eps_ls) ** 2).sum()
    exp_sum = np.sum(((np.exp(eps_ls) - 1) * eps_ls)/(np.exp(eps_ls) + 1))
    delta_ls = np.array(delta_ls)

    delta = 1 - (1 - delta_slack) * np.prod(1 - delta_ls)

    eps_opt1 = sum_eps
    eps_opt2 = exp_sum + np.sqrt(-2 * np.log(delta_slack) * sum_eps_sq)
    eps_opt3 = exp_sum + np.sqrt(2 * np.log(np.e + np.sqrt(sum_eps_sq) / delta_slack) * sum_eps_sq)

    return region_from_dp_params(min(eps_opt1, eps_opt2, eps_opt3), delta)

def region_from_dp_composition_exact_total_var(
        eps: float,
        delta: float,
        eta: float,
        k: int,
        return_d_tv: bool = False
):
    """
    Compute the differential privacy composition region corresponding to the improved result for the composition
    of differentially private mechanisms, accounting for the total variation of the considered mechanisms.

    :param eps: float
            Epsilon parameter of the differentially private mechanisms being composed.

    :param delta: float
            Delta parameter of the differentially private mechanisms being composed.

    :param eta: float
            Total variation of the considered mechanisms.

    :param k: int
            Number of composed mechanisms.

    :param return_d_tv: bool
            If True, return as a second output an upper bound on the total variation of the composed mechanism.
            Defaults to False.

    :return: Region
            List of constraints defining the privacy region.
    """
    assert eps >= 0
    assert 0 <= delta <= 1
    assert k >= 0
    assert delta <= eta <= delta + ((np.exp(eps) - 1) * (1 - delta)) / (np.exp(eps) + 1)

    alpha = 1 - (eta - delta) * (1 + np.exp(eps)) / ((1 - delta) * (np.exp(eps) - 1))

    constraints = []
    d_tv = 1
    for j in range(k+1):
        eps_prime = j * eps
        delta_tmp = sum(
            [ sps.comb(k, a) * sum(
                [
                    sps.comb(k-a, l) * (((1-alpha)/(1+np.exp(eps))) ** (k-a)) * \
                    (alpha ** a) * (np.exp((k-l-a)*eps) - np.exp((l+j)*eps))
                    for l in range(ceil((k-j-a)/2.))
                ]
            )
                for a in range(k-j)
              ]
        )
        delta_prime = 1 - ((1 - delta) ** k) * (1 - delta_tmp)
        constraints.append(region_from_dp_params(eps_prime, delta_prime))

        if j == 0:
            d_tv = delta_prime

    if return_d_tv:
        return intersect_regions(constraints), d_tv

    return intersect_regions(constraints)

def region_from_f_dp(f: TradeOffFunction) -> Region:
    """
    Compute the f-DP region for a given trade-off function.

    :param f: TradeOffFunction
            Function that outputs the best false negative rate for a given false positive rate.

    :return: Region
            List of constraints defining the privacy region, above f.
    """
    sum_line = lambda fp, fn: fp + fn <= 1
    main_region = lambda fp, fn: fn >= f(fp)

    return [sum_line, main_region]

def region_from_gaussian_dp(mu: float) -> Region:
    """
    Compute the Gaussian-DP region for a given mu.

    :param mu: float
            Function that outputs the best false negative rate for a given false positive rate.

    :return: Region
            List of constraints defining the Gaussian-DP privacy region, above the Gaussian trade-off function.
    """
    return region_from_f_dp(lambda fp: stats.norm.cdf(stats.norm.ppf(1 - fp) - mu))

def region_from_gaussian_dp_composition(mu_ls: List[float] | np.ndarray) -> Region:
    """
    Compute the Gaussian-DP region for a composition of GDP mechanisms.

    :param mu_ls: List[float] | np.ndarray
            List of mu GDP parameters of the composed mechanisms.

    :return: Region
            List of constraints defining the Gaussian-DP privacy region of the composition.
    """
    return region_from_gaussian_dp(float(np.linalg.norm(np.array(mu_ls))))

