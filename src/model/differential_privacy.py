import scipy.special as sps
import numpy as np

from definitions import Region
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


def region_from_dp_params(eps, delta) -> Region:
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


def region_from_dp_composition_basic(eps, delta, k) -> Region:
    """
    Compute the differential privacy region basic composition corresponding to the basic result for the composition
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
    return region_from_dp_params(k * eps, k * delta)


def region_from_dp_composition_exact(eps, delta, k) -> Region:
    """
    Compute the differential privacy region basic composition corresponding to the improved result for the composition
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
    constraints = []
    for i in range(np.floor(k/2)+1):
        eps_prime = (k - 2 * i) * eps
        delta_tmp = sum([sps.comb(k, l) * (np.exp((k-l) * eps) - np.exp((k-2*i+l) * eps)) for l in range(i)])
        delta_tmp /= (1+np.exp(eps)) ** k
        delta_prime = 1 - ((1 - delta) ** k) * (1 - delta_tmp)
        constraints.append(region_from_dp_params(eps_prime, delta_prime))

    return intersect_regions(constraints)