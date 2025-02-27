import numpy as np

from definitions import Constraint
from typing import Sequence

def region_from_dp_params(eps, delta) -> Sequence[Constraint]:
    exp = np.exp(eps)
    ineq = lambda fp, fn: fp + exp * fn >= 1 - delta
    reverse_ineq = lambda fp, fn: ineq(fn, fp)
    sum_line = lambda fp, fn: fp + fn <= 1

    return [ineq, reverse_ineq, sum_line]
