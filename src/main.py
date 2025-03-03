import numpy as np

from differential_privacy import *
from regions import *

eps = 2
delta = 0.01
constraints = region_from_dp_params(eps, delta)
# draw_single_region_from_constraints(constraints)
# draw_single_region_from_constraints(region_from_f_dp(lambda fp: np.maximum(np.maximum(0, 1 - delta - np.exp(eps) * fp), np.exp(-eps) * (1 - delta - fp))))
# draw_single_region_from_constraints(region_from_gaussian_dp(1))

fig = MultiRegionFigure()
fig.add_region(region_from_gaussian_dp(1), "1")
fig.add_region(region_from_gaussian_dp(0.4), "0.4")
fig.add_region(region_from_gaussian_dp(0.1), "0.1")
fig.finish_figure()
fig.show_figure()
