from laplacemechanism import *
from regions import *

def f_eps_delta_dp(eps, delta):
    return lambda fp: np.maximum(np.maximum(0, 1 - delta - np.exp(eps) * fp), np.exp(-eps) * (1 - delta - fp))

def draw_multiple_regions():
    eps = 0.3
    delta = 0.3
    constraints = region_from_dp_params(eps, delta)
    draw_single_region_from_constraints(constraints)
    draw_single_region_from_constraints(region_from_f_dp(f_eps_delta_dp(eps, delta)))
    draw_single_region_from_constraints(region_from_gaussian_dp(1))

def gaussians():
    fig = MultiRegionFigure()
    fig.add_region(region_from_gaussian_dp(1), "1")
    fig.add_region(region_from_gaussian_dp(0.4), "0.4")
    fig.add_region(region_from_gaussian_dp(0.1), "0.1")
    fig.finish_figure()
    fig.show_figure()

def laplacemech():
    eps = 1
    fig = MultiRegionFigure()
    lap = LaplaceMechanism(eps)
    fig.add_region(lap.region_tv(), "tv")
    fig.add_region(lap.region_exact(), "exact")
    fig.finish_figure()
    fig.show_figure()

laplacemech()
