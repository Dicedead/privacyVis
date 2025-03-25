from gaussian_mechanism import GaussianMechanism
from laplace_mechanism import *
from randomized_response import RandomizedResponse
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
    draw_single_region_from_constraints(region_from_dp_composition_exact(eps, delta, 3))

def gdp():
    fig = MultiRegionFigure()
    fig.add_region(region_from_gaussian_dp(2), "2")
    fig.add_region(region_from_gaussian_dp(1), "1")
    fig.add_region(region_from_gaussian_dp(0.4), "0.4")
    fig.add_region(region_from_gaussian_dp(0.1), "0.1")
    fig.finish_figure()
    fig.show_figure()

def composition_comparison():
    eps = 0.3
    delta = 0.3
    k = 3
    fig = MultiRegionFigure()
    fig.add_region(region_from_dp_composition_basic(eps, delta, k), "basic")
    fig.add_region(region_from_dp_composition_exact(eps, delta, k), "exact")
    fig.finish_figure(title="Comparison of basic and exact composition theorems")
    fig.show_figure()

def laplacemech():
    eps = 1
    fig = MultiRegionFigure()
    lap = LaplaceMechanism(eps)
    fig.add_region(region_from_dp_params(eps, 0), "dp no tv")
    fig.add_region(lap.region_tv(), "tv")
    fig.add_region(lap.region_exact(), "exact")
    fig.finish_figure()
    fig.show_figure()

def gaussianmech():
    eps = 1.3
    delta = 0.3
    fig = MultiRegionFigure()
    gau = GaussianMechanism(eps, delta)
    fig.add_region(gau.region_exact(), "exact")
    fig.add_region(gau.region_tv(), "tv")
    fig.finish_figure()
    fig.show_figure()

def randomized_resp():
    eps = 0.3
    mech = RandomizedResponse(eps, 5)
    fig = MultiRegionFigure()
    fig.add_region(region_from_dp_params(eps, 0), "dp no tv")
    fig.add_region(mech.region_exact(), "actual")
    fig.finish_figure()
    fig.show_figure()

randomized_resp()
