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
    lap = LaplaceMechanism(eps, 1)
    fig.add_region(region_from_dp_params(eps, 0), "dp no tv")
    fig.add_region(lap.region_tv(), "tv")
    fig.add_region(lap.region_exact(), "exact")
    fig.finish_figure()
    fig.show_figure()

def gaussianmech():
    eps = 1.3
    delta = 0.3
    fig = MultiRegionFigure()
    gau = GaussianMechanism(eps, delta, 1)
    fig.add_region(gau.region_exact(), "exact")
    fig.add_region(gau.region_tv(), "tv")
    fig.finish_figure()
    fig.show_figure()

def randomized_resp():
    eps = 0.9
    mech = RandomizedResponse(eps, 5)
    fig = MultiRegionFigure()
    fig.add_region(region_from_dp_params(eps, 0), "dp no tv")
    fig.add_region(mech.region_exact(), "actual")
    fig.finish_figure()
    fig.show_figure()

def composing_dps_intersection_tv(eps = 0.5, delta = 0.1, eta_factor = 0.75):

    eta = eta_factor * (delta + ((np.exp(eps) - 1) * (1 - delta)) / (np.exp(eps) + 1))

    intersection = intersect_regions([
        region_from_dp_composition_exact(eps, delta, 2),  # (eps, delta)-DP x (eps, delta) - DP
        region_from_dp_params(0, 1 - (1 - eta) * (1 - eta)),  # (0, eta)-DP x (0, eta)-DP
        region_from_dp_params(eps, 1 - (1-delta) * (1-eta)) # (eps, delta)-DP x (0, eta)-DP
        ]
    )
    factor = 0.2

    intersection_2 = intersect_regions([
        region_from_dp_composition_exact(eps, delta, 2),  # (eps, delta)-DP x (eps, delta) - DP
        region_from_dp_params(0, 1 - (1 - eta) * (1 - eta)),  # (0, eta)-DP x (0, eta)-DP
        region_from_dp_params(eps, 1 - (1-delta) * (1-eta)), # (eps, delta)-DP x (0, eta)-DP
        region_from_dp_composition_exact(np.log(((1 - factor * delta)/(1-delta)) * (1 + np.exp(eps)) - 1), delta, 2)
        ] # instead of 1.0002 * delta : play with another delta between the original delta and eta
    )

    fig = MultiRegionFigure()
    #fig.add_region(intersection, "intersection")
    fig.add_region(intersection_2, "intersection_2")
    fig.add_region(region_from_dp_composition_exact_total_var(eps, delta, eta, 2), "theorem")

    fig.finish_figure(f"epsilon={eps}, delta={delta}, eta={eta}")
    fig.show_figure()

def composing_dps_intersection_tv_2(eps = 0.9, delta = 0.2, eta = 0.4):

    intersection = intersect_regions([
        region_from_dp_composition_exact(eps, delta, 2),  # (eps, delta)-DP x (eps, delta) - DP
        region_from_dp_params(0, 1 - (1 - eta) * (1 - eta)),  # (0, eta)-DP x (0, eta)-DP
        region_from_dp_composition_simplified([eps, 0], [delta, eta], delta_slack=0.001)  # (eps, delta)-DP x (0, eta)-DP
        ]
    )

    fig = MultiRegionFigure()
    fig.add_region(intersection, "intersection 2")
    fig.finish_figure(f"epsilon={eps}, delta={delta}, eta={eta}")
    fig.show_figure()

def composing_dps_theorem_tv(eps = 0.9, delta = 0.2, eta = 0.4):
    fig = MultiRegionFigure()
    fig.add_region(region_from_dp_composition_exact_total_var(eps, delta, eta, 2), "theorem")
    fig.finish_figure()
    fig.show_figure()

def composing_dps_comp(eps = 0.9, delta = 0.2, eta = 0.4):
    fig = MultiRegionFigure()
    fig.add_region(region_from_dp_params(eps, 1 - (1-delta) * (1-eta)), "dp")
    fig.add_region(region_from_dp_composition_simplified([eps, 0], [delta, eta], delta_slack=0.001), "slack")
    fig.finish_figure()
    fig.show_figure()

def composition_heter():
    eps_1 = 0.5
    eps_2 = eps_1
    delta_1 = 0.2
    delta_2 = delta_1

    fig_1 = MultiRegionFigure()
    fig_1.add_region(region_from_dp_params(eps_1, delta_1), "1")
    fig_1.finish_figure("eps1-delta1")
    fig_1.show_figure()

    fig_2 = MultiRegionFigure()
    fig_2.add_region(region_from_dp_params(eps_2, delta_2), "2")
    fig_2.finish_figure("eps2-delta2")
    fig_2.show_figure()

    fig = MultiRegionFigure(grid_res=1000)
    ls = []
    for delta_s in np.logspace(-5, 0, num=500):
        ls.append(region_from_dp_composition_simplified([eps_1, eps_2], [delta_1, delta_2], delta_s))

    fig.add_region(intersect_regions(ls), "intersection thm 3.5")
    fig.finish_figure()
    fig.show_figure()

    fig_prime = MultiRegionFigure()
    fig_prime.add_region(region_from_dp_composition_exact(eps_1, delta_1, 2), "exact")
    fig_prime.finish_figure()
    fig_prime.show_figure()


composition_heter()
