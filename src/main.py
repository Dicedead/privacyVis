from differential_privacy import *
from regions import *

eps = 2
delta = 0.01
constraints = region_from_dp_params(eps, delta)
draw_region_from_constraints(constraints)
