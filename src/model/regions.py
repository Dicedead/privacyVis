import numpy as np
import matplotlib.pyplot as plt

from definitions import Constraint
from typing import Sequence
from functools import reduce

def draw_region_from_constraints(
        constraints: Sequence[Constraint],
        start_grid=0,
        stop_grid=1,
        grid_res=500,
        cmap="Greys",
        alpha=0.3
):
    # TODO make this function not show directly + possibility to plot multiple regions on same figure + make figure larger
    """
    Draw a region from a list of constraints.

    :param constraints: Sequence[Constraint]
            Given constraints.

    :param start_grid: float
            Smallest grid value, defaults to 0.

    :param stop_grid: float
            Largest grid value, defaults to 1.

    :param grid_res: int
            Grid resolution, defaults to 500.

    :param cmap: str
            Colormap, defaults to "Greys".

    :param alpha: float
            Alpha value, defaults to 0.3.
    """

    assert len(constraints) > 0

    d = np.linspace(start_grid, stop_grid, num=grid_res)
    x, y = np.meshgrid(d, d)
    applied_constraints = [constraint(x,y) for constraint in constraints]
    plt.imshow((reduce(lambda c1, c2: c1 & c2, applied_constraints)).astype(int),
               extent=(start_grid, stop_grid, start_grid, stop_grid), origin="lower", cmap=cmap, alpha=alpha)
    plt.xlim(start_grid, stop_grid)
    plt.ylim(start_grid, stop_grid)

    plt.xlabel('FN')
    plt.ylabel('FP')
    plt.show()
