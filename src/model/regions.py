import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from definitions import Constraint
from typing import Sequence
from functools import reduce

def draw_single_region_from_constraints(
        constraints: Sequence[Constraint],
        start_grid=0,
        stop_grid=1,
        grid_res=500,
        cmap="Blues",
        alpha=1
):
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


class MultiRegionFigure:
    def __init__(self,
        start_grid=0,
        stop_grid=1,
        grid_res=600
    ):
        d = np.linspace(start_grid, stop_grid, num=grid_res)
        self._k = 1
        self._x, self._y = np.meshgrid(d, d)
        self._start = start_grid
        self._stop = stop_grid
        self._labels = []

        # TODO fix and generalize palette choice
        palette = [[255, 255, 255, 0], [255, 0, 0, 200], [0, 255, 0, 200]]
        palette.extend([[120, 120, 255-50*i, 255] for i in range(10)])
        self._palette = np.array(palette)

    def add_region(self, constraints: Sequence[Constraint], label: str):
        applied_constraints = [constraint(self._x, self._y) for constraint in constraints]
        plt.imshow(self._palette[self._k * (reduce(lambda c1, c2: c1 & c2, applied_constraints)).astype(int)],
                   extent=(self._start, self._stop, self._start, self._stop),
                   origin="lower")
        self._labels.append(label)
        self._k += 1

    def finish_figure(self, title=""):
        patches = [mpatches.Patch(color=self._palette[i+1]/255., label=lab)
                   for i, lab in enumerate(self._labels)]
        plt.legend(handles=patches)
        plt.xlim(self._start, self._stop)
        plt.ylim(self._start, self._stop)
        plt.xlabel('FN')
        plt.ylabel('FP')
        plt.title(title)

    def show_figure(self):
        plt.show()

    def save_figure(self, path):
        plt.savefig(path=path)

