import copy
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from definitions import Constraint, Region
from typing import Sequence, List, Tuple
from functools import reduce

from palettes import colourblind_palette

_TO_REMOVE = None


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
        grid_res=600,
        palette=None,
        figsize=(6, 6),
        dpi=100
    ):
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._plot = self._fig.add_subplot()
        d = np.linspace(start_grid, stop_grid, num=grid_res)
        self._labelled_regions: List[Tuple[Sequence[Constraint], str]] = []
        self._x, self._y = np.meshgrid(d, d)
        self._start = start_grid
        self._stop = stop_grid
        self._region_id = -1

        if palette is None:
            palette = copy.deepcopy(colourblind_palette())

        self._palette = np.array(palette)

    def add_region(self, constraints: Sequence[Constraint], label: str) -> int:
        double = (constraints, label)
        self._labelled_regions.append(double)
        self._region_id += 1
        return self._region_id

    def remove_region(self, region_id: int):
        self._labelled_regions[region_id] = _TO_REMOVE

    def finish_figure(self, title=""):
        self.draw_figure(title=title)

    def draw_figure(self, title=""):
        shown_regions = [reg for reg in self._labelled_regions if reg is not _TO_REMOVE]
        labels = []

        for idx, labelled_computed_region in enumerate(self._compute_and_sort_regions(shown_regions)):
            k = idx + 1
            computed_region, label = labelled_computed_region
            self._plot.imshow(self._palette[k * computed_region],
                       extent=(self._start, self._stop, self._start, self._stop),
                       origin="lower")
            labels.append(label)
        patches = [mpatches.Patch(color=self._palette[(i+1) % len(self._palette)]/255., label=lab)
                   for i, lab in enumerate(labels)]
        self._plot.legend(handles=patches)
        self._plot.set(xlim=(self._start, self._stop), ylim=(self._start, self._stop))
        self._plot.set_title(title)
        self._plot.set(xlabel="FN", ylabel="FP")

    def show_figure(self):
        self._fig.show()

    def get_figure(self):
        return self._fig

    def clear_figure(self):
        self._plot.clear()

    def reset_figure(self):
        self._labelled_regions.clear()
        self._region_id = -1
        self.clear_figure()

    def save_figure(self, path):
        self._fig.savefig(fname=path)

    def _compute_region(self, region: Region):
        applied_constraints = [constraint(self._x, self._y) for constraint in region]
        return reduce(lambda c1, c2: c1 & c2, applied_constraints).astype(int)

    def _compute_and_sort_regions(self, labelled_regions: List[Tuple[Sequence[Constraint], str]])\
            -> List[Tuple[np.ndarray, str]]:
        computed_labelled_regions = [
            (self._compute_region(reg), label) for (reg, label) in labelled_regions
        ]

        computed_labelled_regions.sort(key=functools.cmp_to_key(MultiRegionFigure._compare_regions), reverse=True)

        return computed_labelled_regions

    @staticmethod
    def _compare_regions(region1: Tuple[np.ndarray, str], region2: Tuple[np.ndarray, str]) -> int:
        diff = region1[0] - region2[0]
        one_contains_two = np.all(diff >= 0)
        two_contains_one = np.all(-diff >= 0)

        if one_contains_two and not two_contains_one:
            return 1

        if two_contains_one and not one_contains_two:
            return -1

        if one_contains_two and two_contains_one:
            return 0

        return diff.sum()
