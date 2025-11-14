import copy
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from definitions import Constraint, Region, LINE_REGION_THICKNESS, SUM_LINE
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
        dpi=100,
        show_line=True
    ):
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._plot = self._fig.add_subplot()
        d = np.linspace(start_grid, stop_grid, num=grid_res)
        self._labelled_regions: List[Tuple[Sequence[Constraint], str]] = []
        self._x, self._y = np.meshgrid(d, d)
        self._start = start_grid
        self._stop = stop_grid
        self._region_id = -1
        self._show_line = show_line

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

    def draw_figure(self, title="", prioritize_region=-1, show_legend=True):
        shown_regions = [(reg[0], reg[1], idx) for idx, reg in enumerate(self._labelled_regions) if reg is not _TO_REMOVE]
        labels = []

        for idx, labelled_computed_region in enumerate(self._compute_and_sort_regions(shown_regions, prioritize_region)):
            k = (idx + 1) % len(self._palette)
            computed_region, label = labelled_computed_region
            self._plot.imshow(self._palette[k * computed_region],
                       extent=(self._start, self._stop, self._start, self._stop),
                       origin="lower")
            labels.append(label)

        if show_legend:
            patches = [mpatches.Patch(color=self._palette[(i+1) % len(self._palette)]/255., label=lab)
                       for i, lab in enumerate(labels)]
            self._plot.legend(handles=patches)
        self._plot.set(xlim=(self._start, self._stop), ylim=(self._start, self._stop))
        self._plot.set_title(title)
        self._plot.set(xlabel="False negative probability", ylabel="False positive probability")

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
        applied_constraints = [constraint(self._x, self._y) for constraint in region
                               if constraint is not SUM_LINE or not self._show_line]
        whole_reg = reduce(lambda c1, c2: c1 & c2, applied_constraints).astype(int)

        if not self._show_line:
            return whole_reg

        row_csum = np.cumsum(whole_reg, axis=1)
        line_reg = (row_csum <= LINE_REGION_THICKNESS).astype(whole_reg.dtype) * whole_reg

        col_csum = np.cumsum(whole_reg, axis=0)
        col_reg = (col_csum <= LINE_REGION_THICKNESS).astype(whole_reg.dtype) * whole_reg

        return np.bitwise_or(line_reg, col_reg)

    def _compute_and_sort_regions(self, labelled_regions: List[Tuple[Sequence[Constraint], str, int]], prioritize_region) \
            -> List[Tuple[np.ndarray, str]]:
        computed_labelled_regions = [
            (self._compute_region(reg), label, idx) for (reg, label, idx) in labelled_regions
        ]

        computed_labelled_regions.sort(key=functools.cmp_to_key(MultiRegionFigure._region_comparator(prioritize_region)),
                                       reverse=True)

        return [(reg, label) for (reg, label, idx) in computed_labelled_regions]

    @staticmethod
    def _region_comparator(prioritize_region: int):

        def inner(region1: Tuple[np.ndarray, str, int], region2: Tuple[np.ndarray, str, int]):
            reg1, _, index1 = region1
            reg2, _, index2 = region2

            if index1 == prioritize_region:
                return -1
            elif index2 == prioritize_region:
                return 1

            return (reg1 - reg2).sum()

        return inner
