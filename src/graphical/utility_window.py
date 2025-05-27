import copy
import tkinter as tk
from typing import List, Dict, Type, MutableSet

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from mpmath import limit

from histogram import DPHistogram
from query import DPQuery
from regions import MultiRegionFigure


class UtilityWindow:
    def __init__(self,
        dpqcls: Type[DPQuery],
        window_size="1300x900"
    ):
        self._dpqcls = dpqcls

        self._window = tk.Tk()
        self._window.title(f"Utility / Privacy trade-off for the {dpqcls.pretty_name()} query")
        self._window.geometry(window_size)

        self._window.rowconfigure(0, weight=15)
        self._window.rowconfigure(1, weight=1)
        self._window.rowconfigure(2, weight=13)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)

        #self._window.resizable(width=False, height=False)
        self._window.configure(background="white")

        param_to_defaults = self._dpqcls.params_to_default_vals()
        self._param_vals = {param: tk.DoubleVar(value=val) for param, val in param_to_defaults.items()}


    def build_window(self, main_param: str):
        self.build_sliders()
        self.plot_utility(main_param)
        self.plot_privacy()
        self.plot_example()

        self._window.mainloop()

    def plot_utility(self, main_param: str):
        utility_fig = Figure(figsize=(5, 5), dpi=100)

        utility_plot = utility_fig.add_subplot()

        kwargs_builder = {}
        if self._dpqcls.params_to_log()[main_param]:
            x_vals = np.logspace(*self._dpqcls.params_to_limits()[main_param])
            utility_plotting_func = utility_plot.semilogx
        else:
            x_vals = np.linspace(*self._dpqcls.params_to_limits()[main_param])
            utility_plotting_func = utility_plot.plot

        kwargs_builder.update({self._dpqcls.params_to_kwargs()[main_param]: x_vals})

        other_params = {param: val.get() for param, val in self._param_vals.items()}
        other_params.pop(main_param)

        for other_param in other_params.keys():
            kwargs_builder.update(
                {self._dpqcls.params_to_kwargs()[other_param]: other_params[other_param]}
            )

        utility_plotting_func(x_vals, self._dpqcls.utility_func(**kwargs_builder))
        utility_plot.set_xlabel(self._dpqcls.params_to_graph_labels()[main_param])
        utility_plot.set_ylabel(self._dpqcls.utility_label())
        utility_plot.set_title("Utility")

        utility_canvas = FigureCanvasTkAgg(utility_fig, master=self._window)
        utility_toolbar_frame = tk.Frame(self._window)
        utility_toolbar = NavigationToolbar2Tk(utility_canvas, utility_toolbar_frame)
        utility_toolbar_frame.grid(column=0, row=1, sticky="n")
        utility_canvas.get_tk_widget().grid(column=0, row=0)

    def plot_privacy(self):
        privacy_fig = MultiRegionFigure()
        privacy_fig.add_region(DPHistogram(0.1, 5).privacy_region(), "Hist")
        privacy_fig.finish_figure("Differential privacy")

        privacy_canvas = FigureCanvasTkAgg(privacy_fig.get_figure(), master=self._window)
        privacy_toolbar_frame = tk.Frame(self._window)
        privacy_toolbar = NavigationToolbar2Tk(privacy_canvas, privacy_toolbar_frame)
        privacy_toolbar_frame.grid(column=1, row=1, sticky="n")
        privacy_canvas.get_tk_widget().grid(column=1, row=0)

    def plot_example(self):
        bogus_fig = MultiRegionFigure(figsize=(5,5))
        #bogus_fig.add_region(DPHistogram(0.1, 5).privacy_region(), "Hist bogus")
        bottom_fig = FigureCanvasTkAgg(bogus_fig.get_figure(), master=self._window)
        bottom_fig.get_tk_widget().grid(column=1, row=2)

    def build_sliders(self):
        slider_frame = tk.Frame(self._window)

        param_list = self._dpqcls.params_to_default_vals().keys()
        limit_map = self._dpqcls.params_to_limits()
        labels_map = self._dpqcls.params_to_slider_labels()

        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=5)
        for idx, param in enumerate(param_list):
            slider_frame.rowconfigure(idx, weight=1)
            a, b = limit_map[param]
            scale = tk.Scale(slider_frame, from_=a, to=b, orient=tk.HORIZONTAL, variable=self._param_vals[param],
                             resolution=0.1, background="white")
            # TODO replot the utility plot on command
            scale.grid(column=1, row=idx)

            label = tk.Label(slider_frame, text=labels_map[param])
            label.grid(column=0, row=idx)


        slider_frame.grid(column=0, row=2)

if __name__ == "__main__":
    utility_window = UtilityWindow(DPHistogram)
    utility_window.build_window("eps")
