import tkinter as tk
from typing import Type

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from histogram import DPHistogram
from mean import DPMean
from query import DPQuery
from region_figures import MultiRegionFigure

_RESOLUTION_NON_INTEGER = 0.05
_RESOLUTION_INTEGER = 1
_SLIDER_LENGTH = 300
_WINDOW_SIZE = "1300x900"
_PRIVACY_PLOT_TITLE = "Differential privacy"

class UtilityWindow:
    def __init__(self,
        dpqcls: Type[DPQuery],
        main_param: str,
        window_size=_WINDOW_SIZE
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

        self._window.configure(background="white")

        param_to_defaults = self._dpqcls.params_to_default_vals()
        self._param_vals = {param: tk.DoubleVar(value=val) for param, val in param_to_defaults.items()}
        self._utility_canvas = None
        self._utility_plot = None
        self._privacy_canvas = None
        self._privacy_fig = None

        self.build_sliders(main_param)
        self.plot_utility(main_param)
        self.plot_privacy()

        self._window.mainloop()

    def plot_utility(self, main_param: str):
        utility_fig = Figure(figsize=(5, 5), dpi=100)

        self._utility_plot = utility_fig.add_subplot()
        self._utility_canvas = FigureCanvasTkAgg(utility_fig, master=self._window)
        self.replot_utility(main_param)

        utility_toolbar_frame = tk.Frame(self._window)
        utility_toolbar = NavigationToolbar2Tk(self._utility_canvas, utility_toolbar_frame)
        utility_toolbar_frame.grid(column=0, row=1, sticky="n")
        self._utility_canvas.get_tk_widget().grid(column=0, row=0)

    def replot_utility(self, main_param: str):
        utility_plot = self._utility_plot
        utility_plot.clear()

        kwargs_builder = {}
        if self._dpqcls.params_are_in_logscale()[main_param]:
            x_vals = np.logspace(*self._dpqcls.params_to_limits()[main_param])
            utility_plotting_func = utility_plot.loglog
        else:
            x_vals = np.linspace(*self._dpqcls.params_to_limits()[main_param])
            utility_plotting_func = utility_plot.plot

        kwargs_builder.update({self._dpqcls.params_to_kwargs()[main_param]: x_vals})

        param_vals = {param: val.get() for param, val in self._param_vals.items()}
        for param in self._dpqcls.params():
            if self._dpqcls.params_are_in_logscale()[param]:
                param_vals[param] = 10 ** param_vals[param]

        main_param_val = param_vals.pop(main_param)

        for param in param_vals.keys():
            kwargs_builder.update(
                {self._dpqcls.params_to_kwargs()[param]: param_vals[param]}
            )

        utility_plotting_func(x_vals, self._dpqcls.utility_func(**kwargs_builder))
        utility_plot.axvline(x=main_param_val, color='black', linestyle='--')
        utility_plot.set_xlabel(self._dpqcls.params_to_graph_labels()[main_param])
        utility_plot.set_ylabel(self._dpqcls.utility_label())
        utility_plot.set_title("Utility")

        self._utility_canvas.draw()
        self._utility_canvas.flush_events()

    def plot_privacy(self):
        self._privacy_fig = MultiRegionFigure()
        self._privacy_canvas = FigureCanvasTkAgg(self._privacy_fig.get_figure(), master=self._window)

        self.replot_privacy()

        privacy_toolbar_frame = tk.Frame(self._window)
        privacy_toolbar = NavigationToolbar2Tk(self._privacy_canvas, privacy_toolbar_frame)
        privacy_toolbar_frame.grid(column=1, row=2)
        self._privacy_canvas.get_tk_widget().grid(column=1, row=0, rowspan=3)

    def replot_privacy(self):
        self._privacy_fig.clear_figure()
        construct_args = {param: self._param_vals[param].get() for param in self._dpqcls.params()}
        for param in self._dpqcls.params():
            if self._dpqcls.params_are_in_logscale()[param]:
                construct_args[param] = 10 ** self._param_vals[param].get()


        self._privacy_fig.add_region(
            self._dpqcls(**construct_args).privacy_region(),
            f"({", ".join([f'{self._dpqcls.params_to_graph_labels()[param]}: {construct_args[param]:.2f}'
                           for param in self._dpqcls.params() if self._dpqcls.params_change_privacy()[param]])})")
        self._privacy_fig.finish_figure(_PRIVACY_PLOT_TITLE)

        self._privacy_canvas.draw()
        self._privacy_canvas.flush_events()

    def replot_privacy_and_utility(self, main_param: str):
        def func(_):
            self.replot_utility(main_param)
            self.replot_privacy()
        return func

    def build_sliders(self, main_param: str):

        def slider_command(slider_param: str):
            if self._dpqcls.params_change_privacy()[slider_param]:
                return self.replot_privacy_and_utility(main_param)

            return lambda x: self.replot_utility(main_param)

        slider_frame = tk.Frame(self._window)

        param_list = self._dpqcls.params()
        limit_map = self._dpqcls.params_to_limits()
        labels_map = self._dpqcls.params_to_slider_labels()
        resolution_map = self._dpqcls.params_are_integers()

        slider_frame.columnconfigure(0, weight=1)
        slider_frame.columnconfigure(1, weight=10)
        for idx, param in enumerate(param_list):
            slider_frame.rowconfigure(idx, weight=1)
            a, b = limit_map[param]
            resolution = _RESOLUTION_INTEGER if resolution_map[param] else _RESOLUTION_NON_INTEGER
            scale = tk.Scale(slider_frame,
                             from_=a, to=b,
                             variable=self._param_vals[param],
                             command=slider_command(param),
                             resolution=resolution,
                             background="white",
                             orient=tk.HORIZONTAL,
                             length=_SLIDER_LENGTH)

            scale.grid(column=1, row=idx)
            label = tk.Label(slider_frame, text=labels_map[param])
            label.grid(column=0, row=idx)

        slider_frame.grid(column=0, row=2)

if __name__ == "__main__":
    # utility_window = UtilityWindow(DPHistogram, "eps")
    utility_window = UtilityWindow(DPMean, "delta")