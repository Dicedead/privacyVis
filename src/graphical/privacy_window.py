import tkinter as tk
from tkinter import ttk
from typing import Type

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from region_figures import MultiRegionFigure
from adapters import *

_WINDOW_SIZE = "1300x900"
_FIGSIZE = (7, 7)
_DPI = 100

_REGION_VALUES = [DPRegion]

_ADDER_LABELS_TO_CLS_MAP = {}
for cls in _REGION_VALUES:
    _ADDER_LABELS_TO_CLS_MAP[cls.adder_label()] = cls

_REGION_VALUES = [cls.adder_label() for cls in _REGION_VALUES]

_INITIAL_SELECTOR_VALUES = ["No region selected"]
_PRIVACY_PLOT_TITLE = "Differential privacy"

class PrivacyWindow:
    def __init__(self, window_size=_WINDOW_SIZE):
        self._window = tk.Tk()
        self._window.title(f"Privacy regions")
        self._window.geometry(window_size)

        # TODO define grid
        self._window.rowconfigure(0, weight=1)
        self._window.rowconfigure(1, weight=5)
        self._window.rowconfigure(2, weight=1)
        self._window.rowconfigure(3, weight=1)
        self._window.rowconfigure(4, weight=1)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)

        self._window.configure(background="white")

        self._privacy_fig = None
        self._privacy_canvas = None
        self._param_vals = None
        self._reg_cls: Type[AdaptedRegionComputer] = None
        self._selector_val = None
        self._selector_combob = None
        self._adder_val = None

        self.plot_privacy()
        self.build_selection_dropdown()
        self.build_addition_dropdown()

        self._window.mainloop()

    def plot_privacy(self):
        self._privacy_fig = MultiRegionFigure(figsize=_FIGSIZE, dpi=_DPI)
        self._privacy_canvas = FigureCanvasTkAgg(self._privacy_fig.get_figure(), master=self._window)

        privacy_toolbar_frame = tk.Frame(self._window)
        privacy_toolbar = NavigationToolbar2Tk(self._privacy_canvas, privacy_toolbar_frame)
        privacy_toolbar_frame.grid(column=0, row=3)
        self._privacy_canvas.get_tk_widget().grid(column=0, row=0, rowspan=2) # TODO define rowspan
        
    def replot_privacy(self):
        self._privacy_fig.clear_figure()
        
        construct_args = {param: self._param_vals[param] for param in self._reg_cls.params()}
        for param in self._reg_cls.params():
            if self._reg_cls.params_to_log()[param]:
                construct_args[param] = 10 ** self._param_vals[param]


        self._privacy_fig.add_region(
            self._reg_cls.region_computation(**construct_args),
            f"{self._reg_cls.adder_label()} ({", ".join(
                [f'{self._reg_cls.params_to_graph_labels()[param]}: {construct_args[param]:.2f}'
                           for param in self._reg_cls.params()])})"
        )
        self._privacy_fig.finish_figure(_PRIVACY_PLOT_TITLE)

        self._privacy_canvas.draw()
        self._privacy_canvas.flush_events()

    def build_selection_dropdown(self):

        def onclick(event):
            # TODO update this: put region in front
            curr_val = self._selector_val.get()
            self._selector_combob['values'] = ["1", "2"]

        selector_frame = tk.Frame(self._window)
        selector_label = tk.Label(selector_frame, text="Select a region:")
        selector_label.pack()

        self._selector_val = tk.StringVar()
        self._selector_combob = ttk.Combobox(selector_frame, textvariable=self._selector_val)
        self._selector_combob.pack()

        self._selector_combob['state'] = 'readonly'
        self._selector_combob['values'] = _INITIAL_SELECTOR_VALUES

        self._selector_combob.bind('<<ComboboxSelected>>', onclick)

        selector_frame.grid(column=1, row=0)

    def build_addition_dropdown(self):
        def onclick(event):
            curr_val = self._adder_val.get()
            
            ls = list(self._selector_combob['values'])
            ls.append(curr_val)
            self._selector_combob['values'] = ls

            self._reg_cls = _ADDER_LABELS_TO_CLS_MAP[curr_val]
            self._param_vals = self._reg_cls.params_to_default_vals()

            self.replot_privacy()
            # TODO add plotting + slider selection

        adder_frame = tk.Frame(self._window)
        adder_label = tk.Label(adder_frame, text="Add a region:")
        adder_label.pack()

        self._adder_val = tk.StringVar()
        adder_combob = ttk.Combobox(adder_frame, textvariable=self._adder_val)
        adder_combob.pack()

        adder_combob['state'] = 'readonly'
        adder_combob['values'] = _REGION_VALUES

        adder_combob.bind('<<ComboboxSelected>>', onclick)
        adder_frame.grid(column=1, row=2)


if __name__ == "__main__":
    PrivacyWindow()