import tkinter as tk
from copy import copy
from tkinter import ttk
from typing import Type

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from region_figures import MultiRegionFigure
from adapters import *

_WINDOW_SIZE = "1300x900"

_INTERSECTOR_HEIGHT_FACTOR = 25
_INTERSECTOR_BASE_HEIGHT = 50
_INTERSECTOR_WIDTH = 300

_FIGSIZE = (7, 7)
_DPI = 100
_RESOLUTION_NON_INTEGER = 0.05
_RESOLUTION_INTEGER = 1
_SLIDER_LENGTH = 200
_COMBOB_LENGTH = 35
_INTERSECT_REGIONS = "Intersect regions"
_DEFAULT_REGION_INTERSECTION_NAME = "Intersection"

_REGION_VALUES = [
    DPRegion,
    DPBasicCompositionRegion,
    DPExactCompositionRegion,
    DPTVRegion,
    DPTVCompositionRegion,
    GaussianDPRegion,
    GaussianDPCompositionRegion,
    LaplaceMechanismRegion,
    GaussianMechanismRegion,
    RandomizedResponseRegion
]

_ADDER_LABELS_TO_CLS_MAP = {}
for cls in _REGION_VALUES:
    _ADDER_LABELS_TO_CLS_MAP[cls.adder_label()] = cls

_REGION_VALUES = [cls.adder_label() for cls in _REGION_VALUES]
_REGION_VALUES.append(_INTERSECT_REGIONS)

_INITIAL_SELECTOR_VALUES = ["No region selected"]
_PRIVACY_PLOT_TITLE = "Differential privacy"

class PrivacyWindow:

    def __init__(self, window_size=_WINDOW_SIZE):
        self._window = tk.Tk()
        self._window.title(f"Privacy regions")
        self._window.geometry(window_size)

        self._window.rowconfigure(0, weight=1)
        self._window.rowconfigure(1, weight=5)
        self._window.rowconfigure(2, weight=1)
        self._window.rowconfigure(3, weight=1)
        self._window.rowconfigure(4, weight=1)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)

        self._window.configure(background="white")

        self._selector_val = tk.StringVar()

        self._privacy_fig = None
        self._privacy_canvas = None
        self._curr_param_vals = None
        self._curr_selector_label = None
        self._curr_reg_cls: Type[AdaptedRegionComputer] = None
        self._selector_label_to_param_vals: Dict[str, Dict[str, float]] = {}
        self._selector_label_to_reg_id: Dict[str, int] = {}
        self._selector_label_to_cls: Dict[str, Type[AdaptedRegionComputer]] = {}
        self._selector_label_to_reg_num: Dict[str, int] = {}
        self._toggle_reordering = tk.BooleanVar(value=False)

        self._selector_combob = None
        self._slider_frame: tk.Frame = None
        self._curr_reg_id = None
        self._region_counter = 0
        self._curr_reg_num = 0

        self.plot_privacy()
        self.build_selection_dropdown()
        self.build_addition_dropdown()
        self.build_slider_frame()
        self.build_delete_buttons()

        self._window.mainloop()

    def plot_privacy(self):
        self._privacy_fig = MultiRegionFigure(figsize=_FIGSIZE, dpi=_DPI)
        self._privacy_canvas = FigureCanvasTkAgg(self._privacy_fig.get_figure(), master=self._window)

        privacy_toolbar_frame = tk.Frame(self._window)
        privacy_toolbar = NavigationToolbar2Tk(self._privacy_canvas, privacy_toolbar_frame)
        privacy_toolbar_frame.grid(column=0, row=3)
        self._privacy_canvas.get_tk_widget().grid(column=0, row=0, rowspan=3)

    def replot_privacy(self):

        prioritized_reg = -1 if ((self._curr_selector_label in _INITIAL_SELECTOR_VALUES)
                                 or not self._toggle_reordering.get())\
            else self._curr_reg_id

        self._privacy_fig.clear_figure()
        self._privacy_fig.draw_figure(_PRIVACY_PLOT_TITLE, prioritize_region=prioritized_reg)
        self._privacy_canvas.draw()
        self._privacy_canvas.flush_events()
        
    def add_region(self):
        def _graph_label() -> str:
            def _param_label(param: str):
                if self._curr_reg_cls.params_are_integers()[param]:
                    return f'{self._curr_reg_cls.params_to_graph_labels()[param]}: {int(construct_args[param])}'
                return f'{self._curr_reg_cls.params_to_graph_labels()[param]}: {construct_args[param]:.2f}'

            if not self._curr_reg_cls.params():
                return f"{self._curr_reg_cls.region_graph_name()} [#{self._curr_reg_num}]"

            return f"{self._curr_reg_cls.region_graph_name()} ({", ".join(
                [_param_label(param) for param in self._curr_reg_cls.params()])}) [#{self._curr_reg_num}]"

        construct_args = PrivacyWindow._construct_kwargs_from_params(self._curr_param_vals, self._curr_reg_cls)
        self._curr_reg_id = self._privacy_fig.add_region(
            self._curr_reg_cls.region_computation(**construct_args),
            _graph_label()
        )

        self.replot_privacy()

    def hide_region(self, region_id: int):
        self._privacy_fig.remove_region(region_id)

    def build_selection_dropdown(self):

        def onclick(event):
            curr_val = self._selector_val.get()
            self._curr_selector_label = curr_val

            if curr_val in _INITIAL_SELECTOR_VALUES:
                self.destroy_slider_frame()
                self._curr_reg_id = None
                self._curr_reg_cls = None
                self._curr_param_vals = None
                self._curr_reg_num = None
                self.replot_privacy()
                return

            self._curr_reg_id = self._selector_label_to_reg_id[curr_val]
            self._curr_reg_cls = self._selector_label_to_cls[curr_val]
            self._curr_param_vals = self._selector_label_to_param_vals[curr_val]
            self._curr_reg_num = self._selector_label_to_reg_num[curr_val]

            self.rebuild_slider_frame()
            self.replot_privacy()

        selector_frame = tk.Frame(self._window, background="white")
        selector_label = tk.Label(selector_frame, text="Select a region:", background="white")
        selector_label.pack()

        self._selector_combob = ttk.Combobox(selector_frame, textvariable=self._selector_val, width=_COMBOB_LENGTH)
        self._selector_combob.pack()
        reordering_toggle = ttk.Checkbutton(selector_frame,
                                            text="Selected region in front",
                                            command=lambda: self.replot_privacy(),
                                            variable=self._toggle_reordering
                                            )
        reordering_toggle.pack()

        self._selector_combob['state'] = 'readonly'
        self._selector_combob['values'] = _INITIAL_SELECTOR_VALUES

        self._selector_combob.bind('<<ComboboxSelected>>', onclick)

        selector_frame.grid(column=1, row=0)

    def build_intersection_window(self):

        def command_main_button():
            intersection_region = list(contained_regions.values())
            intersection_name = text_box.get("1.0", tk.END).strip()

            if not intersection_region:
                intersector_window.destroy()
                return

            if not intersection_name:
                intersection_name = "Intersection"

            self._curr_reg_cls = intersected_regions(intersection_region, intersection_name)
            intersector_window.destroy()
            self._finish_adding(intersection_name)

        def onclick(reg_label):
            def inner():

                if reg_label in contained_regions.keys():
                    contained_regions.pop(reg_label)
                else:
                    region = self._selector_label_to_cls[reg_label].region_computation(
                        **PrivacyWindow._construct_kwargs_from_params(
                            self._selector_label_to_param_vals[reg_label],
                            self._selector_label_to_cls[reg_label]
                        )
                    )
                    contained_regions[reg_label] = region

            return inner

        intersector_window = tk.Tk()
        intersector_window.title(f"Intersection of regions")
        height = len(self._selector_combob['values']) * _INTERSECTOR_HEIGHT_FACTOR + _INTERSECTOR_BASE_HEIGHT
        length = _INTERSECTOR_WIDTH
        intersector_window.geometry(f"{length}x{height}")
        intersector_window.configure(background="white")

        contained_regions = {}

        combob_vals = copy(self._selector_combob['values'])
        combob_vals = combob_vals[1:]
        for region_label in combob_vals:
            checkbox = tk.Checkbutton(intersector_window,
                text=region_label,
                command=onclick(region_label),
                bg="white"
            )
            checkbox.pack()

        text_box = tk.Text(intersector_window, height=2, width=30)
        text_box.pack()
        text_box.insert(tk.END, _DEFAULT_REGION_INTERSECTION_NAME)

        button = tk.Button(intersector_window,
                           text="Intersect regions",
                           command=lambda: command_main_button()
                           )
        button.pack()
        intersector_window.mainloop()


    def build_addition_dropdown(self):
        def onclick(event):
            curr_val = adder_val.get()
            if curr_val != _INTERSECT_REGIONS:
                self._curr_reg_cls = _ADDER_LABELS_TO_CLS_MAP[curr_val]
            else:
                if len(self._selector_combob['values']) == 1:
                    return
                self.build_intersection_window()
                return

            self._finish_adding(curr_val)

        adder_frame = tk.Frame(self._window, background="white")
        adder_label = tk.Label(adder_frame, text="Add a region:", background="white")
        adder_label.pack()

        adder_val = tk.StringVar()
        adder_combob = ttk.Combobox(adder_frame, textvariable=adder_val, width=_COMBOB_LENGTH)
        adder_combob.pack()

        adder_combob['state'] = 'readonly'
        adder_combob['values'] = _REGION_VALUES

        adder_combob.bind('<<ComboboxSelected>>', onclick)
        adder_frame.grid(column=1, row=1)

    def build_slider_frame(self):
        self._slider_frame = tk.Frame(self._window)
        self._slider_frame.columnconfigure(0, weight=1)
        self._slider_frame.columnconfigure(1, weight=10)
        self._slider_frame.grid(column=1, row=2)

    def destroy_slider_frame(self):
        self._slider_frame.destroy()

    def rebuild_slider_frame(self):
        self.destroy_slider_frame()
        self.build_slider_frame()

        def slider_command(slider_param: str):
            def command(x):
                self._curr_param_vals[slider_param] = slider_vars[slider_param].get()
                self.hide_region(self._curr_reg_id)
                self.add_region()
                self.update_curr_reg()
            return command

        param_list = self._curr_reg_cls.params()
        limit_map = self._curr_reg_cls.params_to_limits()
        labels_map = self._curr_reg_cls.params_to_slider_labels()
        resolution_map = self._curr_reg_cls.params_are_integers()

        slider_vars = {}
        for idx, param in enumerate(param_list):
            slider_vars[param] = tk.IntVar(value=self._curr_param_vals[param]) \
                if resolution_map[param] else tk.DoubleVar(value=self._curr_param_vals[param])

            self._slider_frame.rowconfigure(idx, weight=1)
            a, b = limit_map[param]
            resolution = _RESOLUTION_INTEGER if resolution_map[param] else _RESOLUTION_NON_INTEGER
            scale = tk.Scale(self._slider_frame,
                             from_=a, to=b,
                             variable=slider_vars[param],
                             command=slider_command(param),
                             resolution=resolution,
                             background="white",
                             orient=tk.HORIZONTAL,
                             length=_SLIDER_LENGTH)

            scale.grid(column=1, row=idx)
            label = tk.Label(self._slider_frame, text=labels_map[param])
            label.grid(column=0, row=idx)

    def build_delete_buttons(self):

        def remove_region():
            if self._curr_reg_id is not None:
                self.hide_region(self._curr_reg_id
                                 )
                ls = list(self._selector_combob['values'])
                ls.pop(ls.index(self._curr_selector_label))
                self._selector_combob['values'] = ls

                self.replot_privacy()
                self._curr_reg_id = None
                self._curr_reg_cls = None
                self._curr_param_vals = None
                self._curr_reg_num = None

                self._selector_label_to_reg_num.pop(self._curr_selector_label)
                self._selector_label_to_cls.pop(self._curr_selector_label)
                self._selector_label_to_param_vals.pop(self._curr_selector_label)
                self._selector_label_to_reg_id.pop(self._curr_selector_label)

                self.destroy_slider_frame()
                self._selector_val.set("No region selected")

        def remove_everything():
            self._window.destroy()
            newwindow = PrivacyWindow()

        delete_frame = tk.Frame(self._window, background="white")

        delete_reg_button = tk.Button(delete_frame,
                                      text="Delete current region",
                                      command=lambda: remove_region(),
                                      bg="white"
                                      )
        delete_reg_button.pack()

        delete_everything_button = tk.Button(delete_frame,
                                             text="      Clear all      ",
                                             command=lambda: remove_everything(),
                                             bg="white"
                                             )
        delete_everything_button.pack()

        delete_frame.grid(column=1, row=3)

    def update_curr_reg(self):
        self._selector_label_to_reg_id[self._curr_selector_label] = self._curr_reg_id
        self._selector_label_to_cls[self._curr_selector_label] = self._curr_reg_cls
        self._selector_label_to_param_vals[self._curr_selector_label] = copy(self._curr_param_vals)
        self._selector_label_to_reg_num[self._curr_selector_label] = self._curr_reg_num

    def _finish_adding(self, curr_val):
        self._region_counter += 1
        self._curr_reg_num = self._region_counter
        selector_label = curr_val + f" [#{self._region_counter}]"

        ls = list(self._selector_combob['values'])
        ls.append(selector_label)
        self._selector_combob['values'] = ls

        self._curr_param_vals = self._curr_reg_cls.params_to_default_vals()

        self.add_region()
        self.rebuild_slider_frame()

        self._curr_selector_label = selector_label
        self.update_curr_reg()

        self._selector_val.set(selector_label)

    @staticmethod
    def _construct_kwargs_from_params(region_params: Dict[str, float], region_cls: Type[AdaptedRegionComputer]):
        construct_args = copy(region_params)
        for param in region_cls.params():
            if region_cls.params_are_logscale()[param]:
                construct_args[param] = 10 ** region_params[param]

        return construct_args

if __name__ == "__main__":
    PrivacyWindow()