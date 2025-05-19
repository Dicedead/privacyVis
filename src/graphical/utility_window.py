import tkinter as tk
from typing import List, Dict, Type

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from histogram import DPHistogram
from query import DPQuery
from regions import MultiRegionFigure


# plot function is created for
# plotting the graph in  
# tkinter window 
# def plot():
#     # the figure that will contain the plot
#     fig = Figure(figsize=(5, 5),
#                  dpi=100)
#
#     # list of squares
#     y = [i ** 2 for i in range(101)]
#
#     # adding the subplot
#     plot1 = fig.add_subplot(111)
#
#     # plotting the graph
#     plot1.plot(y)
#
#     # creating the Tkinter canvas
#     # containing the Matplotlib figure
#     canvas = FigureCanvasTkAgg(fig, master=window)
#     canvas.draw()
#
#     # placing the canvas on the Tkinter window
#     canvas.get_tk_widget().pack()
#
#     # creating the Matplotlib toolbar
#     toolbar = NavigationToolbar2Tk(canvas,
#                                    window)
#     toolbar.update()
#
#     # placing the toolbar on the Tkinter window
#     canvas.get_tk_widget().pack()
#
#
# # the main Tkinter window
# window = tk.Tk()
#
# # setting the title
# window.title('Plotting in Tkinter')
#
# # dimensions of the main window
# window.geometry("500x500")
#
# # button that displays the plot
# plot_button = tk.Button(master=window,
#                      command=plot,
#                      height=2,
#                      width=10,
#                      text="Plot")
#
# # place the button
# # in main window
# plot_button.pack()
#
# # run the gui
# window.mainloop()

class UtilityWindow:
    def __init__(self,
        dpqcls: Type[DPQuery],
        window_size="1200x700"
    ):
        self._dpqcls = dpqcls

        self._window = tk.Tk()
        self._window.title(f"Utility / Privacy trade-off for the {dpqcls.pretty_name()} query")
        self._window.geometry(window_size)

        self._window.rowconfigure(0, weight=6)
        self._window.rowconfigure(1, weight=1)
        self._window.rowconfigure(2, weight=5)
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=1)

        #self._window.resizable(width=False, height=False)
        self._window.configure(background="white")

    def plot_utility(self, main_param: str, other_params: Dict[str, float]):
        fig = Figure(figsize=(6, 6), dpi=100)
        plot1 = fig.add_subplot()

        kwargs_builder = {
            self._dpqcls.params_to_kwargs()[main_param]:np.logspace(*self._dpqcls.params_to_limits()[main_param])
        }

        for other_param in other_params.keys():
            kwargs_builder.update(
                {self._dpqcls.params_to_kwargs()[other_param]: other_params[other_param]}
            )

        plot1.plot(self._dpqcls.utility_func(**kwargs_builder))

        utility_canvas = FigureCanvasTkAgg(fig, master=self._window)
        utility_toolbar_frame = tk.Frame(self._window)
        utility_toolbar = NavigationToolbar2Tk(utility_canvas, utility_toolbar_frame)
        utility_toolbar_frame.grid(column=0, row=1)
        utility_canvas.get_tk_widget().grid(column=0, row=0)

        privacy_fig = MultiRegionFigure()
        privacy_fig.add_region(DPHistogram(0.1, 5).privacy_region(), "Hist")
        privacy_fig.finish_figure("Differential privacy")

        privacy_canvas = FigureCanvasTkAgg(privacy_fig.get_figure(), master=self._window)
        privacy_toolbar_frame = tk.Frame(self._window)
        privacy_toolbar = NavigationToolbar2Tk(privacy_canvas, privacy_toolbar_frame)
        privacy_toolbar_frame.grid(column=1, row=1)
        privacy_canvas.get_tk_widget().grid(column=1, row=0)

        self._window.mainloop()

if __name__ == "__main__":
    utility_window = UtilityWindow(DPHistogram)
    utility_window.plot_utility("eps", {"l1_sens": 1})
