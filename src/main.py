import tkinter as tk
import tkinter.ttk as ttk

from histogram import DPHistogram
from mean import DPMean
from median import DPMedian
from randomized_response import RandomizedResponse

from privacy_window import PrivacyWindow
from utility_window import UtilityWindow

_BUTTON_LENGTH = 50

class MainWindow:
    def __init__(self):
        self._window = tk.Tk()
        self._window.configure(background="white")
        self._window.title("Differential privacy")
        self._window.geometry("400x200")

        self._window.rowconfigure(0, weight=1)
        self._window.rowconfigure(1, weight=1)
        self._window.rowconfigure(2, weight=1)
        self._window.rowconfigure(3, weight=1)
        self._window.columnconfigure(0, weight=1)

        self._build_button_privacy()
        self._build_combob_utility()

        self._window.mainloop()


    def _build_button_privacy(self):
        def onclick():
            self._window.destroy()
            PrivacyWindow()

        privacy_button = ttk.Button(self._window,
                                    text="Open privacy regions window",
                                    command=lambda: onclick(),
                                    width=_BUTTON_LENGTH
                                    )
        privacy_button.grid(column=0, row=1)

    def _build_combob_utility(self):
        def onclick(event):
            curr_val = combob_utilities.get()
            self._window.destroy()
            if curr_val == "DP Histogram with Laplace mechanism (ε)":
                UtilityWindow(DPHistogram, "eps")
            elif curr_val == "DP Mean with Gaussian mechanism (ε)":
                UtilityWindow(DPMean, "eps")
            elif curr_val == "DP Mean with Gaussian mechanism (δ)":
                UtilityWindow(DPMean, "delta")
            elif curr_val == "DP Median with exponential mechanism (ε)":
                UtilityWindow(DPMedian, "eps")
            elif curr_val == "DP Median with exponential mechanism (alphabet size)":
                UtilityWindow(DPMedian, "alphabet_size")
            elif curr_val == "Randomized response (ε)":
                UtilityWindow(RandomizedResponse, "eps")
            elif curr_val == "Randomized response (alphabet size)":
                UtilityWindow(RandomizedResponse, "alphabet_size")
            else:
                raise ValueError("Unknown utility")

        utilities_frame = tk.Frame(self._window, background="white")

        combob_utilities = ttk.Combobox(utilities_frame, width=_BUTTON_LENGTH)
        combob_utilities['values'] = (
            "DP Histogram with Laplace mechanism (ε)",
            "DP Mean with Gaussian mechanism (ε)",
            "DP Mean with Gaussian mechanism (δ)",
            "DP Median with exponential mechanism (ε)",
            "DP Median with exponential mechanism (alphabet size)",
            "Randomized response (ε)",
            "Randomized response (alphabet size)"
        )
        combob_utilities['state'] = 'readonly'
        combob_utilities.bind('<<ComboboxSelected>>', onclick)

        utilities_label = tk.Label(utilities_frame, text="Open privacy/utility trade-off window", bg="white")

        utilities_label.pack()
        combob_utilities.pack()

        utilities_frame.grid(column=0, row=2)

if __name__ == "__main__":
    main_window = MainWindow()
