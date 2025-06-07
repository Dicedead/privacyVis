import tkinter as tk
import tkinter.ttk as ttk

from histogram import DPHistogram
from mean import DPMean
from privacy_window import PrivacyWindow
from utility_window import UtilityWindow

class MainWindow:
    def __init__(self):
        self._window = tk.Tk()
        self._window.configure(background="white")
        self._window.title("Differential privacy")
        self._window.geometry("250x150")

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
                                    text="Open privacy window",
                                    command=lambda: onclick()
                                    )
        privacy_button.grid(column=0, row=1)

    def _build_combob_utility(self):
        def onclick(event):
            curr_val = combob_utilities.get()
            self._window.destroy()
            if curr_val == "DP Histogram (ε)":
                UtilityWindow(DPHistogram, "eps")
            elif curr_val == "DP Mean (δ)":
                UtilityWindow(DPMean, "delta")
            elif curr_val == "DP Mean (ε)":
                UtilityWindow(DPMean, "delta")
            else:
                raise ValueError("Unknown utility")

        utilities_frame = tk.Frame(self._window, background="white")

        combob_utilities = ttk.Combobox(utilities_frame, width=20)
        combob_utilities['values'] = ("DP Histogram (ε)", "DP Mean (ε)", "DP Mean (δ)")
        combob_utilities['state'] = 'readonly'
        combob_utilities.bind('<<ComboboxSelected>>', onclick)

        utilites_label = tk.Label(utilities_frame, text="Open privacy/utility trade-off window", bg="white")

        utilites_label.pack()
        combob_utilities.pack()

        utilities_frame.grid(column=0, row=2)

if __name__ == "__main__":
    main_window = MainWindow()
