"""
Overview
--------

Gui for picking data and appending it to the overvation file.
"""
import tkinter as tk
import h5py
import numpy as np
from data_handler import get_TOD_fset

class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.btn = tk.Button(self, text="Submit", command=self.btn_handler)
        self.config(bg="lightgrey")
        self.title("HDF5 Observation Data")
        self.rawchoices = get_TOD_fset()
        self.varRawChoices = tk.StringVar(value=self.rawchoices)
        self.rawlist = tk.Listbox(self, listvariable=self.varRawChoices, selectmode="multiple", height=15, width=60)

        self.label1 = tk.Label(self, text="Please select the hdf5 files you wish to merge from today.")
        self.label2 = tk.Label(self, text="HDF Files")

        self.label1.grid(row=1, column=1, padx=10, pady=10)
        self.label2.grid(row=2, column=1, padx=10, pady=10)
        self.btn.grid(row=4, column=1, padx=10, pady=10)
        self.rawlist.grid(row=3, column=1, padx=10, pady=10)

    def btn_handler(self):
        files = []
        try:
            indx = self.rawlist.curselection()
            files = [self.rawlist.get(i) for i in indx]
        except Exception as e:
            pass
        print(files)
        self.destroy()

def run():
    gui = Gui()
    try:
        gui.mainloop()
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    run()