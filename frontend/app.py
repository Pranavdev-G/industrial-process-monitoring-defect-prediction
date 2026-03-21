import os
import sys

# Ensure root project path is on sys.path so utils/analytics imports work from frontend script location
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import tkinter as tk
from frontend.dashboard import DashboardPage
from frontend.upload_page import UploadPage
from frontend.data_analysis_page import DataAnalysisPage
from frontend.spc_page import SPCPage
from frontend.prediction_page import PredictionPage
from frontend.clustering_page import ClusteringPage
from frontend.sidebar import Sidebar


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Industrial Process Monitoring Dashboard")
        self.state("zoomed")

        self.dataset = None

        container = tk.Frame(self, bg="#ffffff")
        container.pack(side="right", fill="both", expand=True)

        self.frames = {}

        for F in (DashboardPage, UploadPage, DataAnalysisPage, SPCPage, PredictionPage, ClusteringPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        Sidebar(self, self)

        self.show_frame(DashboardPage)

    def show_frame(self, page):
        frame = self.frames[page]
        frame.tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()