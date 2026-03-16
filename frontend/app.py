import tkinter as tk
from dashboard import DashboardPage
from upload_page import UploadPage
from data_analysis_page import DataAnalysisPage
from sidebar import Sidebar


class App(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Industrial Process Monitoring Dashboard")
        self.state("zoomed")

        self.dataset = None

        container = tk.Frame(self, bg="#0f172a")
        container.pack(side="right", fill="both", expand=True)

        self.frames = {}

        for F in (DashboardPage, UploadPage, DataAnalysisPage):
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