import tkinter as tk
from frontend.dashboard import DashboardPage
from frontend.upload_page import UploadPage
from frontend.data_analysis_page import DataAnalysisPage
from frontend.spc_page import SPCPage
from frontend.prediction_page import PredictionPage
from frontend.clustering_page import ClusteringPage

class Sidebar(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(
            parent,
            bg="#1e293b",
            width=260,
            highlightbackground="#22d3ee",
            highlightthickness=2
        )

        self.controller = controller
        self.pack(side="left", fill="y")
        self.pack_propagate(False)

        title = tk.Label(
            self,
            text="NAVIGATION",
            bg="#1e293b",
            fg="#22d3ee",
            font=("Roboto", 18, "bold")
        )
        title.pack(pady=25)

        self.buttons = []
        self.create_button("Dashboard", DashboardPage)
        self.create_button("Upload Dataset", UploadPage)
        self.create_button("Data Analysis", DataAnalysisPage)
        self.create_button("SPC Monitoring", SPCPage)
        self.create_button("Prediction", PredictionPage)
        self.create_button("Clustering", ClusteringPage)

    def create_button(self, text, page):
        box = tk.Frame(
            self,
            bg="#1e293b",
            highlightbackground="#22d3ee",
            highlightthickness=2
        )
        box.pack(pady=8, padx=12, fill="x")

        btn = tk.Button(
            box,
            text=text,
            font=("Roboto", 12),
            bg="#1e293b",
            fg="#f1f5f9",
            relief="flat",
            command=lambda: self.select_page(page, box)
        )
        btn.pack(fill="x", pady=8)

        # Store reference
        self.buttons.append((btn, box, page))

        # Hover effects
        btn.bind("<Enter>", lambda e, b=box: self.on_hover(b))
        btn.bind("<Leave>", lambda e, b=box: self.on_leave(b))

    def select_page(self, page, selected_box):
        # Reset all buttons
        for btn, box, _ in self.buttons:
            box.config(highlightbackground="#22d3ee")
            btn.config(bg="#1e293b", fg="#f1f5f9")

        # Highlight selected
        selected_box.config(highlightbackground="#38bdf8")
        # Glow effect
        self.glow_effect(selected_box)

        self.controller.show_frame(page)

    def on_hover(self, box):
        if box.cget("highlightbackground") != "#38bdf8":  # Not selected
            box.config(highlightbackground="#38bdf8")

    def on_leave(self, box):
        if box.cget("highlightbackground") != "#38bdf8":  # Not selected
            box.config(highlightbackground="#22d3ee")

    def glow_effect(self, box):
        def glow():
            for _ in range(3):
                box.config(highlightbackground="#67e8f9")
                box.update()
                self.after(50)
                box.config(highlightbackground="#38bdf8")
                box.update()
                self.after(50)
        self.after(0, glow)