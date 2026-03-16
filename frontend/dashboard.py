import tkinter as tk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DashboardPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#0f172a")

        self.controller = controller

        title = tk.Label(
            self,
            text="Dashboard",
            font=("Roboto", 24, "bold"),
            bg="#0f172a",
            fg="white"
        )
        title.pack(pady=20)

        self.output_frame = tk.Frame(self, bg="#0f172a")
        self.output_frame.pack(fill="both", expand=True)

    def auto_run(self):

     if self.controller.dataset is not None:
        self.run_analysis()

        for widget in self.output_frame.winfo_children():
            widget.destroy()

        df = self.controller.dataset

        if df is None:
            msg = tk.Label(
                self.output_frame,
                text="Please upload dataset first.",
                bg="#0f172a",
                fg="white",
                font=("Arial", 12)
            )
            msg.pack(pady=20)
            return

        rows = df.shape[0]
        cols = df.shape[1]
        missing = df.isnull().sum().sum()

        kpi_frame = tk.Frame(self.output_frame, bg="#0f172a")
        kpi_frame.pack(pady=10)

        self.kpi_card(kpi_frame, "Total Records", rows)
        self.kpi_card(kpi_frame, "Variables", cols)
        self.kpi_card(kpi_frame, "Missing Values", missing)

        numeric = df.select_dtypes(include="number")

        if not numeric.empty:

            fig, ax = plt.subplots(figsize=(6,4))

            sns.lineplot(data=numeric.iloc[:,0], ax=ax)

            ax.set_title("Trend Chart")

            canvas = FigureCanvasTkAgg(fig, self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        stats = df.describe()

        table = tk.Text(self.output_frame, height=10, width=80)
        table.pack(pady=20)

        table.insert(tk.END, stats.to_string())

    def kpi_card(self, parent, title, value):

        frame = tk.Frame(parent, bg="#334155", width=150, height=80)
        frame.pack(side="left", padx=10)

        label = tk.Label(
            frame,
            text=title,
            bg="#334155",
            fg="#00e5ff",
            font=("Roboto", 10)
        )
        label.pack()

        val = tk.Label(
            frame,
            text=value,
            bg="#334155",
            fg="#f1f5f9",
            font=("Roboto", 16, "bold")
        )
        val.pack()