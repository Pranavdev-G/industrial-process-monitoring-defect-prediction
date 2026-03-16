import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataAnalysisPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#0f172a")

        self.controller = controller

        title = tk.Label(
            self,
            text="Data Analysis",
            font=("Roboto", 24, "bold"),
            bg="#0f172a",
            fg="white"
        )
        title.pack(pady=20)

        run_btn = tk.Button(
            self,
            text="RUN ANALYSIS",
            font=("Roboto", 12),
            bg="#22d3ee",
            fg="black",
            command=self.run_analysis
        )
        run_btn.pack(pady=10)

        self.output_frame = tk.Frame(self, bg="#0f172a")
        self.output_frame.pack(fill="both", expand=True)

    def run_analysis(self):

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

        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            msg = tk.Label(
                self.output_frame,
                text="Dataset has no numeric columns.",
                bg="#0f172a",
                fg="white"
            )
            msg.pack(pady=20)
            return

        # Correlation Heatmap
        fig1, ax1 = plt.subplots(figsize=(6,4))
        sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax1)
        ax1.set_title("Correlation Heatmap")

        canvas1 = FigureCanvasTkAgg(fig1, self.output_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(pady=10)

        # Covariance Table
        cov = numeric.cov()

        cov_label = tk.Label(
            self.output_frame,
            text="Covariance Matrix",
            font=("Roboto", 16),
            bg="#0f172a",
            fg="#22d3ee"
        )
        cov_label.pack(pady=10)

        table = tk.Text(self.output_frame, height=10, width=80)
        table.pack()
        table.insert(tk.END, cov.to_string())

        # Histogram
        fig2, ax2 = plt.subplots(figsize=(6,4))
        numeric.iloc[:,0].hist(ax=ax2)
        ax2.set_title("Data Distribution")

        canvas2 = FigureCanvasTkAgg(fig2, self.output_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(pady=10)