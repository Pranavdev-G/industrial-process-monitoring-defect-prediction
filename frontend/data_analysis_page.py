import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analytics.descriptive_stats import get_descriptive_stats, get_correlation_matrix, get_covariance_matrix


class DataAnalysisPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#ffffff")

        self.controller = controller
        self.canvas_refs = []

        title = tk.Label(
            self,
            text="Data Analysis",
            font=("Roboto", 24, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        title.pack(pady=20)

        run_btn = tk.Button(
            self,
            text="RUN ANALYSIS",
            font=("Roboto", 12),
            bg="#22d3ee",
            fg="#1e293b",
            relief="flat",
            command=self.run_analysis
        )
        run_btn.pack(pady=10)

        self.output_frame = tk.Frame(self, bg="#ffffff")
        self.output_frame.pack(fill="both", expand=True)

    def run_analysis(self):

        for widget in self.output_frame.winfo_children():
            widget.destroy()
        self.canvas_refs = []

        df = self.controller.dataset

        if df is None:
            msg = tk.Label(
                self.output_frame,
                text="Please upload dataset first.",
                bg="#ffffff",
                fg="#64748b",
                font=("Arial", 12)
            )
            msg.pack(pady=20)
            return

        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            msg = tk.Label(
                self.output_frame,
                text="Dataset has no numeric columns.",
                bg="#ffffff",
                fg="#64748b"
            )
            msg.pack(pady=20)
            return

        # Descriptive Statistics
        stats = get_descriptive_stats(df)
        if not stats.empty:
            stats_label = tk.Label(
                self.output_frame,
                text="Descriptive Statistics",
                font=("Roboto", 16, "bold"),
                bg="#ffffff",
                fg="#22d3ee"
            )
            stats_label.pack(pady=10)

            text = tk.Text(
                self.output_frame,
                height=8,
                bg="#f8fafc",
                fg="#0f172a",
                font=("Courier", 10)
            )
            text.pack(fill="x", padx=20, pady=5)
            text.insert(tk.END, stats.to_string())
            text.config(state="disabled")

        # Correlation Heatmap
        corr = get_correlation_matrix(df)
        if not corr.empty:
            fig1, ax1 = plt.subplots(figsize=(6,4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax1, fmt=".2f",
                       cbar_kws={'shrink': 0.8})
            ax1.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')

            canvas1 = FigureCanvasTkAgg(fig1, self.output_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(pady=10)
            self.canvas_refs.append(canvas1)

        # Covariance Matrix
        cov = get_covariance_matrix(df)
        if not cov.empty:
            cov_label = tk.Label(
                self.output_frame,
                text="Covariance Matrix",
                font=("Roboto", 16, "bold"),
                bg="#ffffff",
                fg="#22d3ee"
            )
            cov_label.pack(pady=10)

            text2 = tk.Text(
                self.output_frame,
                height=8,
                bg="#f8fafc",
                fg="#0f172a",
                font=("Courier", 10)
            )
            text2.pack(fill="x", padx=20, pady=5)
            text2.insert(tk.END, cov.round(4).to_string())
            text2.config(state="disabled")