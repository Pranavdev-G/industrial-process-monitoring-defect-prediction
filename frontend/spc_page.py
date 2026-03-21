import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analytics.spc_charts import create_control_chart
from utils.data_loader import get_numeric_columns

class SPCPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#ffffff")

        self.controller = controller
        self.canvas_refs = []

        title = tk.Label(
            self,
            text="SPC Monitoring",
            font=("Roboto", 24, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        title.pack(pady=20)

        # Controls
        controls_frame = tk.Frame(self, bg="#ffffff", height=50)
        controls_frame.pack(fill="x", pady=10)

        tk.Label(controls_frame, text="Select Column:", bg="#ffffff", fg="#1e293b").pack(side="left", padx=20)
        self.column_var = tk.StringVar()
        self.column_dropdown = ttk.Combobox(controls_frame, textvariable=self.column_var, state="readonly")
        self.column_dropdown.pack(side="left", padx=10)

        run_btn = tk.Button(
            controls_frame,
            text="Generate Control Chart",
            bg="#22d3ee",
            fg="#1e293b",
            command=self.run_spc
        )
        run_btn.pack(side="right", padx=20)

        self.output_frame = tk.Frame(self, bg="#ffffff")
        self.output_frame.pack(fill="both", expand=True)

    def run_spc(self):
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

        col = self.column_var.get()
        if not col or col not in df.columns:
            msg = tk.Label(
                self.output_frame,
                text="Please select a valid numeric column.",
                bg="#ffffff",
                fg="#fb923c",
                font=("Arial", 12)
            )
            msg.pack(pady=20)
            return

        data = df[col].dropna()
        if len(data) < 2:
            msg = tk.Label(
                self.output_frame,
                text="Insufficient data for control chart.",
                bg="#ffffff",
                fg="#fb923c"
            )
            msg.pack(pady=20)
            return

        # Create control chart
        fig, ax = create_control_chart(data, title=f"Control Chart - {col}")

        canvas = FigureCanvasTkAgg(fig, self.output_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)
        self.canvas_refs.append(canvas)

        # Stats
        mean_val = data.mean()
        std_val = data.std()
        ucl = mean_val + 3 * std_val
        lcl = mean_val - 3 * std_val

        stats_text = f"""
Process Statistics:
Mean: {mean_val:.4f}
Std Dev: {std_val:.4f}
UCL (+3σ): {ucl:.4f}
LCL (-3σ): {lcl:.4f}
Total Points: {len(data)}
Outliers: {(data > ucl).sum() + (data < lcl).sum()}
        """

        stats_label = tk.Label(
            self.output_frame,
            text=stats_text,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Courier", 10),
            justify="left",
            anchor="w"
        )
        stats_label.pack(fill="x", padx=20, pady=10)

    def tkraise(self):
        super().tkraise()
        # Update dropdown when page is shown
        if self.controller.dataset is not None:
            numeric_cols = get_numeric_columns(self.controller.dataset)
            self.column_dropdown['values'] = numeric_cols
            if numeric_cols:
                self.column_var.set(numeric_cols[0])
