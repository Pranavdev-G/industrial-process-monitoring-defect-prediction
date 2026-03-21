import tkinter as tk
from tkinter import ttk

import pandas as pd
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ── COLORS ─────────────────────────────────────────────
BG_DEEP = "#ffffff"        # WHITE dashboard
BG_CARD = "#f1f5f9"        # Light card
BG_HEADER = "#1e293b"      # Sidebar color match
ACCENT = "#22d3ee"         # Cyan
ACCENT2 = "#818cf8"
ACCENT3 = "#34d399"
WARN = "#fb923c"

TEXT_PRI = "#0f172a"
TEXT_MUT = "#64748b"


class DashboardPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_DEEP)

        self.controller = controller
        self.canvas_refs = []
        self.selected_column = None

        self.build_ui()

    # ── UI ─────────────────────────────────────────────
    def build_ui(self):

        # HEADER
        header = tk.Frame(self, bg=BG_HEADER, height=60)
        header.pack(fill="x")

        tk.Label(
            header,
            text="PROCESS MONITOR DASHBOARD",
            bg=BG_HEADER,
            fg=ACCENT,
            font=("Arial", 16, "bold")
        ).pack(side="left", padx=20, pady=15)

        self.status = tk.Label(
            header,
            text="No dataset loaded",
            bg=BG_HEADER,
            fg="#cbd5f5",
            font=("Arial", 10)
        )
        self.status.pack(side="right", padx=20)

        # CONTROLS
        controls_frame = tk.Frame(self, bg=BG_DEEP, height=50)
        controls_frame.pack(fill="x", pady=10)

        tk.Label(controls_frame, text="Select Column:", bg=BG_DEEP, fg=TEXT_PRI).pack(side="left", padx=20)
        self.column_var = tk.StringVar()
        self.column_dropdown = ttk.Combobox(controls_frame, textvariable=self.column_var, state="readonly")
        self.column_dropdown.pack(side="left", padx=10)
        self.column_dropdown.bind("<<ComboboxSelected>>", self.on_column_select)

        refresh_btn = tk.Button(
            controls_frame,
            text="Refresh",
            bg=ACCENT,
            fg=TEXT_PRI,
            command=self.auto_run
        )
        refresh_btn.pack(side="right", padx=20)

        # MAIN FRAME
        self.output_frame = tk.Frame(self, bg=BG_DEEP)
        self.output_frame.pack(fill="both", expand=True)

    # ── MAIN RUN ───────────────────────────────────────
    def auto_run(self):

        for w in self.output_frame.winfo_children():
            w.destroy()

        if self.controller.dataset is None:
            self.show_placeholder()
            return

        df = self.controller.dataset
        self.status.config(text="Dataset Loaded", fg=ACCENT3)

        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            tk.Label(self.output_frame, text="No numeric data found",
                     bg=BG_DEEP, fg=TEXT_PRI, font=("Arial", 14)).pack(pady=50)
            return

        # Update dropdown
        columns = numeric.columns.tolist()
        self.column_dropdown['values'] = columns
        if not self.selected_column or self.selected_column not in columns:
            # Auto-select highest variance
            variances = numeric.var()
            self.selected_column = variances.idxmax()
        self.column_var.set(self.selected_column)

        self.kpi_section(df, numeric)
        self.chart_section(numeric)

    def on_column_select(self, event):
        self.selected_column = self.column_var.get()
        self.auto_run()

    # ── PLACEHOLDER ────────────────────────────────────
    def show_placeholder(self):

        tk.Label(
            self.output_frame,
            text="Upload dataset from sidebar",
            fg=TEXT_MUT,
            bg=BG_DEEP,
            font=("Arial", 16)
        ).pack(pady=80)

    # ── KPI SECTION ────────────────────────────────────
    def kpi_section(self, df, numeric):

        frame = tk.Frame(self.output_frame, bg=BG_DEEP)
        frame.pack(fill="x", pady=15)

        data = [
            ("Records", df.shape[0], ACCENT),
            ("Variables", df.shape[1], ACCENT2),
            ("Numeric", len(numeric.columns), ACCENT3),
            ("Missing", df.isnull().sum().sum(), WARN)
        ]

        for text, val, color in data:

            card = tk.Frame(
                frame,
                bg=BG_CARD,
                padx=15,
                pady=10,
                highlightbackground=ACCENT,
                highlightthickness=1
            )
            card.pack(side="left", padx=15)

            tk.Label(card, text=text,
                     fg=TEXT_MUT, bg=BG_CARD, font=("Arial", 10)).pack()

            tk.Label(card, text=str(val),
                     fg=color, bg=BG_CARD,
                     font=("Arial", 18, "bold")).pack()

    # ── CHART SECTION ─────────────────────────────────
    def chart_section(self, numeric):

        col = self.selected_column
        data = numeric[col].dropna()

        frame = tk.Frame(self.output_frame, bg=BG_DEEP)
        frame.pack(fill="both", expand=True, pady=10)

        # Trend Chart
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)

        ax.plot(data, color=ACCENT, linewidth=2)
        ax.set_title(f"Trend - {col}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        self.canvas_refs.append(canvas)

        # Histogram
        fig2 = Figure(figsize=(4, 3))
        ax2 = fig2.add_subplot(111)

        ax2.hist(data, bins=25, color=ACCENT2, alpha=0.7, edgecolor='black')
        ax2.set_title("Distribution", fontsize=14, fontweight='bold')
        ax2.set_xlabel(col)
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)

        canvas2 = FigureCanvasTkAgg(fig2, frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        self.canvas_refs.append(canvas2)