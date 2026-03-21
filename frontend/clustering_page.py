import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from analytics.clustering import perform_clustering

class ClusteringPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#ffffff")

        self.controller = controller
        self.canvas_refs = []

        title = tk.Label(
            self,
            text="Data Clustering",
            font=("Roboto", 24, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        title.pack(pady=20)

        # Controls
        controls_frame = tk.Frame(self, bg="#ffffff", height=50)
        controls_frame.pack(fill="x", pady=10)

        tk.Label(controls_frame, text="Number of Clusters:", bg="#ffffff", fg="#1e293b").pack(side="left", padx=20)
        self.n_clusters_var = tk.IntVar(value=3)
        self.n_clusters_spin = tk.Spinbox(controls_frame, from_=2, to=10, textvariable=self.n_clusters_var, width=5)
        self.n_clusters_spin.pack(side="left", padx=10)

        run_btn = tk.Button(
            controls_frame,
            text="Run Clustering",
            bg="#22d3ee",
            fg="#1e293b",
            command=self.run_clustering
        )
        run_btn.pack(side="right", padx=20)

        self.output_frame = tk.Frame(self, bg="#ffffff")
        self.output_frame.pack(fill="both", expand=True)

    def run_clustering(self):
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

        n_clusters = self.n_clusters_var.get()

        clusters, centroids, fig, info = perform_clustering(df, n_clusters)

        if clusters is None:
            msg = tk.Label(
                self.output_frame,
                text=info,
                bg="#ffffff",
                fg="#fb923c",
                font=("Arial", 12)
            )
            msg.pack(pady=20)
            return

        # Results
        result_label = tk.Label(
            self.output_frame,
            text=f"Clustering Results\n{info}",
            bg="#ffffff",
            fg="#22d3ee",
            font=("Roboto", 16, "bold")
        )
        result_label.pack(pady=10)

        # Cluster Plot
        if fig:
            canvas = FigureCanvasTkAgg(fig, self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)
            self.canvas_refs.append(canvas)

        # Cluster sizes
        from collections import Counter
        cluster_counts = Counter(clusters)
        sizes_text = "Cluster Sizes:\n" + "\n".join([f"Cluster {i}: {count} points" for i, count in cluster_counts.items()])

        sizes_label = tk.Label(
            self.output_frame,
            text=sizes_text,
            bg="#f8fafc",
            fg="#0f172a",
            font=("Courier", 10),
            justify="left",
            anchor="w"
        )
        sizes_label.pack(fill="x", padx=20, pady=10)