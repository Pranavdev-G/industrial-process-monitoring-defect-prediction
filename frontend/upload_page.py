import tkinter as tk
from tkinter import filedialog
import pandas as pd


class UploadPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#0f172a")

        self.controller = controller

        title = tk.Label(
            self,
            text="Upload Dataset",
            font=("Roboto", 22),
            bg="#0f172a",
            fg="white"
        )
        title.pack(pady=30)

        upload_btn = tk.Button(
            self,
            text="Select CSV File",
            font=("Roboto", 12),
            bg="#00e5ff",
            command=self.load_file
        )
        upload_btn.pack()

        self.status = tk.Label(
            self,
            text="No dataset loaded",
            bg="#0f172a",
            fg="white",
            font=("Arial", 11)
        )
        self.status.pack(pady=20)

    def load_file(self):

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if file_path:
            df = pd.read_csv(file_path)

            self.controller.dataset = df

            self.status.config(
                text=f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns"
            )