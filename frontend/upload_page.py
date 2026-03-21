import tkinter as tk
from tkinter import filedialog, messagebox
from utils.data_loader import load_and_clean_data, validate_dataset

class UploadPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#ffffff")

        self.controller = controller

        title = tk.Label(
            self,
            text="Upload Dataset",
            font=("Roboto", 22, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        title.pack(pady=30)

        upload_btn = tk.Button(
            self,
            text="Select CSV File",
            font=("Roboto", 12),
            bg="#22d3ee",
            fg="#1e293b",
            relief="flat",
            command=self.load_file
        )
        upload_btn.pack(pady=10)

        self.status = tk.Label(
            self,
            text="No dataset loaded",
            bg="#ffffff",
            fg="#64748b",
            font=("Arial", 11)
        )
        self.status.pack(pady=20)

        self.info_label = tk.Label(
            self,
            text="",
            bg="#ffffff",
            fg="#64748b",
            font=("Arial", 10),
            justify="left"
        )
        self.info_label.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )

        if not file_path:
            return

        try:
            df = load_and_clean_data(file_path)
            valid, msg = validate_dataset(df)

            if not valid:
                messagebox.showerror("Invalid Dataset", msg)
                return

            self.controller.dataset = df

            self.status.config(
                text=f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns",
                fg="#22d3ee"
            )

            # Show data info
            numeric_cols = len(df.select_dtypes(include='number').columns)
            missing = df.isnull().sum().sum()
            info_text = f"Numeric columns: {numeric_cols}\nMissing values: {missing}\nData types cleaned automatically."
            self.info_label.config(text=info_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.status.config(text="Failed to load dataset", fg="#fb923c")