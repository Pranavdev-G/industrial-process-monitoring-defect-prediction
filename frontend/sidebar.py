import tkinter as tk
from dashboard import DashboardPage
from upload_page import UploadPage
from data_analysis_page import DataAnalysisPage

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
            text="MENU",
            bg="#1e293b",
            fg="#22d3ee",
            font=("Roboto", 18, "bold")
        )
        title.pack(pady=25)

        self.create_button("Upload Dataset", UploadPage)
        self.create_button("Dashboard", DashboardPage)
        self.create_button("Data Analysis", DataAnalysisPage)

    def create_button(self, text, page):

     box = tk.Frame(
        self,
        bg="#1e293b",
        highlightbackground="#22d3ee",
        highlightthickness=2
    )

     box.pack(pady=12, padx=12, fill="x")

     btn = tk.Button(
        box,
        text=text,
        font=("Roboto", 12),
        bg="#1e293b",
        fg="#f1f5f9",
        relief="flat",
        command=lambda: self.glow_and_switch(page, box)
    )

     btn.pack(fill="x", pady=8)

     def glow_and_switch(self, page, box):

      for i in range(3):

        box.config(highlightbackground="#38bdf8")
        box.update()
        self.after(80)

        box.config(highlightbackground="#22d3ee")
        box.update()
        self.after(80)

     self.controller.show_frame(page)
 
        # Hover effects
     btn.bind("<Enter>", lambda e: btn.config(bg="#38bdf8", fg="black"))
     btn.bind("<Leave>", lambda e: btn.config(bg="#1e293b", fg="#f1f5f9"))