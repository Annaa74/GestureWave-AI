"""Welcome Screen — Demo + Sign In (iOS-inspired)"""
import tkinter as tk
from ui_theme import *


class WelcomeScreen(tk.Frame):
    def __init__(self, parent, on_demo, on_signin):
        super().__init__(parent, bg=BG0)
        self._build(on_demo, on_signin)

    def _build(self, on_demo, on_signin):
        # Center container
        c = tk.Frame(self, bg=BG0)
        c.place(relx=0.5, rely=0.5, anchor="center")

        # Logo
        tk.Label(c, text="≋", font=("Segoe UI", 56, "bold"),
                 fg=ACCENT, bg=BG0).pack()
        tk.Label(c, text="GestureWave AI",
                 font=("Segoe UI Semibold", 32),
