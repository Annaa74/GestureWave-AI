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
                 fg=TEXT, bg=BG0).pack(pady=(0, 6))
        tk.Label(c, text="Control your computer with hand gestures",
                 font=("Segoe UI", 13), fg=MUTED, bg=BG0).pack(pady=(0, 36))

        # Divider
        tk.Frame(c, bg=BORDER, height=1, width=340).pack(pady=12)

        # Sign In button (primary)
        b1 = clr_btn(c, "🔑  Sign In", ACCENT, "white", on_signin,
                     font=("Segoe UI Semibold", 13), padx=60, pady=14)
        b1.pack(pady=(24, 12), ipadx=24)

        # Demo button
        if is_demo_locked():
            b2 = clr_btn(c, "🎮  Demo (Expired)", BG3, MUTED, lambda: None,
                         font=("Segoe UI Semibold", 12), padx=48, pady=12)
            b2.config(state="disabled")
        else:
            b2 = clr_btn(c, "🎮  Try Demo (5 min)", BG3, TEXT, on_demo,
                         font=("Segoe UI Semibold", 12), padx=48, pady=12)
        b2.pack(pady=(0, 10))

        if is_demo_locked():
            tk.Label(c, text="Demo already used on this system",
                     font=("Segoe UI", 10), fg=DANGER, bg=BG0).pack()
        else:
            tk.Label(c, text="Move + Click only · 5 minutes · One-time",
                     font=("Segoe UI", 10), fg=MUTED, bg=BG0).pack()

        # Footer
        tk.Label(c, text="v3.0", font=("Consolas", 10),
                 fg=MUTED, bg=BG0).pack(pady=(36, 0))

# Welcome screen initialization
