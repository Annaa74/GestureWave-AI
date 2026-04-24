"""Main Dashboard - Full-featured tracking interface"""
import tkinter as tk
from tkinter import ttk, messagebox
import time, threading
from ui_theme import *

class Dashboard(tk.Frame):
    GESTURES = [
        ("☝️", "Index Finger Up", "Move Cursor", ACCENT),
        ("🤏", "Thumb + Index Pinch", "Left Click", SUCCESS),
        ("⚡", "Quick Double Pinch", "Double Click", WARNING),
        ("🤟", "Three Fingers Up", "Right Click", DANGER),
        ("✌️", "Peace Sign", "Scroll Up / Down", CYAN),
        ("👍", "Thumbs Up", "Zoom In", ACCENT2),
        ("👎", "Thumbs Down", "Zoom Out", ACCENT2),
        ("✋", "Open Palm", "Pause / Resume", MUTED),
    ]

    def __init__(self, parent, engine, role, is_demo, on_logout, demo_gestures=None):
        super().__init__(parent, bg=BG0)
        self.engine = engine
        self.role = role
        self.is_demo = is_demo
        self.on_logout = on_logout
        self.demo_gestures = demo_gestures or set()
        self._running = False
        self._log_lines = []
        self._start_ts = 0
        self._demo_start = 0
        self._demo_dur = 300  # 5 min

        # Tk variables
        self._alpha = tk.DoubleVar(value=engine.Cfg.SMOOTH_ALPHA)
        self._dead = tk.IntVar(value=engine.Cfg.DEAD_ZONE)
        self._thresh = tk.IntVar(value=engine.Cfg.CLICK_THRESH)
        self._scroll = tk.IntVar(value=engine.Cfg.SCROLL_AMOUNT)
        self._cam = tk.IntVar(value=engine.Cfg.CAMERA_ID)

        self._build()
        self._tick()
        if is_demo:
            self._demo_start = time.time()

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] {msg}")
        if len(self._log_lines) > 100:
            self._log_lines.pop(0)

    # ── Build ──
    def _build(self):
        self._build_header()
        self._build_status()
        self._build_actions()
        self._build_tabs()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG1, height=56)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        # Logo
        lf = tk.Frame(hdr, bg=BG1)
        lf.pack(side="left", padx=16)
        tk.Label(lf, text="≋", font=("Segoe UI", 20, "bold"), fg=ACCENT, bg=BG1).pack(side="left")
        tk.Label(lf, text=" GestureWave ", font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG1).pack(side="left")
        tk.Label(lf, text="AI", font=("Segoe UI", 12, "bold"), fg=ACCENT2, bg=BG1).pack(side="left")

        # Right side - role + logout
        rf = tk.Frame(hdr, bg=BG1)
        rf.pack(side="right", padx=16)

        if self.is_demo:
            self._timer_lbl = tk.Label(rf, text="05:00", font=("Consolas", 11, "bold"), fg=WARNING, bg=BG1)
            self._timer_lbl.pack(side="left", padx=(0,12))
            tk.Label(rf, text="DEMO", font=("Segoe UI", 8, "bold"), fg="#000", bg=WARNING, padx=6, pady=1).pack(side="left", padx=(0,10))
        else:
            role_text = "Admin" if self.role == "admin" else "Pro"
            role_fg = SUCCESS if self.role == "admin" else ACCENT
            role_bg = "#062d1b" if self.role == "admin" else "#1e1e3a"
            tk.Label(rf, text=role_text, font=("Segoe UI", 8, "bold"), fg=role_fg, bg=role_bg, padx=8, pady=2).pack(side="left", padx=(0,10))

        clr_btn(rf, "Logout", BG3, MUTED, self.on_logout, font=SANS_SM, padx=12, pady=4).pack(side="left")

    def _build_status(self):
        sf = tk.Frame(self, bg=BG1, highlightbackground=BORDER, highlightthickness=1)
        sf.pack(fill="x", padx=16, pady=(10,0))

        left = tk.Frame(sf, bg=BG1); left.pack(side="left", padx=14, pady=10)
        right = tk.Frame(sf, bg=BG1); right.pack(side="right", padx=14, pady=10)

        self._dot = tk.Label(left, text="●", font=("Segoe UI", 16), fg=MUTED, bg=BG1)
        self._dot.pack(side="left", padx=(0,8))
        tf = tk.Frame(left, bg=BG1); tf.pack(side="left")
        self._st_big = tk.Label(tf, text="Ready", font=("Segoe UI", 12, "bold"), fg=TEXT, bg=BG1, anchor="w")
        self._st_big.pack(anchor="w")
        self._st_sub = tk.Label(tf, text="Press Start to begin tracking", font=SANS_SM, fg=MUTED, bg=BG1, anchor="w")
        self._st_sub.pack(anchor="w")

        self._clock = tk.Label(right, text="00:00:00", font=MONO, fg=MUTED, bg=BG1)
        self._clock.pack()
