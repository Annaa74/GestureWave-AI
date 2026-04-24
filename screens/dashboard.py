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

    def _build_actions(self):
        bf = tk.Frame(self, bg=BG0); bf.pack(fill="x", padx=16, pady=8)
        self._btn_start = clr_btn(bf, "▶  Start Tracking", ACCENT, "white", self._start, padx=24, pady=10)
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0,6))
        self._btn_stop = clr_btn(bf, "■  Stop", BG2, MUTED, self._stop, padx=20, pady=10)
        self._btn_stop.config(state="disabled"); self._btn_stop.pack(side="right")

    def _build_tabs(self):
        s = ttk.Style(self)
        s.theme_use("default")
        s.configure("D.TNotebook", background=BG0, borderwidth=0)
        s.configure("D.TNotebook.Tab", background=BG2, foreground=MUTED, font=SANS_B, padding=[14,6], borderwidth=0)
        s.map("D.TNotebook.Tab", background=[("selected", BG1)], foreground=[("selected", TEXT)])

        nb = ttk.Notebook(self, style="D.TNotebook")
        nb.pack(fill="both", expand=True, padx=16, pady=(4,12))

        t1 = tk.Frame(nb, bg=BG1); t2 = tk.Frame(nb, bg=BG1); t3 = tk.Frame(nb, bg=BG1)
        nb.add(t1, text="  Gestures  "); nb.add(t2, text="  Settings  "); nb.add(t3, text="  Live Log  ")
        self._build_gestures(t1); self._build_settings(t2); self._build_logview(t3)

    def _build_gestures(self, t):
        lbl(t, "GESTURE REFERENCE", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(12,6))
        for emoji, name, action, color in self.GESTURES:
            locked = self.is_demo and name not in ("Index Finger Up", "Thumb + Index Pinch")
            row = tk.Frame(t, bg=BG2 if not locked else BG0)
            row.pack(fill="x", padx=12, pady=2); row.pack_propagate(False); row.configure(height=36)
            fg_c = MUTED if locked else TEXT
            tk.Label(row, text=emoji, font=("Segoe UI", 13), bg=row["bg"], width=3).pack(side="left", padx=(8,4))
            tk.Label(row, text=name, font=SANS_B, fg=fg_c, bg=row["bg"], anchor="w", width=22).pack(side="left")
            if locked:
                tk.Label(row, text="🔒", font=SANS_SM, fg=MUTED, bg=row["bg"]).pack(side="right", padx=8)
            else:
                tk.Label(row, text=action, font=SANS, fg=color, bg=row["bg"], anchor="e").pack(side="right", padx=10)

        tip = tk.Frame(t, bg=BG1); tip.pack(fill="x", padx=12, pady=(8,0))
        lbl(tip, "💡 Move cursor to screen corner for emergency stop | ESC to exit camera", font=("Segoe UI", 8), fg=MUTED, bg=BG1).pack(anchor="w", padx=4)

    def _build_settings(self, t):
        if self.is_demo:
            lbl(t, "⚠ Settings locked in Demo mode. Sign in to customize.", font=SANS, fg=WARNING, bg=BG1).pack(padx=16, pady=30)
            return

        lbl(t, "TRACKING", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(12,4))
        self._slider(t, "Cursor Smoothing", self._alpha, 0.05, 1.0, "{:.2f}")
        self._slider(t, "Dead Zone (px)", self._dead, 0, 20, "{:.0f}")
        self._slider(t, "Click Sensitivity", self._thresh, 10, 60, "{:.0f}")
        self._slider(t, "Scroll Speed", self._scroll, 5, 60, "{:.0f}")

        tk.Frame(t, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)
        lbl(t, "CAMERA", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(4,6))
        cf = tk.Frame(t, bg=BG1); cf.pack(fill="x", padx=16)
        lbl(cf, "Camera Index", font=SANS_B, bg=BG1).pack(side="left")
        for i in range(3):
            tk.Radiobutton(cf, text=f" {i}", variable=self._cam, value=i, font=SANS, fg=TEXT, bg=BG1, selectcolor=ACCENT, activebackground=BG1, relief="flat").pack(side="left", padx=6)

        tk.Frame(t, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)
        clr_btn(t, "✓  Apply Settings", ACCENT, "white", self._apply, padx=20, pady=8).pack(anchor="w", padx=16)

    def _slider(self, parent, text, var, lo, hi, fmt):
        f = tk.Frame(parent, bg=BG1); f.pack(fill="x", padx=16, pady=4)
        h = tk.Frame(f, bg=BG1); h.pack(fill="x")
        lbl(h, text, font=SANS_B, bg=BG1).pack(side="left")
        vl = lbl(h, fmt.format(var.get()), font=MONO, fg=ACCENT, bg=BG1); vl.pack(side="right")
        sc = ttk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=var,
                       command=lambda v: vl.config(text=fmt.format(float(v))))
        sc.pack(fill="x", pady=(2,0))

    def _build_logview(self, t):
        tb = tk.Frame(t, bg=BG1); tb.pack(fill="x", padx=12, pady=(10,4))
        lbl(tb, "LIVE LOG", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG1).pack(side="left")
        clr_btn(tb, "Clear", BG2, MUTED, self._clear_log, font=SANS_SM, padx=10, pady=3).pack(side="right")
        self._log_txt = tk.Text(t, bg=BG2, fg="#a3e635", font=MONO, relief="flat", bd=0, state="disabled", wrap="word", cursor="arrow")
        sb = tk.Scrollbar(t, command=self._log_txt.yview, bg=BG2, troughcolor=BG2, relief="flat", bd=0)
        self._log_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0,4), pady=4)
        self._log_txt.pack(fill="both", expand=True, padx=(12,0), pady=(0,8))

    def _clear_log(self):
        self._log_lines.clear(); self._refresh_log()

    def _refresh_log(self):
        self._log_txt.config(state="normal"); self._log_txt.delete("1.0", "end")
        self._log_txt.insert("end", "\n".join(reversed(self._log_lines)))
        self._log_txt.config(state="disabled")

    # ── Engine ──
    def _start(self):
        if self._running: return
        self._running = True; self._start_ts = time.perf_counter()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal", bg=DANGER, fg="white")
        self._set_status("Starting…", ACCENT, "Opening camera…")
        self._log("Tracking started")
        self._eng_th = threading.Thread(target=self._worker, daemon=True)
        self._eng_th.start()

    def _worker(self):
        try:
            self._set_status("Tracking", SUCCESS, "Hand detection running")
            self.engine.run()
        except Exception as e:
            self._log(f"Error: {e}"); self._set_status("Error", DANGER, str(e))
        finally:
            self._running = False
            self.after(0, lambda: (self._btn_start.config(state="normal"),
                                    self._btn_stop.config(state="disabled", bg=BG2, fg=MUTED)))
            self._log("Tracking stopped"); self._set_status("Stopped", MUTED, "Press Start again")

    def _stop(self):
        self.engine.stop_flag = True; self._log("Stop requested")

    def _apply(self):
        if self.role not in ("admin", "standard"):
            messagebox.showwarning("Denied", "Sign in to change settings."); return
        self.engine.Cfg.SMOOTH_ALPHA = round(self._alpha.get(), 2)
        self.engine.Cfg.DEAD_ZONE = int(self._dead.get())
        self.engine.Cfg.CLICK_THRESH = int(self._thresh.get())
        self.engine.Cfg.SCROLL_AMOUNT = int(self._scroll.get())
