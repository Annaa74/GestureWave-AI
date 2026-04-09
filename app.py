"""
GestureWave AI — Desktop Launcher v2.1
Premium dark-mode GUI control panel.
"""
import sys, os, threading, time, tkinter as tk
from tkinter import ttk, messagebox

# ── Dependency check ──────────────────────────────────────────────────────────
REQUIRED = [("cv2","opencv-python"),("mediapipe","mediapipe"),
            ("pyautogui","pyautogui"),("numpy","numpy")]
missing = [pkg for imp,pkg in REQUIRED if not __import__("importlib").util.find_spec(imp)]
if missing:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("GestureWave AI – Missing Packages",
        "Please install:\n\n  pip install " + " ".join(missing))
    sys.exit(1)

import main as engine   # our gesture engine

# ── Palette ───────────────────────────────────────────────────────────────────
BG0     = "#0a0a0a"   # deepest background
BG1     = "#111111"   # card / panel
BG2     = "#1a1a1a"   # input / row
BORDER  = "#2a2a2a"
ACCENT  = "#2563eb"   # blue
ACCENT2 = "#7c3aed"   # violet
SUCCESS = "#16a34a"
DANGER  = "#dc2626"
WARNING = "#d97706"
TEXT    = "#f4f4f5"
MUTED   = "#71717a"
MONO    = ("Courier New", 9)
SANS    = ("Segoe UI", 9)
SANS_B  = ("Segoe UI", 9, "bold")
TITLE_F = ("Segoe UI", 11, "bold")

# ── Helpers ───────────────────────────────────────────────────────────────────
def clr_btn(parent, text, bg, fg, cmd, tooltip=None, **kw):
    btn = tk.Button(parent, text=text, bg=bg, fg=fg,
                     activebackground=bg, activeforeground=fg,
                     font=SANS_B, relief="flat", bd=0, cursor="hand2",
                     command=cmd, **kw)
    if tooltip:
        add_tooltip(btn, tooltip)
    return btn

def label(parent, text, font=SANS, fg=TEXT, bg=BG1, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

def sep(parent, bg=BORDER, pad=8):
    tk.Frame(parent, bg=bg, height=1).pack(fill="x", padx=20, pady=pad)

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tw = None
        self._id = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)

    def enter(self, event=None):
        self.unschedule()
        self._id = self.widget.after(400, self.show)

    def leave(self, event=None):
        self.unschedule()
        self.hide()

    def unschedule(self):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None

    def show(self, event=None):
        self.hide()
        x = self.widget.winfo_rootx() + (self.widget.winfo_width() // 2)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        # Fix for some window managers holding focus
        self.tw.attributes("-topmost", True)
        self.tw.wm_geometry(f"+{x}+{y}")
        
        lbl = tk.Label(self.tw, text=self.text, justify="left",
                       bg=BG2, fg=TEXT, font=("Segoe UI", 8),
                       relief="solid", bd=1, highlightbackground=BORDER)
        lbl.pack(ipadx=6, ipady=3)

    def hide(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None

def add_tooltip(widget, text):
    Tooltip(widget, text)


# ── Log buffer ────────────────────────────────────────────────────────────────
LOG_MAX = 120

class GestureLog:
    def __init__(self): self._lines = []
    def add(self, msg):
        ts  = time.strftime("%H:%M:%S")
        self._lines.append(f"[{ts}]  {msg}")
        if len(self._lines) > LOG_MAX:
            self._lines.pop(0)
    def all(self): return "\n".join(reversed(self._lines))

# ── Main App ──────────────────────────────────────────────────────────────────
class GestureWaveApp(tk.Tk):

    W, H = 520, 640

    def __init__(self):
        super().__init__()
        self.title("GestureWave AI")
        self.geometry(f"{self.W}x{self.H}")
        self.resizable(False, False)
        self.configure(bg=BG0)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._running   = False
        self._eng_th    = None
        self._log       = GestureLog()

        # Live vars
        self._alpha_var  = tk.DoubleVar(value=engine.Cfg.SMOOTH_ALPHA)
        self._dead_var   = tk.IntVar  (value=engine.Cfg.DEAD_ZONE)
        self._thresh_var = tk.IntVar  (value=engine.Cfg.CLICK_THRESH)
        self._scroll_var = tk.IntVar  (value=engine.Cfg.SCROLL_AMOUNT)
        self._cam_var    = tk.IntVar  (value=engine.Cfg.CAMERA_ID)

        self._build()
        self._tick()   # start live clock / status polling

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        # ── Header bar ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG1, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        logo_f = tk.Frame(hdr, bg=BG1)
        logo_f.place(relx=0, rely=0.5, anchor="w", x=20)
        tk.Label(logo_f, text="≋", font=("Segoe UI", 22, "bold"),
                 fg=ACCENT, bg=BG1).pack(side="left")
        tk.Label(logo_f, text="  GestureWave", font=("Segoe UI", 13, "bold"),
                 fg=TEXT, bg=BG1).pack(side="left")
        tk.Label(logo_f, text="AI", font=("Segoe UI", 13, "bold"),
                 fg=ACCENT2, bg=BG1).pack(side="left")

        self._ver_lbl = tk.Label(hdr, text="v2.1", font=MONO,
                                  fg=MUTED, bg=BG1)
        self._ver_lbl.place(relx=1, rely=0.5, anchor="e", x=-16)

        # ── Status card ─────────────────────────────────────────────────────
        sc = tk.Frame(self, bg=BG1, relief="flat")
        sc.pack(fill="x", padx=16, pady=(12, 0))

        left  = tk.Frame(sc, bg=BG1)
        left.pack(side="left", padx=14, pady=12)
        right = tk.Frame(sc, bg=BG1)
        right.pack(side="right", padx=14, pady=12)

        self._dot = tk.Label(left, text="●", font=("Segoe UI", 18),
                              fg=MUTED, bg=BG1)
        self._dot.pack(side="left", padx=(0, 10))
        txt_f = tk.Frame(left, bg=BG1)
        txt_f.pack(side="left")
        self._status_big = tk.Label(txt_f, text="Ready", font=("Segoe UI", 12, "bold"),
                                     fg=TEXT, bg=BG1, anchor="w")
        self._status_big.pack(anchor="w")
        self._status_sub = tk.Label(txt_f, text="Press Start to begin hand tracking",
                                     font=SANS, fg=MUTED, bg=BG1, anchor="w")
        self._status_sub.pack(anchor="w")

        self._clock = tk.Label(right, text="00:00:00", font=MONO,
                                fg=MUTED, bg=BG1)
        self._clock.pack()

        # ── Action buttons ──────────────────────────────────────────────────
        bf = tk.Frame(self, bg=BG0)
        bf.pack(fill="x", padx=16, pady=10)

        self._btn_start = clr_btn(bf, "▶   Start Tracking", ACCENT, "white",
                                   self._start, tooltip="Initialize MediaPipe and start hand tracking",
                                   padx=24, pady=10)
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._btn_stop = clr_btn(bf, "■  Stop", BG2, MUTED,
                                   self._stop, tooltip="Stop hand tracking and release camera",
                                   padx=20, pady=10, state="disabled")
        self._btn_stop.pack(side="right")

        # ── Notebook tabs ────────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Dark.TNotebook",        background=BG0, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",    background=BG2, foreground=MUTED,
                         font=SANS_B, padding=[14, 6], borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG1)],
                  foreground=[("selected", TEXT)])

        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        self._tab_gestures = tk.Frame(nb, bg=BG1)
        self._tab_settings  = tk.Frame(nb, bg=BG1)
        self._tab_log       = tk.Frame(nb, bg=BG1)

        nb.add(self._tab_gestures, text="  Gestures  ")
        nb.add(self._tab_settings,  text="  Settings  ")
        nb.add(self._tab_log,       text="  Live Log  ")

        self._build_gestures_tab()
        self._build_settings_tab()
        self._build_log_tab()

    # ── Tab: Gestures ─────────────────────────────────────────────────────────
    GESTURES = [
        ("☝",  "Index finger",            "Move cursor",      ACCENT),
        ("🤏", "Index + Thumb pinch",      "Left click",       SUCCESS),
        ("⚡",  "Quick double pinch",      "Double click",     WARNING),
        ("🤟", "Three fingers up",         "Right click",      DANGER),
        ("✌",  "Peace sign up / down",    "Scroll",           "#06b6d4"),
        ("👍", "Thumbs up / down",         "Zoom in / out",    "#8b5cf6"),
        ("✋", "Open palm",               "Pause / Resume",   MUTED),
    ]

    def _build_gestures_tab(self):
        t = self._tab_gestures
        label(t, "GESTURE REFERENCE", font=("Segoe UI", 8, "bold"),
               fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(14, 8))

        for emoji, name, action, color in self.GESTURES:
            row = tk.Frame(t, bg=BG2, cursor="arrow")
            row.pack(fill="x", padx=12, pady=2)
            row.pack_propagate(False)
            row.configure(height=38)

            tk.Label(row, text=emoji, font=("Segoe UI", 14),
                     bg=BG2, width=3).pack(side="left", padx=(10, 6))
            tk.Label(row, text=name, font=SANS_B, fg=TEXT,
                     bg=BG2, anchor="w", width=24).pack(side="left")
            tk.Label(row, text=action, font=SANS, fg=color,
                     bg=BG2, anchor="e").pack(side="right", padx=12)

        tip = tk.Frame(t, bg=BG1)
        tip.pack(fill="x", padx=12, pady=(10, 0))
        label(tip, "💡  Move cursor to any screen corner to emergency-stop (PyAutoGUI failsafe)",
               font=("Segoe UI", 8), fg=MUTED, bg=BG1).pack(anchor="w", padx=4)
        label(tip, "     Press ESC in the camera window to exit cleanly.",
               font=("Segoe UI", 8), fg=MUTED, bg=BG1).pack(anchor="w", padx=4)

    # ── Tab: Settings ─────────────────────────────────────────────────────────
    def _build_settings_tab(self):
        t = self._tab_settings

        def slider_row(parent, text, var, from_, to, fmt="{:.2f}", resolution=0.01):
            f = tk.Frame(parent, bg=BG1)
            f.pack(fill="x", padx=16, pady=6)
            hdr = tk.Frame(f, bg=BG1)
            hdr.pack(fill="x")
            label(hdr, text, font=SANS_B, bg=BG1).pack(side="left")
            val_lbl = label(hdr, fmt.format(var.get()), font=MONO, fg=ACCENT, bg=BG1)
            val_lbl.pack(side="right")

            def on_change(v):
                try:   val_lbl.config(text=fmt.format(float(v)))
                except: pass

            s = ttk.Scale(f, from_=from_, to=to, orient="horizontal",
                          variable=var, command=on_change)
            style = ttk.Style()
            style.configure("Dark.Horizontal.TScale", background=BG1, troughcolor=BG2,
                             sliderthickness=14, sliderlength=18)
            s.configure(style="Dark.Horizontal.TScale")
            s.pack(fill="x", pady=(4, 0))
            return s

        label(t, "TRACKING SETTINGS", font=("Segoe UI", 8, "bold"),
               fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(14, 4))

        slider_row(t, "Cursor Smoothing  (lower = smoother, higher = faster)",
                   self._alpha_var, 0.05, 1.0, "{:.2f}")
        slider_row(t, "Dead Zone  (pixels — suppresses tremor)",
                   self._dead_var, 0, 20, "{:.0f}", 1)
        slider_row(t, "Click Sensitivity  (pinch distance threshold)",
                   self._thresh_var, 10, 60, "{:.0f}", 1)
        slider_row(t, "Scroll Speed",
                   self._scroll_var, 5, 60, "{:.0f}", 1)

        sep(t, pad=6)
        label(t, "CAMERA", font=("Segoe UI", 8, "bold"),
               fg=MUTED, bg=BG1).pack(anchor="w", padx=16, pady=(4, 6))

        cam_f = tk.Frame(t, bg=BG1)
        cam_f.pack(fill="x", padx=16)
        label(cam_f, "Camera Index", font=SANS_B, bg=BG1).pack(side="left")
        for i in range(3):
            tk.Radiobutton(cam_f, text=f"  {i}", variable=self._cam_var, value=i,
                           font=SANS, fg=TEXT, bg=BG1, selectcolor=ACCENT,
                           activebackground=BG1, relief="flat").pack(side="left", padx=6)

        sep(t, pad=6)
        apply_btn = clr_btn(t, "✓  Apply Settings", ACCENT, "white",
                             self._apply_settings, tooltip="Save runtime settings (applies on next Start)",
                             padx=20, pady=8)
        apply_btn.pack(anchor="w", padx=16, pady=(0, 8))

        label(t, "Settings take effect on next Start.", font=("Segoe UI", 8),
               fg=MUTED, bg=BG1).pack(anchor="w", padx=16)

    def _apply_settings(self):
        engine.Cfg.SMOOTH_ALPHA  = round(self._alpha_var.get(),  2)
        engine.Cfg.DEAD_ZONE     = int(self._dead_var.get())
        engine.Cfg.CLICK_THRESH  = int(self._thresh_var.get())
        engine.Cfg.SCROLL_AMOUNT = int(self._scroll_var.get())
        engine.Cfg.CAMERA_ID     = int(self._cam_var.get())
        self._log.add(f"Settings applied — alpha={engine.Cfg.SMOOTH_ALPHA}, "
                      f"dead={engine.Cfg.DEAD_ZONE}px, thresh={engine.Cfg.CLICK_THRESH}")
        self._status("Settings applied ✓", "#22c55e", "Restart tracking to use new values.")

    # ── Tab: Log ─────────────────────────────────────────────────────────────
    def _build_log_tab(self):
        t = self._tab_log
        toolbar = tk.Frame(t, bg=BG1)
        toolbar.pack(fill="x", padx=12, pady=(10, 4))
        label(toolbar, "LIVE LOG", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=BG1).pack(side="left")
        clr_btn(toolbar, "Clear", BG2, MUTED, self._clear_log, tooltip="Wipe session logs history",
                padx=10, pady=3).pack(side="right")

        self._log_txt = tk.Text(t, bg=BG2, fg="#a3e635", font=MONO,
                                 relief="flat", bd=0, state="disabled",
                                 wrap="word", cursor="arrow",
                                 insertbackground=ACCENT, selectbackground=ACCENT)
        sb = tk.Scrollbar(t, command=self._log_txt.yview, bg=BG2, troughcolor=BG2,
                          relief="flat", bd=0)
        self._log_txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 4), pady=4)
        self._log_txt.pack(fill="both", expand=True, padx=(12, 0), pady=(0, 8))

    def _clear_log(self):
        self._log._lines.clear()
        self._refresh_log()

    def _refresh_log(self):
        self._log_txt.config(state="normal")
        self._log_txt.delete("1.0", "end")
        self._log_txt.insert("end", self._log.all())
        self._log_txt.config(state="disabled")

    # ── Engine control ────────────────────────────────────────────────────────
    def _start(self):
        if self._running: return
        self._running   = True
        self._start_ts  = time.perf_counter()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal", bg=DANGER, fg="white")
        self._status("Starting…", "#3b82f6", "Opening camera and loading MediaPipe…")
        self._log.add("Tracking started")
        self._eng_th = threading.Thread(target=self._engine_worker, daemon=True)
        self._eng_th.start()

    def _engine_worker(self):
        try:
            def on_status(msg):
                self._log.add(msg)
                self._status("Running", SUCCESS,  msg)
            engine.Cfg.STATUS_CB = on_status
            self._status("Tracking active", SUCCESS, "Hand detection running — camera window open")
            engine.run()
        except Exception as e:
            self._log.add(f"Error: {e}")
            self._status("Error", DANGER, str(e))
        finally:
            self._running = False
            self.after(0, self._reset_btns)
            self._log.add("Tracking stopped")
            self._status("Stopped", MUTED, "Press Start to begin a new session.")

    def _stop(self):
        engine.stop_flag = True
        self._log.add("Stop requested")

    def _reset_btns(self):
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled", bg=BG2, fg=MUTED)

    # ── Status helpers ────────────────────────────────────────────────────────
    def _status(self, title, dot_color, subtitle=""):
        def _do():
            self._status_big.config(text=title)
            self._status_sub.config(text=subtitle)
            self._dot.config(fg=dot_color)
        self.after(0, _do)

    # ── Clock / log ticker ────────────────────────────────────────────────────
    def _tick(self):
        if self._running:
            elapsed = int(time.perf_counter() - getattr(self, "_start_ts", time.perf_counter()))
            h, r = divmod(elapsed, 3600)
            m, s = divmod(r,       60)
            self._clock.config(text=f"{h:02}:{m:02}:{s:02}", fg=SUCCESS)
        else:
            self._clock.config(text=time.strftime("%H:%M:%S"), fg=MUTED)
        self._refresh_log()
        self.after(1000, self._tick)

    # ── Exit ──────────────────────────────────────────────────────────────────
    def _quit(self):
        engine.stop_flag = True
        self.destroy()


if __name__ == "__main__":
    GestureWaveApp().mainloop()
