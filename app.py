"""
GestureWave AI — Desktop Launcher v2.0
Wraps the enhanced gesture engine with a GUI control panel.
"""
import sys
import os
import threading
import tkinter as tk
from tkinter import messagebox

# ── Dependency check ─────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg, imp in [
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("pyautogui", "pyautogui"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    return missing

missing = check_deps()
if missing:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror(
        "Missing Dependencies",
        "The following packages are required:\n\n  " +
        "\n  ".join(missing) +
        "\n\nRun:  pip install " + " ".join(missing)
    )
    sys.exit(1)

# Import the gesture engine after dep check
import main as engine


# ── GUI App ──────────────────────────────────────────────────────────
class GestureWaveApp(tk.Tk):
    BG      = "#0d0d0d"
    BG2     = "#161616"
    ACCENT  = "#2563eb"
    TEXT    = "#f0f0f0"
    MUTED   = "#666"
    FONT    = ("Courier New", 10)
    FONT_B  = ("Courier New", 10, "bold")

    def __init__(self):
        super().__init__()
        self.title("GestureWave AI")
        self.geometry("460x380")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._running    = False
        self._eng_thread = None

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=24)

        # Logo row
        hdr = tk.Frame(self, bg=self.BG)
        hdr.pack(fill="x", pady=(22, 0), **pad)
        tk.Label(hdr, text="≋  GestureWave AI", font=("Courier New", 17, "bold"),
                 fg=self.ACCENT, bg=self.BG).pack(side="left")
        tk.Label(hdr, text="v2.0", font=self.FONT, fg=self.MUTED, bg=self.BG).pack(side="right", pady=4)

        self._sep()

        # Status row
        sf = tk.Frame(self, bg=self.BG2, relief="flat")
        sf.pack(fill="x", padx=20, pady=(0, 14))
        self._dot = tk.Label(sf, text="●", font=("Arial", 13), fg=self.MUTED, bg=self.BG2)
        self._dot.pack(side="left", padx=(12, 6), pady=10)
        self._status = tk.Label(sf, text="Ready — press Start", font=self.FONT,
                                fg="#aaa", bg=self.BG2, anchor="w")
        self._status.pack(side="left", pady=10)

        # Buttons
        bf = tk.Frame(self, bg=self.BG)
        bf.pack(padx=20, fill="x", pady=(0, 14))
        self._btn_start = self._btn(bf, "▶  Start Tracking", self.ACCENT, self._start)
        self._btn_start.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._btn_stop = self._btn(bf, "■  Stop", "#374151", self._stop, state="disabled")
        self._btn_stop.pack(side="right")

        self._sep()

        # Gesture guide
        gf = tk.Frame(self, bg=self.BG)
        gf.pack(fill="x", padx=24, pady=(8, 0))
        tk.Label(gf, text="GESTURE REFERENCE", font=("Courier New", 8, "bold"),
                 fg=self.MUTED, bg=self.BG).pack(anchor="w", pady=(0, 6))

        gestures = [
            ("☝  Index finger",          "Move cursor"),
            ("🤏  Index + Thumb close",   "Left click"),
            ("✌  Two fingers [hold]",    "Drag & drop"),
            ("🤌  Middle + Thumb close",  "Right click"),
            ("✌  Peace sign [up/down]",  "Scroll"),
            ("🔍  Two fingers spread",    "Zoom in / out"),
            ("⚡  Quick double pinch",    "Double click"),
            ("✋  Open palm",             "Pause / Resume"),
        ]
        for gesture, action in gestures:
            row = tk.Frame(gf, bg=self.BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=gesture, font=self.FONT, fg="#888", bg=self.BG,
                     width=28, anchor="w").pack(side="left")
            tk.Label(row, text=action, font=self.FONT, fg="#4f9eff", bg=self.BG,
                     anchor="w").pack(side="left")

        self._sep()

        # Footer
        tk.Label(self, text="ESC in camera window to exit  •  Ctrl+corner = emergency stop",
                 font=("Courier New", 8), fg=self.MUTED, bg=self.BG).pack(pady=(4, 10))

    def _sep(self):
        tk.Frame(self, bg="#1f1f1f", height=1).pack(fill="x", padx=20, pady=8)

    def _btn(self, parent, text, bg, cmd, state="normal"):
        return tk.Button(parent, text=text, font=self.FONT_B, fg="white", bg=bg,
                         activebackground=bg, activeforeground="white",
                         relief="flat", bd=0, padx=18, pady=9, cursor="hand2",
                         state=state, command=cmd)

    # ── Actions ──────────────────────────────────────────────────────
    def _start(self):
        if self._running:
            return
        self._running = True
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._set_status("🔵 Starting camera…", "#3b82f6")
        self._eng_thread = threading.Thread(target=self._run_engine, daemon=True)
        self._eng_thread.start()

    def _run_engine(self):
        try:
            self._set_status("🟢 Tracking active", "#22c55e")
            engine.run()
        except Exception as e:
            self._set_status(f"❌ Error: {e}", "#ef4444")
        finally:
            self._running = False
            self.after(0, lambda: self._btn_start.config(state="normal"))
            self.after(0, lambda: self._btn_stop.config(state="disabled"))
            self._set_status("⚫ Stopped", self.MUTED)

    def _stop(self):
        # Signal engine to stop by setting the flag (engine checks .running if wrapped)
        # For now: destroy the OpenCV window which causes cap.read() to fail → clean exit
        import cv2
        cv2.destroyAllWindows()

    def _set_status(self, msg, dot_color=None):
        def _do():
            self._status.config(text=msg)
            if dot_color:
                self._dot.config(fg=dot_color)
        self.after(0, _do)

    def _on_close(self):
        import cv2
        cv2.destroyAllWindows()
        self.destroy()


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    GestureWaveApp().mainloop()
