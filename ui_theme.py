"""GestureWave AI — UI Theme & Helpers (iOS-inspired)"""
import tkinter as tk, os, json, time

# ── Palette ──
BG0 = "#050510"; BG1 = "#0d0d1a"; BG2 = "#12122a"; BG3 = "#1a1a2e"
BORDER = "#1e1e3a"; ACCENT = "#3b82f6"; ACCENT2 = "#8b5cf6"; CYAN = "#06b6d4"
SUCCESS = "#10b981"; DANGER = "#ef4444"; WARNING = "#f59e0b"
TEXT = "#f1f5f9"; MUTED = "#64748b"

# ── iOS-inspired Fonts (Segoe UI is the closest to SF Pro on Windows) ──
# Bold variants for headings/labels
SANS = ("Segoe UI", 11)
SANS_B = ("Segoe UI Semibold", 11)
SANS_SM = ("Segoe UI", 9)
SANS_SM_B = ("Segoe UI Semibold", 9)
MONO = ("Consolas", 10)
TITLE = ("Segoe UI Semibold", 16)
HERO = ("Segoe UI", 32, "bold")
HEADING = ("Segoe UI Semibold", 13)

DEMO_LOCK = os.path.join(os.path.expanduser("~"), ".gesturewave_demo_used")

def is_demo_locked():
    return os.path.exists(DEMO_LOCK)

def lock_demo():
    try:
        with open(DEMO_LOCK, "w") as f: json.dump({"ts": time.time()}, f)
    except: pass

def clr_btn(parent, text, bg, fg, cmd, font=SANS_B, padx=20, pady=8, **kw):
    return tk.Button(parent, text=text, bg=bg, fg=fg, activebackground=bg,
                     activeforeground=fg, font=font, relief="flat", bd=0,
                     cursor="hand2", command=cmd, padx=padx, pady=pady, **kw)

def lbl(parent, text, font=SANS, fg=TEXT, bg=BG1, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)

class Tooltip:
    def __init__(self, w, text):
