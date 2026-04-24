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
