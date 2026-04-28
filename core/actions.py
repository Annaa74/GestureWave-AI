"""
GestureWave AI — Safe Action Executor
Controls what gestures can actually DO on the system.
Safety: Only mouse movement, clicks, scroll, and zoom (Ctrl+/Ctrl-) are allowed.
No file operations, no system settings, no keyboard typing.
"""
import pyautogui
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# ── Blocked hotkeys that could modify system state ──────────────────────────
# These are hotkeys that should NEVER be triggered by gesture control
BLOCKED_HOTKEYS = {
    ("alt", "f4"),       # Close application
    ("ctrl", "w"),       # Close tab
    ("ctrl", "q"),       # Quit app
    ("ctrl", "delete"),  # Task manager / delete
    ("alt", "tab"),      # Switch window (could be confusing)
    ("win",),            # Start menu
    ("ctrl", "shift", "delete"),  # Clear browsing data
    ("ctrl", "s"),       # Save (unintended saves)
    ("ctrl", "z"),       # Undo
    ("ctrl", "shift", "esc"),    # Task manager
}


class ActionExecutor:
    """
    Safely executes mouse/scroll/zoom actions from gesture input.
    
    Safety guarantees:
    - Only mouse movement, left/right/double click, scroll, and zoom are allowed
    - Click actions are blocked in the top taskbar region (safe_top_bar)
    - No file system operations (create, delete, rename)
    - No system settings modification
    - No arbitrary keyboard input
    - All zoom is limited to Ctrl+Plus / Ctrl+Minus (browser/app zoom only)
    """

    def __init__(self, safe_top_bar=90, enable_click_actions=True):
        self.safe_top_bar = safe_top_bar
        self.enable_click_actions = enable_click_actions

    def safe_to_click(self, y):
        """Block clicks in the system taskbar/title bar region."""
        return self.enable_click_actions and y > self.safe_top_bar

    # ── Allowed Actions ─────────────────────────────────────────────────────

    def move_cursor(self, x, y):
        """Move the mouse cursor. Safe — no side effects."""
        pyautogui.moveTo(x, y, _pause=False)

    def left_click(self, x, y):
        """Left click at position. Blocked in taskbar region."""
        if self.safe_to_click(y):
            pyautogui.click(x, y)

    def right_click(self, x, y):
        """Right click at position. Blocked in taskbar region."""
        if self.safe_to_click(y):
            pyautogui.rightClick(x, y)

    def double_click(self, x, y):
        """Double click at position. Blocked in taskbar region."""
        if self.safe_to_click(y):
            pyautogui.doubleClick(x, y)

    def scroll_up(self, amount):
        """Scroll up. Safe — only moves the scroll wheel."""
        pyautogui.scroll(amount)

    def scroll_down(self, amount):
        """Scroll down. Safe — only moves the scroll wheel."""
        pyautogui.scroll(-amount)

    def zoom_in(self):
        """Zoom in using Ctrl+Plus. Safe — only changes view zoom level."""
        pyautogui.hotkey("ctrl", "+")

    def zoom_out(self):
        """Zoom out using Ctrl+Minus. Safe — only changes view zoom level."""
        pyautogui.hotkey("ctrl", "-")