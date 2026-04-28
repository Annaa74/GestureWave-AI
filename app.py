"""
GestureWave AI — Desktop Launcher v3.0
Premium dark-mode GUI with Auth, Demo mode & full dashboard.
"""
import sys, os, tkinter as tk, json
import importlib.util
from tkinter import messagebox
from dotenv import load_dotenv

load_dotenv()

# ── Dependency check ──
REQUIRED = [("cv2","opencv-python"),("mediapipe","mediapipe"),
            ("pyautogui","pyautogui"),("numpy","numpy"),
            ("supabase","supabase"),("dotenv","python-dotenv")]
missing = [pkg for imp,pkg in REQUIRED if not importlib.util.find_spec(imp)]
if missing:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("GestureWave AI – Missing Packages",
        "Please install:\n\n  pip install " + " ".join(missing))
    sys.exit(1)

import main as engine
from ui_theme import *
from screens.welcome import WelcomeScreen
from screens.auth import AuthScreen
from screens.dashboard import Dashboard

# ── Supabase Init ──
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase_client = None
supabase_init_error = None

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    supabase_init_error = "missing_env_vars"
    print("[Supabase] SKIPPED (missing env vars)")
else:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("[Supabase] Client: OK")
    except Exception as e:
        supabase_init_error = str(e)
        print(f"[Supabase] Client: FAILED - {e}")

# ── Session Cache ──────────────────────────────────────────────────────────
SESSION_FILE = os.path.join(os.path.expanduser("~"), ".gesturewave_session.json")

def save_session(access_token, refresh_token, user_id):
    """Save login session to disk for auto-login on next launch."""
    try:
        data = {"access_token": access_token, "refresh_token": refresh_token, "user_id": user_id}
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f)
        print(f"[Session] Saved for user {user_id[:8]}...")
    except Exception as e:
        print(f"[Session] Save failed: {e}")

def load_session():
    """Load saved session from disk. Returns (access_token, refresh_token, user_id) or None."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            return data.get("access_token"), data.get("refresh_token"), data.get("user_id")
    except Exception:
        pass
    return None

def clear_session():
    """Remove saved session file on logout."""
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
            print("[Session] Cleared.")
    except Exception:
        pass


class GestureWaveApp(tk.Tk):
    W, H = 720, 820

    def __init__(self):
        super().__init__()
        self.title("GestureWave AI")
        # Center the window on screen
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - self.W) // 2
        y = (sh - self.H) // 2
        self.geometry(f"{self.W}x{self.H}+{x}+{y}")
        self.resizable(False, False)
        self.configure(bg=BG0)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._current = None

        # Try auto-login from saved session
        if supabase_client and self._try_auto_login():
            return  # Already navigated to dashboard
        self._show_welcome()

    def _try_auto_login(self):
        """Attempt to restore a previously saved session."""
        session_data = load_session()
        if not session_data:
            return False

        access_token, refresh_token, user_id = session_data
        if not access_token or not refresh_token:
            return False

        try:
            print("[Session] Restoring saved session...")
            supabase_client.auth.set_session(access_token, refresh_token)
            user = supabase_client.auth.get_user()
            if user and user.user:
                print(f"[Session] Auto-login successful: {user.user.id[:8]}...")
                self._on_auth_success("standard", [], [])
                return True
        except Exception as e:
            print(f"[Session] Auto-login failed (token expired): {e}")
            clear_session()
        return False

    def _clear(self):
        if self._current:
            if hasattr(self._current, 'cleanup'):
                self._current.cleanup()
            self._current.destroy()
            self._current = None

    def _show_welcome(self):
        self._clear()
        engine.Cfg.USER_ROLE = "guest"
        engine.stop_flag = True
        self._current = WelcomeScreen(self, on_demo=self._start_demo, on_signin=self._show_auth)
        self._current.pack(fill="both", expand=True)

    def _show_auth(self):
        self._clear()
        self._current = AuthScreen(self, supabase_client, supabase_init_error,
                                    on_success=self._on_auth_success, on_back=self._show_welcome)
        self._current.pack(fill="both", expand=True)

    def _start_demo(self):
        if is_demo_locked():
            messagebox.showinfo("Demo Expired", "Demo already used on this system.\nPlease sign in.")
            return
        self._clear()
        engine.Cfg.USER_ROLE = "guest"
        # Demo: only MOVE + LEFT_PINCH allowed
        engine.Cfg.ALLOWED_GESTURES = {"MOVE", "LEFT_PINCH"}
        self._current = Dashboard(self, engine, role="guest", is_demo=True,
                                   on_logout=self._show_welcome,
                                   demo_gestures={"MOVE", "LEFT_PINCH"})
        self._current.pack(fill="both", expand=True)

    def _on_auth_success(self, role, settings_data, perms_data):
        self._clear()
        engine.Cfg.USER_ROLE = role

        # Save session for auto-login next time
        if supabase_client:
            try:
                session = supabase_client.auth.get_session()
                if session:
                    save_session(session.access_token, session.refresh_token, session.user.id)
            except Exception as e:
                print(f"[Session] Could not save session: {e}")

        # Apply settings
        if settings_data:
            s = settings_data[0]
            if "smooth_alpha" in s: engine.Cfg.SMOOTH_ALPHA = float(s["smooth_alpha"])
            if "dead_zone" in s: engine.Cfg.DEAD_ZONE = int(s["dead_zone"])
            if "click_thresh" in s: engine.Cfg.CLICK_THRESH = int(s["click_thresh"])

        # Apply permissions
        if perms_data:
            allowed = {p["gesture_name"] for p in perms_data if p.get("is_allowed")}
            if allowed: engine.Cfg.ALLOWED_GESTURES = allowed
        else:
            # Full access
            engine.Cfg.ALLOWED_GESTURES = {
                "MOVE", "LEFT_PINCH", "RIGHT_CLICK", "DOUBLE_CLICK",
                "SCROLL_UP", "SCROLL_DOWN", "ZOOM_IN", "ZOOM_OUT", "PAUSE"
            }

        self._current = Dashboard(self, engine, role=role, is_demo=False,
                                   on_logout=self._do_logout)
        self._current.pack(fill="both", expand=True)

    def _do_logout(self):
        if supabase_client:
            try: supabase_client.auth.sign_out()
            except: pass
        clear_session()  # Remove saved login
        engine.stop_flag = True
        engine.Cfg.ALLOWED_GESTURES = {
            "MOVE", "LEFT_PINCH", "RIGHT_CLICK", "DOUBLE_CLICK",
            "SCROLL_UP", "SCROLL_DOWN", "ZOOM_IN", "PAUSE"
        }
        self._show_welcome()

    def _quit(self):
        engine.stop_flag = True
        self.destroy()


if __name__ == "__main__":
    GestureWaveApp().mainloop()
