"""Sign-In / Sign-Up Screen — Supabase Auth + gesturewave_users table"""
import tkinter as tk
import threading
import webbrowser
import http.server
import urllib.parse
import time
from ui_theme import *

OAUTH_PORT = 5789
OAUTH_REDIRECT = f"http://localhost:{OAUTH_PORT}/auth/callback"


# ── Local OAuth Server ───────────────────────────────────────────────────────

class _OAuthHandler(http.server.BaseHTTPRequestHandler):
    """Handles Supabase OAuth callback (PKCE + implicit)."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
