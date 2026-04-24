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
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/auth/callback":
            if "code" in params:
                self.server.auth_result = {"type": "pkce", "code": params["code"][0]}
                self._respond(200, "Authenticated! You can close this tab.")
            elif "access_token" in params:
                self.server.auth_result = {
                    "type": "token",
                    "access_token": params["access_token"][0],
                    "refresh_token": params.get("refresh_token", [""])[0],
                }
                self._respond(200, "Authenticated! You can close this tab.")
            elif "error" in params:
                desc = params.get("error_description", ["Unknown error"])[0]
                self._respond(200, f"Error: {desc}")
            else:
                self._serve_hash_extractor()
        elif parsed.path == "/auth/tokens":
            if "access_token" in params:
