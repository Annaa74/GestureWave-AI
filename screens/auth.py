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
                self.server.auth_result = {
                    "type": "token",
                    "access_token": params["access_token"][0],
                    "refresh_token": params.get("refresh_token", [""])[0],
                }
                self._respond(200, "Authenticated! You can close this tab.")
            else:
                self._respond(200, "No token received. Try again.")
        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        color = "#10b981" if "Authenticated" in msg else "#ef4444"
        html = (
            f'<html><body style="font-family:system-ui;background:#050510;color:#f1f5f9;'
            f'display:flex;justify-content:center;align-items:center;height:100vh;margin:0">'
            f'<div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:16px;'
            f'padding:48px;text-align:center">'
            f'<h2 style="color:{color};margin:0 0 8px">{msg}</h2>'
            f'<p style="color:#64748b;margin:0">Return to GestureWave AI.</p>'
            f'</div></body></html>'
        )
        self.wfile.write(html.encode("utf-8"))

    def _serve_hash_extractor(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8"><title>GestureWave AI</title></head>'
            '<body style="font-family:system-ui;background:#050510;color:#f1f5f9;'
            'display:flex;justify-content:center;align-items:center;height:100vh;margin:0">'
            '<div style="background:#0d0d1a;border:1px solid #1e1e3a;border-radius:16px;'
            'padding:48px;text-align:center;min-width:300px">'
            '<p id="s" style="color:#64748b;margin:0">Completing authentication...</p></div>'
            '<script>(function(){var a=0;function t(){a++;var h=window.location.hash;'
            'if(h&&h.length>1){window.location.replace("/auth/tokens?"+h.substring(1));return;}'
            'if(a<30){setTimeout(t,100);}else{'
            'document.getElementById("s").innerHTML="<span style=color:#ef4444>No token received.</span>";'
            '}}setTimeout(t,50);})();</script></body></html>'
        )
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, *a):
        pass


# ── Auth Screen ──────────────────────────────────────────────────────────────

class AuthScreen(tk.Frame):
    """Full-width auth screen — no scrolling, fills entire window."""

    def __init__(self, parent, supabase_client, supabase_error, on_success, on_back):
        super().__init__(parent, bg=BG0)
        self.sb = supabase_client
        self.sb_err = supabase_error
        self.on_success = on_success
        self.on_back = on_back
        self._mode = "signin"
        self._build()

    # ── Layout ───────────────────────────────────────────────────────────────
    def _build(self):
        # Back button
        top = tk.Frame(self, bg=BG0)
        top.pack(fill="x", padx=28, pady=(8, 0))
        clr_btn(top, "<-  Back", BG0, MUTED, self.on_back,
                font=("Segoe UI", 10), padx=6, pady=2).pack(anchor="w")

        # Center area
        center = tk.Frame(self, bg=BG0)
        center.pack(fill="both", expand=True, padx=40, pady=(0, 10))

        # Logo
        tk.Label(center, text="~", font=("Segoe UI", 24, "bold"),
