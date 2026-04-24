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
                 fg=ACCENT, bg=BG0).pack(pady=(2, 0))
        tk.Label(center, text="GestureWave AI", font=("Segoe UI", 16, "bold"),
                 fg=TEXT, bg=BG0).pack(pady=(0, 6))

        # Tab row
        tf = tk.Frame(center, bg=BG0)
        tf.pack(pady=(0, 6))
        self._tab_si = clr_btn(tf, "Sign In", ACCENT, "white",
                               lambda: self._switch("signin"),
                               font=("Segoe UI", 10, "bold"), padx=24, pady=5)
        self._tab_si.pack(side="left", padx=4)
        self._tab_su = clr_btn(tf, "Sign Up", BG3, MUTED,
                               lambda: self._switch("signup"),
                               font=("Segoe UI", 10, "bold"), padx=24, pady=5)
        self._tab_su.pack(side="left", padx=4)

        # Card
        card = tk.Frame(center, bg=BG1, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="both", expand=True)

        # Form inside card
        self._form = tk.Frame(card, bg=BG1)
        self._form.pack(fill="both", expand=True, padx=50, pady=12)

        self._build_form()

    def _switch(self, mode):
        self._mode = mode
        if mode == "signin":
            self._tab_si.config(bg=ACCENT, fg="white")
            self._tab_su.config(bg=BG3, fg=MUTED)
        else:
            self._tab_si.config(bg=BG3, fg=MUTED)
            self._tab_su.config(bg=ACCENT, fg="white")
        self._build_form()

    # ── Build Form ───────────────────────────────────────────────────────────
    def _build_form(self):
        for w in self._form.winfo_children():
            w.destroy()
        is_signup = self._mode == "signup"

        FNT_LBL = ("Segoe UI Semibold", 10)
        FNT_IN = ("Segoe UI", 11)
        FNT_BTN = ("Segoe UI", 11, "bold")
        IPAD = 5

        # Google button
        g = clr_btn(self._form, "  Continue with Google", "#ffffff", "#1a1a1a",
                    self._do_google, font=("Segoe UI", 10, "bold"), padx=12, pady=7)
        g.pack(fill="x", pady=(2, 4))

        # Divider
        df = tk.Frame(self._form, bg=BG1)
        df.pack(fill="x", pady=(1, 4))
        tk.Frame(df, bg=BORDER, height=1).pack(side="left", fill="x", expand=True)
        tk.Label(df, text="  or  ", font=("Segoe UI", 10), fg=MUTED, bg=BG1).pack(side="left")
        tk.Frame(df, bg=BORDER, height=1).pack(side="left", fill="x", expand=True)

        # Display Name (signup only)
        if is_signup:
            tk.Label(self._form, text="Display Name", font=FNT_LBL,
                     fg=TEXT, bg=BG1, anchor="w").pack(fill="x", pady=(2, 1))
            self._name_e = tk.Entry(self._form, bg=BG2, fg=TEXT, insertbackground=TEXT,
                                    font=FNT_IN, relief="flat",
                                    highlightbackground=BORDER, highlightthickness=1)
            self._name_e.pack(fill="x", ipady=IPAD, pady=(0, 2))

        # Email
        tk.Label(self._form, text="Email", font=FNT_LBL,
                 fg=TEXT, bg=BG1, anchor="w").pack(fill="x", pady=(2, 1))
        self._email_e = tk.Entry(self._form, bg=BG2, fg=TEXT, insertbackground=TEXT,
                                 font=FNT_IN, relief="flat",
                                 highlightbackground=BORDER, highlightthickness=1)
        self._email_e.pack(fill="x", ipady=IPAD, pady=(0, 2))

        # Password
        tk.Label(self._form, text="Password", font=FNT_LBL,
                 fg=TEXT, bg=BG1, anchor="w").pack(fill="x", pady=(2, 1))
        self._pw_e = tk.Entry(self._form, bg=BG2, fg=TEXT, insertbackground=TEXT,
                              font=FNT_IN, show="*", relief="flat",
                              highlightbackground=BORDER, highlightthickness=1)
        self._pw_e.pack(fill="x", ipady=IPAD, pady=(0, 2))

        # Confirm Password (signup only)
        if is_signup:
            tk.Label(self._form, text="Confirm Password", font=FNT_LBL,
                     fg=TEXT, bg=BG1, anchor="w").pack(fill="x", pady=(2, 1))
            self._pw2_e = tk.Entry(self._form, bg=BG2, fg=TEXT, insertbackground=TEXT,
                                   font=FNT_IN, show="*", relief="flat",
                                   highlightbackground=BORDER, highlightthickness=1)
            self._pw2_e.pack(fill="x", ipady=IPAD, pady=(0, 2))

        # Message
        self._msg = tk.Label(self._form, text="", font=("Segoe UI", 9),
                             fg=DANGER, bg=BG1, wraplength=500, justify="center")
        self._msg.pack(fill="x", pady=(4, 1))
        if not self.sb:
            self._msg.config(text="Supabase not configured. Check .env file.", fg=WARNING)

        # Submit button
        btn_text = "Create Account" if is_signup else "Sign In"
        cmd = self._do_signup if is_signup else self._do_login
        self._btn = clr_btn(self._form, btn_text, ACCENT, "white",
                            cmd, font=FNT_BTN, padx=24, pady=8)
        self._btn.pack(fill="x", pady=(2, 4))

        # Enter key
        entries = [self._email_e, self._pw_e]
        if is_signup:
            entries += [self._name_e, self._pw2_e]
        for entry in entries:
            entry.bind("<Return>", lambda e, c=cmd: c())

        # Focus first field
        first = self._name_e if is_signup else self._email_e
        self.after(100, lambda: first.focus_set())

    # ── Sign In ──────────────────────────────────────────────────────────────
    def _do_login(self):
        if not self.sb:
            self._msg.config(text="Supabase not available.", fg=DANGER)
            return
        email = self._email_e.get().strip()
        pw = self._pw_e.get().strip()
        if not email or not pw:
            self._msg.config(text="Enter email and password.", fg=DANGER)
            return
        self._btn.config(text="Signing in...", state="disabled")
        self._msg.config(text="")
        threading.Thread(target=self._auth_login, args=(email, pw), daemon=True).start()

    def _auth_login(self, email, pw):
        try:
            res = self.sb.auth.sign_in_with_password({"email": email, "password": pw})
            if res.user and res.session:
                # Update sign-in count in gesturewave_users
                try:
                    self.sb.table("gesturewave_users").update({
                        "sign_in_count": res.user.user_metadata.get("sign_in_count", 1),
                        "last_sign_in": "now()"
                    }).eq("id", res.user.id).execute()
                except Exception:
                    pass
                self._load_user(res.user.id)
            else:
                self.after(0, lambda: self._fail("Sign-in failed. Check credentials."))
        except Exception as e:
            err = str(e)
            # Parse common errors into friendly messages
            if "Invalid login" in err:
                msg = "Invalid email or password."
            elif "Email not confirmed" in err:
                msg = "Email not confirmed. Check your inbox."
            elif "rate limit" in err.lower():
                msg = "Too many attempts. Wait a minute."
            elif "fetch" in err.lower() or "connection" in err.lower():
                msg = "Connection error. Check your internet."
            else:
                msg = f"Error: {err[:150]}"
            self.after(0, lambda m=msg: self._fail(m))

    # ── Sign Up ──────────────────────────────────────────────────────────────
    def _do_signup(self):
        if not self.sb:
            self._msg.config(text="Supabase not available.", fg=DANGER)
            return
        name = self._name_e.get().strip()
        email = self._email_e.get().strip()
        pw = self._pw_e.get().strip()
        pw2 = self._pw2_e.get().strip()
        if not name or not email or not pw:
            self._msg.config(text="Please fill all fields.", fg=DANGER)
            return
        if len(pw) < 6:
            self._msg.config(text="Password must be at least 6 characters.", fg=DANGER)
            return
        if pw != pw2:
            self._msg.config(text="Passwords do not match.", fg=DANGER)
            return
        self._btn.config(text="Creating account...", state="disabled")
        self._msg.config(text="")
        threading.Thread(target=self._auth_signup, args=(email, pw, name), daemon=True).start()

    def _auth_signup(self, email, pw, name):
        try:
            res = self.sb.auth.sign_up({
                "email": email, "password": pw,
                "options": {"data": {"display_name": name}}
            })
            if res.user and res.session:
                # Signup with auto-confirm -> go straight to dashboard
                self._load_user(res.user.id)
            elif res.user:
                # User created but no session -> try sign-in (auto-confirm trigger)
                try:
                    login = self.sb.auth.sign_in_with_password({
                        "email": email, "password": pw
                    })
                    if login.user and login.session:
                        self._load_user(login.user.id)
                        return
                except Exception:
                    pass
                self.after(0, lambda: self._info("Account created! Sign in to continue."))
            else:
                self.after(0, lambda: self._fail("Signup failed. Try again."))
        except Exception as e:
            err = str(e)
            if "already" in err.lower():
                msg = "Email already registered. Use Sign In instead."
            elif "rate limit" in err.lower():
                msg = "Too many attempts. Wait a minute."
            elif "invalid" in err.lower() and "email" in err.lower():
                msg = "Please enter a valid email address."
            else:
                msg = f"Error: {err[:150]}"
            self.after(0, lambda m=msg: self._fail(m))

    # ── Google OAuth ─────────────────────────────────────────────────────────
    def _do_google(self):
        if not self.sb:
            self._msg.config(text="Supabase not available.", fg=DANGER)
            return
        self._msg.config(text="Opening browser...", fg=ACCENT)
        threading.Thread(target=self._google_flow, daemon=True).start()

    def _google_flow(self):
        server = None
        try:
            server = http.server.HTTPServer(("127.0.0.1", OAUTH_PORT), _OAuthHandler)
            server.auth_result = None
            server.timeout = 10

            try:
                resp = self.sb.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {"redirect_to": OAUTH_REDIRECT}
                })
            except Exception as e:
                err = str(e)
                if "not enabled" in err.lower() or "unsupported" in err.lower():
                    self.after(0, lambda: self._fail(
                        "Google sign-in not enabled yet.\n"
                        "Enable it in Supabase Dashboard > Auth > Providers > Google."))
                else:
                    self.after(0, lambda m=err: self._fail(f"Google OAuth: {m[:100]}"))
                return

            webbrowser.open(resp.url)

            deadline = time.time() + 120
            while server.auth_result is None and time.time() < deadline:
                server.handle_request()

            if server.auth_result:
                result = server.auth_result
                try:
                    if result.get("type") == "pkce":
                        self.sb.auth.exchange_code_for_session({"auth_code": result["code"]})
                    elif result.get("type") == "token":
                        self.sb.auth.set_session(result["access_token"], result["refresh_token"])

                    user = self.sb.auth.get_user()
                    if user and user.user:
                        self._load_user(user.user.id)
                    else:
                        self.after(0, lambda: self._fail("Could not get user after Google auth."))
                except Exception as e:
                    self.after(0, lambda m=str(e): self._fail(f"Google auth: {m[:100]}"))
            else:
                self.after(0, lambda: self._fail("Google sign-in timed out. Try again."))

        except OSError as e:
            if "10048" in str(e) or "Address already in use" in str(e):
                self.after(0, lambda: self._fail("Port busy. Close other instances."))
            else:
                self.after(0, lambda m=str(e): self._fail(m))
        except Exception as e:
            self.after(0, lambda m=str(e): self._fail(f"Google auth error: {m[:100]}"))
        finally:
            if server:
                try:
                    server.server_close()
                except Exception:
                    pass

    # ── Load user -> dashboard ───────────────────────────────────────────────
    def _load_user(self, uid):
        role = "standard"
        settings = []
        permissions = []

        try:
            p = self.sb.table("profiles").select("role").eq("id", uid).execute()
            role = p.data[0]["role"] if p.data else "standard"
        except Exception:
            pass

        try:
            s = self.sb.table("user_settings").select("*").eq("user_id", uid).execute()
            settings = s.data or []
        except Exception:
            pass

        try:
            g = self.sb.table("gesture_permissions").select(
                "gesture_name,is_allowed").eq("user_id", uid).execute()
            permissions = g.data or []
        except Exception:
            pass

        self.after(0, lambda: self.on_success(role, settings, permissions))

    # ── UI Helpers ───────────────────────────────────────────────────────────
    def _fail(self, msg):
        txt = "Create Account" if self._mode == "signup" else "Sign In"
        try:
            self._btn.config(text=txt, state="normal")
            self._msg.config(text=msg, fg=DANGER)
        except Exception:
            pass

    def _info(self, msg):
        try:
            btn_txt = "Create Account" if self._mode == "signup" else "Sign In"
            self._btn.config(text=btn_txt, state="normal")
            self._msg.config(text=msg, fg=SUCCESS)
            self.after(2000, lambda: self._switch("signin"))
        except Exception:
            pass
