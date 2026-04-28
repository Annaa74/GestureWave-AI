<div align="center">
  <img src="assets/banner.png" alt="GestureWave AI Banner" width="100%" />

  # GestureWave AI

  ### Real-Time, Touch-Free Gesture Control System

  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
  [![MediaPipe](https://img.shields.io/badge/MediaPipe-ML_Pipeline-FF6F00.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
  [![Supabase](https://img.shields.io/badge/Supabase-Auth_%26_DB-3FCF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
  [![License](https://img.shields.io/badge/License-MIT-A855F7.svg)](LICENSE)

  *Control your computer with hand gestures — no hardware, no gloves, just your webcam.*

  **Developed by [Aditya Yadav],[Ananya Baghel](https://github.com/Annaa74)**

</div>

---

## 🌊 Overview

**GestureWave AI** is a production-grade, touch-free gesture control engine that translates physical hand movements into precise desktop actions in real-time. Built on top of Google's MediaPipe ML pipeline and OpenCV, it transforms any standard webcam into a high-precision input device — enabling cursor control, clicking, scrolling, and zooming through natural hand gestures.

The system features a complete authentication layer (Supabase), a premium dark-mode desktop UI, real-time gesture analytics, and a safety-first architecture that prevents accidental system modifications.

### Why This Exists

Traditional input devices require physical contact. GestureWave AI eliminates that barrier entirely — enabling:
- **Accessible computing** for users with motor disabilities
- **Touchless presentations** where you control slides from across the room
- **Hygienic interfaces** in medical/industrial environments
- **Spatial computing research** as a foundation for gesture-based UIs

---

## ✨ Key Features

### 🎯 Gesture Engine
- **Click-on-Release State Machine** — Clicks fire on pinch *release*, not detection, eliminating accidental spam-clicks
- **Pinch Stability Requirement** — Requires 3 consecutive stable frames before registering a pinch
- **EMA Cursor Smoothing** — Exponential Moving Average with dynamic velocity boosting for precise, jitter-free cursor movement
- **Dead Zone Filtering** — Ignores micro-movements below a configurable pixel threshold

### 🛡️ Safety Architecture
- **Banned Gesture Detection** — Offensive gestures (middle finger) are detected with highest priority and blocked with a 1-second freeze penalty
- **System Safety Boundaries** — The `ActionExecutor` is restricted to mouse actions, scroll, and zoom only. No file operations, no arbitrary keyboard input, no system settings modification
- **Blocked Hotkeys Registry** — Dangerous key combinations (`Alt+F4`, `Ctrl+W`, `Ctrl+Delete`, etc.) are explicitly listed and can never be triggered
- **Safe Top Bar** — Clicks are blocked in the top 90px of the screen to prevent accidental taskbar/title bar interactions

### 🔐 Authentication & Persistence
- **Supabase Integration** — Full Email/Password and Google OAuth (PKCE flow) authentication
- **Session Persistence** — Login tokens saved locally for auto-login on subsequent launches
- **Error Visibility** — All database errors are logged with actionable instructions (no silent failures)
- **Auto-Table Detection** — If the database table is missing, the system prints the exact SQL to create it

### 📊 Real-Time Analytics
- **Live Gesture Log** — Every gesture event is logged with timestamp, action type, and duration in milliseconds
- **Frequency Analytics** — Dashboard shows color-coded gesture frequency counts (Click: 12, Scroll: 8, Zoom: 3)
- **Thread-Safe Event Bus** — Engine and dashboard share events via a lock-protected in-memory deque

### 🎨 Premium Desktop UI
- **720×820 Fixed-Resolution Window** — iOS-inspired dark theme with Segoe UI typography
- **Tab-Based Dashboard** — Gestures reference, live settings tuning, and real-time log tabs
- **Live Settings** — Smoothing, dead zone, click sensitivity, and scroll speed adjustable during tracking without restart
- **DirectShow Camera Backend** — Reliable camera open/release cycling on Windows

---

## 🎮 Gesture Reference

| Gesture | Hand Shape | Action | Status |
|:--------|:-----------|:-------|:------:|
| ☝️ **Move Cursor** | Index finger only | Cursor follows fingertip position | ✅ Stable |
| 🤏 **Left Click** | Thumb + Index pinch → release | Fires click on pinch release | ✅ Stable |
| ⚡ **Double Click** | Two quick pinch-releases | Double click within 0.40s window | ✅ Stable |
| 🤟 **Right Click** | Index + Middle + Ring fingers up | Right-click at cursor position | ✅ Stable |
| ✌️ **Scroll Up** | Peace sign + hand in top half | Scroll up (configurable speed) | ✅ Stable |
| ✌️ **Scroll Down** | Peace sign + hand in bottom half | Scroll down | ✅ Stable |
| 👍 **Zoom In** | Closed fist + thumb extended up | Ctrl+Plus (browser/app zoom) | ✅ Stable |
| 👎 **Zoom Out** | Closed fist + thumb extended down | Ctrl+Minus | ⚠️ Experimental |
| ✋ **Pause/Resume** | Open palm (4+ fingers) | Freeze/unfreeze all tracking | ✅ Stable |
| 🖕 **Middle Finger** | Only middle finger extended | **BLOCKED** — 1s freeze penalty | 🚫 Banned |

---

## 🏗️ Architecture

### System Flow

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Webcam    │────▶│  MediaPipe  │────▶│ Gesture Classifier│
│ (DirectShow)│     │  Hands ML  │     │  (gestures.py)    │
└─────────────┘     └─────────────┘     └────────┬─────────┘
                                                  │
                                    ┌─────────────▼──────────────┐
                                    │   Is it BANNED?            │
                                    │   YES → Freeze 1s + Block  │
                                    │   NO  → Continue           │
                                    └─────────────┬──────────────┘
                                                  │
                         ┌────────────────────────▼────────────────────────┐
                         │              State Machine (main.py)            │
                         │  IDLE → MOVING → PINCHING → click-on-RELEASE   │
                         │         SCROLLING / ZOOMING / PAUSED            │
                         └────────────────────────┬────────────────────────┘
                                                  │
              ┌───────────────────┬───────────────▼────────────────┐
              │                   │                                │
     ┌────────▼────────┐  ┌──────▼──────┐              ┌──────────▼──────────┐
     │ ActionExecutor  │  │ Gesture Log │              │    Dashboard UI     │
     │  (actions.py)   │  │ (in-memory) │              │   (dashboard.py)    │
     │                 │  │             │              │                     │
     │ ✅ Mouse move   │  │ Events +    │──────────────▶│ Live Log tab       │
     │ ✅ Click/scroll │  │ Frequencies │              │ Frequency bar      │
     │ ✅ Zoom only    │  └─────────────┘              │ Settings (live)    │
     │ ❌ No file ops  │                               └─────────────────────┘
     │ ❌ No sys mods  │
     └─────────────────┘
```

### Project Structure

```text
GestureWave-AI/
├── app.py                    # Application launcher with session persistence
├── main.py                   # Core gesture engine (click-on-release state machine)
├── ui_theme.py               # Design tokens, palette, fonts, and UI helpers
│
├── core/                     # Engine internals
│   ├── config.py             # Runtime configuration (thresholds, timing, safety)
│   ├── gestures.py           # Hand landmark classification with banned detection
│   ├── actions.py            # Safe system action executor (mouse/scroll/zoom only)
│   ├── gesture_log.py        # Thread-safe real-time gesture event logging
│   └── __init__.py
│
├── screens/                  # Modular UI components (Tkinter)
│   ├── welcome.py            # Landing screen with Demo / Sign In options
│   ├── auth.py               # Email/Password + Google OAuth authentication
│   ├── dashboard.py          # Tracking dashboard with live log + settings
│   └── __init__.py
│
├── frontend-app/             # Next.js web showcase (separate deployment)
├── assets/                   # Banner image and visual assets
│
├── requirements.txt          # Python dependencies
├── installer.iss             # Windows installer config (Inno Setup)
├── .gitignore                # Python, Node, IDE, secrets, session exclusions
├── LICENSE                   # MIT License
└── README.md                 # This file
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Click-on-release** instead of click-on-detect | Eliminates continuous click spam from jittery detection |
| **DirectShow** (`cv2.CAP_DSHOW`) instead of default MSMF | Reliable camera release on Windows — can stop/restart tracking |
| **try/finally** camera wrapper | Camera is *always* released, even if engine crashes |
| **BANNED checked first** in classifier | Offensive gestures can't accidentally trigger scroll/zoom |
| **Session file** (`~/.gesturewave_session.json`) | Users skip login on subsequent launches |
| **Supabase** instead of MongoDB | Built-in Auth + Realtime + PostgREST — no extra infra needed |
| **In-memory deque** instead of DB for gesture events | Zero latency, thread-safe, no persistence overhead for ephemeral data |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** with pip
- A working **webcam**
- **Windows 10/11** (DirectShow camera backend)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Annaa74/GestureWave-AI.git
cd GestureWave-AI
git checkout upgrade
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment**

Create a `.env` file in the project root:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
```

> [!TIP]
> Don't have a Supabase project? The app works fine without one — you can use **Demo Mode** from the welcome screen (5-minute trial with Move + Click gestures).

**5. Launch the application**
```bash
python app.py
```

---

## 💻 Usage

### Normal Mode (Recommended)
```bash
python app.py
```
1. Sign in with Google OAuth or Email/Password (or try Demo mode)
2. Click **▶ Start Tracking** to begin
3. Use hand gestures to control your desktop
4. Press **ESC** in the camera window or click **■ Stop** to end
5. Adjust settings in the **Settings** tab — changes apply live

### Development Mode
```bash
python main.py
```
Bypasses authentication and launches the tracking engine directly.

> [!WARNING]
> Do not run `app.py` and `main.py` simultaneously — both will try to lock the camera.

---

## ⚙️ Configuration

All parameters are tunable in `core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SMOOTH_ALPHA` | 0.30 | Cursor smoothing factor (lower = smoother, higher = faster) |
| `DEAD_ZONE` | 2px | Minimum pixel movement to register |
| `CLICK_THRESH` | 30px | Distance threshold for pinch detection |
| `CLICK_COOLDOWN` | 0.30s | Minimum time between consecutive clicks |
| `PINCH_STABLE_FRAMES` | 3 | Frames the pinch must be held before it counts |
| `LEFT_CLICK_FREEZE` | 0.05s | Brief cursor lock after click fires |
| `SCROLL_AMOUNT` | 100 | Scroll wheel amount per gesture |
| `SAFE_TOP_BAR` | 90px | Clicks blocked in this top region |

---

## 🛡️ Safety & Security

### What GestureWave AI CAN Do
- ✅ Move the mouse cursor
- ✅ Left click, right click, double click
- ✅ Scroll up/down
- ✅ Zoom in/out (Ctrl+Plus / Ctrl+Minus)

### What GestureWave AI CANNOT Do
- ❌ Create, delete, or modify files
- ❌ Type text or press arbitrary keys
- ❌ Access system settings
- ❌ Trigger dangerous hotkeys (Alt+F4, Ctrl+W, Ctrl+Delete, Win key, etc.)
- ❌ Execute shell commands
- ❌ Modify the registry or environment variables

The `BLOCKED_HOTKEYS` constant in `core/actions.py` explicitly lists every dangerous key combination that is permanently prohibited.

---

## 🔮 Roadmap

- [ ] **Kalman Filter** — Replace EMA smoothing with Kalman filter for cursor stabilization
- [ ] **Multi-Hand Support** — Two-handed chord gestures for complex shortcuts
- [ ] **Gesture Telemetry** — Log gesture sessions to Supabase for usage analytics
- [ ] **Custom Gesture Registry** — User-defined gestures mapped to custom actions
- [ ] **System Tray Mode** — Run as a background process with tray icon
- [ ] **Cross-Platform** — macOS and Linux camera backend support

---

## 🛑 Known Limitations

- **Lighting Dependency** — Strong backlighting or very low light reduces MediaPipe's detection confidence
- **Occlusion Errors** — If the thumb is hidden behind the palm (from the camera's perspective), pinch detection may fail
- **Single Hand** — Currently tracks only one hand at a time
- **Windows Only** — DirectShow camera backend is Windows-specific (macOS/Linux would need a different backend)

---

## 🏢 Use Cases

| Scenario | How GestureWave AI Helps |
|----------|--------------------------|
| **Presentations** | Control slides naturally from across the room |
| **Accessibility** | Interact with desktop UIs without physical peripherals |
| **Medical/Industrial** | Touchless interfaces in sterile or hazardous environments |
| **Interactive Art** | Foundation for smart mirrors and spatial installations |
| **Research** | Baseline for HCI and spatial computing experiments |

---

## 📦 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Vision** | MediaPipe 0.10.5 | Hand landmark detection (21-point model) |
| **Computer Vision** | OpenCV 4.8.1 | Camera capture, frame processing, HUD rendering |
| **System Control** | PyAutoGUI | Mouse/keyboard action execution |
| **Backend** | Supabase (PostgreSQL) | Authentication, user records, session management |
| **UI Framework** | Tkinter | Desktop GUI with dark theme |
| **Math** | NumPy | Vector calculations, distance computation |

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

  **GestureWave AI** — Control space, not screens.

  Developed with 💙 by **[Aditya Yadav](https://github.com/Annaa74)**

  [![GitHub](https://img.shields.io/badge/GitHub-Annaa74-181717.svg?logo=github&logoColor=white)](https://github.com/Annaa74)

</div>
