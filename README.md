<div align="center">
  <img src="assets/banner.png" alt="GestureWave AI Banner" width="100%" />

  # GestureWave AI
  **Real-Time, Touch-Free Gesture Control System**

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![MediaPipe](https://img.shields.io/badge/MediaPipe-Enabled-orange.svg)](https://developers.google.com/mediapipe)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)](https://opencv.org/)
  [![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
</div>

---

## 🌊 Overview

**GestureWave AI** is an advanced, touch-free gesture control engine that maps physical hand movements to digital desktop actions in real-time. By leveraging computer vision and machine learning models, it establishes a seamless, invisible interface between you and your computer—turning your webcam into a high-precision peripheral.

Whether you are presenting to a large room, looking for accessible computing alternatives, or building spatial interfaces, GestureWave AI pushes the boundaries of human-computer interaction (HCI) using standard hardware.

---

## ✨ Features

- **Zero-Latency Tracking:** High-speed hand landmark detection optimized for standard CPU execution.
- **Dynamic Spatial Mapping:** Translates 3D coordinates into precise 2D desktop cursor operations with Exponential Moving Average (EMA) smoothing and dynamic velocity boosting.
- **Strict Heuristic Classification:** Highly reliable gesture detection using strict physical constraints (e.g., inverted knuckles, specific finger folding) to eliminate false positives.
- **Premium Desktop UI:** A highly refined, fixed-resolution (720x820) dark-mode Tkinter GUI (`app.py`) featuring an optimized non-scrolling authentication flow and interactive dashboard.
- **Robust Authentication:** Full integration with Supabase for Email/Password and Google OAuth authentication (with local PKCE redirect support).
- **Persistent Profiles:** Custom database triggers (`gesturewave_users`) automatically track user sessions, login counts, and metadata.
- **Failsafe Integrated:** Built-in OS-level fail-safes (PyAutoGUI) to instantly regain control if needed.

---

##  Final Gesture Mapping

The gesture engine has been refined to prioritize stability and reliability. All gestures are mapped to exact physical hand shapes.

| Action | Gesture | Description |
| :--- | :--- | :--- |
| **Move Cursor** | One finger (Pointer) | Point your index finger. The cursor mimics the absolute 2D position of your fingertip. |
| **Left Click** | Quick Pinch | Tap the tips of your thumb and index finger together quickly. |
| **Right Click** | Three Fingers Up | Extend your Index, Middle, and Ring fingers (Pinky folded down). |
| **Double Click** | Two Quick Pinches | Perform two standard left-click pinches rapidly within 0.36 seconds. |
| **Scroll Up/Down** | Peace Sign | Point index and middle fingers up. Move hand to the top half of the camera view to scroll up, or bottom half to scroll down. |
| **Zoom In** | Thumbs Up | Closed fist with the thumb extended upwards. |
| **Zoom Out** | Thumbs Down | Inverted closed fist with the thumb extended downwards. |
| **Pause / Resume** | Open Palm | Raise an open hand (4+ fingers up) to pause or unpause tracking. |

---

## 🏗️ Project Structure

The repository is highly modular, deeply separating the vision processing pipeline from operating system overrides.

```text
GestureWave-AI/
├── app.py                 # Desktop launcher UI & Route Manager
├── main.py                # Core gesture engine runtime
├── core/
│   ├── config.py          # Global runtime configurations
│   ├── gestures.py        # Landmark heuristics and classification logic
│   └── actions.py         # PyAutoGUI OS action execution
├── screens/               # Modular UI components
│   ├── welcome.py         # Landing and Demo initialization
│   ├── auth.py            # Optimized Sign-In / Sign-Up / Google OAuth UI
│   └── dashboard.py       # User control panel and live camera host
├── ui_theme.py            # Global styling, colors, and layout constants
├── frontend-app/          # Next.js frontend showcase & documentation
├── installer.iss          # Windows execution installer config (Inno Setup)
├── assets/                # Visual assets, banners, and interface icons
│
└── Standalone Utilities (Not actively used by main app loop):
├── gesture_registry.py    # Custom gesture storage and matching logic
├── gesture_utils.py       # Landmark normalization and spatial helpers
└── feature_extraction.py  # Advanced feature vector calculations
```

---

## 🚀 Installation

Getting GestureWave AI running locally only takes a few minutes. Follow the procedure below to configure your environment.

> [!NOTE]  
> **Prerequisites:** Ensure you have **Python 3.8+** installed and your operating system grants the terminal access to a working webcam.

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/GestureWave-AI.git
cd GestureWave-AI
```

**2. Create a Virtual Environment**
Generating an isolated environment ensures application stability and prevents dependency conflicts.
```bash
python -m venv venv
```

**3. Activate the Environment**
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

**4. Install Dependencies**
```bash
pip install -r requirements.txt
```

**5. Environment Configuration**
Create a `.env` file in the root directory to connect your Supabase backend:
```env
SUPABASE_URL=your_project_url
SUPABASE_ANON_KEY=your_anon_key
```

---

## 💻 Running the Project

GestureWave AI has two distinct entry points, depending on if you want to use the application normally or debug the pipeline.

> [!WARNING]  
> **Important Runtime Note:** Do not run `app.py` and `main.py` simultaneously. Both files will attempt to lock your machine's primary camera stream, resulting in a system error or crash.

### Normal Flow (Recommended)
Use the UI launcher to securely boot the program. You can adjust your preferences via the interface.
```bash
python app.py
```
*Click **"Start Tracking"** inside the UI window to boot the tracking pipeline.*

### Development Testing Flow
Use this flow if you are testing modifications to the recognition engine. It bypasses the launcher completely.
```bash
python main.py
```

---

## 👋 Gesture Reference Guide

### 🟢 Stable Core Gestures
The core gesture loop relies on absolute tracking primitives. These are highly optimized and stable across various lighting conditions.

| Intended Action | Physical Hand Shape | Description |
| :--- | :--- | :--- |
| **Move Cursor** | Index finger only | The cursor mimics the absolute 2D position of your index fingertip. |
| **Left Click** | Thumb + Index pinch | Tap the tips of your thumb and index finger together quickly. |
| **Pause / Resume**| Open palm | Completely stops cursor snapping until the open palm is recognized again. |

### 🟡 Experimental Gestures
These advanced gestures are available in the engine but may require threshold tuning in `core/config.py` depending on the focal length of your camera.

| Intended Action | Physical Hand Shape | Description |
| :--- | :--- | :--- |
| **Right Click** | Middle + Thumb pinch | Tap the tips of your middle finger and thumb together. |
| **Double Click** | Quick repeated pinch | Perform a standard left-click pinch twice in rapid succession. |
| **Scroll** | Peace sign | Point index and middle fingers up; move your hand vertically. |
| **Zoom** | Two fingers spread | Expand the distance between your index and thumb horizontally. |

### ⌨️ Testing & Override Controls
If you have the tracker window officially in focus, the following debug commands are active:
- Press **`R`** to instantly snapshot and record a custom target gesture to the registry.
- Press **`Esc`** to safely execute a shutdown and release the camera hook.

---

## 🛑 Known Limitations

- **Lighting Dependency:** Severe backlighting or ultra-low light can drastically reduce MediaPipe's confidence matrix.
- **Occlusion Errors:** If the thumb is hidden behind the palm (relative to the camera), pinch thresholds may fail to fire.

---

## 🔮 Future Improvements

- Implementation of dynamic smoothing filters (e.g., Euro filter mapping) to eliminate micro-jitter.
- Upgraded multi-hand architecture to support complicated, two-handed chorded shortcuts.
- Deep system tray integration logic for running entirely via background processes.

---

## 🏢 Use Cases

- **Presentation Spaces:** Control digital slideshows naturally while standing significantly away from your laptop.
- **Accessible Workstations:** Interact with standard UIs without requiring physical peripheral grips.
- **Hands-Free Prototyping:** A baseline foundation capable of integration into smart-mirrors, interactive art, and shop-floor manufacturing terminals.

---

## 📄 License & Usage

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
  <i>Control space, not screens. Developed by the GestureWave AI Team.</i>
</div>