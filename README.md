<div align="center">
  <img src="assets/banner.png" alt="GestureWave AI Banner" width="100%" />

  # GestureWave AI
  **Real-Time, Touch-Free Gesture Control System**

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
  [![MediaPipe](https://img.shields.io/badge/MediaPipe-Enabled-orange.svg)](https://developers.google.com/mediapipe)
  [![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)](https://opencv.org/)
  [![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)](https://gesture-wave-ai.vercel.app/)
  [![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

  [**Explore the Live Dashboard →**](https://gesture-wave-ai.vercel.app/)
</div>

---

## 🌊 Overview

**GestureWave AI (v2.1)** is an advanced, touch-free gesture control engine that maps physical hand movements to digital desktop actions in real-time. By leveraging computer vision and machine learning models, it establishes a seamless, invisible interface between you and your computer—turning your webcam into a high-precision peripheral.

Whether you are presenting to a large room, experimenting with spatial interfaces, or building accessible environments, GestureWave AI pushes the boundaries of human-computer interaction (HCI) using standard hardware.

---

## ✨ Features

- **Zero-Latency Tracking:** High-speed hand landmark detection optimized for standard CPU execution using MediaPipe.
- **Premium GUI Dashboard:** A sleek, Dark Mode desktop launcher (`app.py`) with real-time logs, status indicators, and live settings adjustment.
- **Advanced Cursor Smoothing:** Implements **Exponential Moving Average (EMA)** and **Velocity-adaptive dampening** to eliminate jitter while maintaining surgical precision.
- **Dead Zone Suppression:** Intelligent filtering that ignores micro-tremors for a stable productivity experience.
- **Custom Gesture Engine:** Built-in capability to record (`R` key), extract features, and register new multi-dimensional gestures on the fly.
- **Web Showcase Dashboard:** A full-scale Next.js application (`/frontend-app/`) featuring deep documentation and a community portal.

---

## 🛠️ Tech Stack

This system is built entirely on a robust ecosystem specifically chosen for inference speed and operational stability.

| Technology | Core Responsibility |
| :--- | :--- |
| **Python 3.8+** | Primary application logic and state management |
| **MediaPipe** | Sub-millisecond hand tracking and topology extraction |
| **OpenCV** | Matrix manipulation, frame streaming, and HUD rendering |
| **PyAutoGUI** | Operating-system-level simulated mouse and keyboard control |
| **Tkinter** | Lightweight, high-performance configuration and launcher UI |
| **NumPy** | High-speed vector math and coordinate transformations |
| **Next.js 15** | Modern web dashboard and documentation showcase |
| **GitHub Actions** | Automated CI/CD pipeline for generating Windows executables |

---

## 🏗️ Project Structure

The repository is highly modular, deeply separating the vision processing pipeline from operating system overrides.

```text
GestureWave-AI/
├── app.py                 # Premium Desktop launcher (Tkinter)
├── main.py                # Core Gesture Engine runtime (MediaPipe/OpenCV)
├── core/
│   └── config.py          # Global runtime & threshold configurations
├── gesture_registry.py    # Custom gesture storage and matching logic
├── gesture_utils.py       # Landmark normalization and spatial helpers
├── feature_extraction.py  # Advanced feature vector calculations
├── frontend-app/          # Next.js 15 frontend showcase & metrics dashboard
├── installer.iss          # Windows execution installer config (Inno Setup)
├── assets/                # Visual assets, banners, and interface icons
└── requirements.txt       # Python dependency manifest
```

---

## ✅ Work Accomplished (Milestones)

- [x] **Core Engine (v1.0):** Basic hand tracking and cursor mapping implemented.
- [x] **GUI Launcher (v2.0):** Developed the Dark Mode Tkinter interface with threading to prevent UI freezing during tracking.
- [x] **Precision Layer:** Added Velocity-adaptive EMA smoothing and Dead Zone thresholds to solve the "shaky cursor" problem.
- [x] **Gesture Registry:** Implemented a recording pipeline that allows users to save custom hand poses to a local registry.
- [x] **Web Integration:** Launched a high-fidelity Next.js landing page with Framer Motion animations.
- [x] **Packaging:** Configured `Inno Setup` and `PyInstaller` for a seamless `.exe` distribution experience.

---

## 🚀 Installation

Getting GestureWave AI running locally only takes a few minutes. Follow the procedure below to configure your environment.

> [!NOTE]  
> **Prerequisites:** Ensure you have **Python 3.8+** installed and your operating system grants the terminal access to a working webcam.

**1. Clone the repository**
```bash
git clone https://github.com/Annaa74/GestureWave-AI.git
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

---

## 💻 Running the Project

GestureWave AI has two distinct entry points, depending on if you want to use the application normally or debug the pipeline.

> [!WARNING]  
> **Important Runtime Note:** Do not run `app.py` and `main.py` simultaneously. Both files will attempt to lock your machine's primary camera stream, resulting in a system error or crash.

### Normal Flow (Recommended)
Use the UI launcher to securely boot the program. You can adjust your smoothing and sensitivity preferences via the interface.
```bash
python app.py
```
*Click **"Start Tracking"** inside the UI window to boot the tracking pipeline.*

### Development Testing Flow
Use this flow if you are testing modifications to the recognition engine. It bypasses the launcher completely and prints debug logs directly to the console.
```bash
python main.py
```

---

## 👋 Gesture Reference Guide

### 🟢 Stable Core Gestures
The core gesture loop relies on absolute tracking primitives. These are highly optimized and stable across various lighting conditions.

| Emoji | Intended Action | Physical Hand Shape | Description |
| :--- | :--- | :--- | :--- |
| ☝️ | **Move Cursor** | Index finger only | Cursor mimics the absolute 2D position of your index fingertip. |
| 🤏 | **Left Click** | Thumb + Index pinch | Tap the tips of your thumb and index finger together. |
| 🤌 | **Right Click** | Middle + Thumb pinch | Tap the tips of your middle finger and thumb together. |
| ⚡ | **Double Click**| Quick double pinch | Perform two rapid thumb-index pinches in succession. |
| 🤏→ | **Drag & Drop** | Hold Thumb+Index | Pinch items and keep fingers held to move; release to drop. |
| ✌️ | **Scroll** | Peace sign | Point index/middle up; move hand vertically to scroll. |
| 🔍 | **Zoom** | Two fingers spread | Expand/contract distance between index and middle fingers. |
| ✋ | **Pause/Resume**| Open palm | Completely toggles tracking on or off. |

### ⌨️ Testing & Override Controls
If the tracker window is in focus, the following debug commands are active:
- Press **`R`** to instantly snapshot and record a custom target gesture to the registry.
- Press **`Esc`** to safely execute a shutdown and release the camera hook.

---

## 🛑 Known Limitations

- **Lighting Dependency:** Severe backlighting or ultra-low light can drastically reduce MediaPipe's confidence matrix.
- **Occlusion Errors:** If the thumb is hidden behind the palm (relative to the camera), pinch thresholds may fail to fire.
- **Numpy Versioning:** Requires `numpy < 2.0` due to MediaPipe's internal architecture constraints.

---

## 🔮 Future Scope

- **AI Smoothing (Euro Filter):** Implementation of 1€ filter logic for even smoother, lag-free heavy movement.
- **Multi-Hand Logic:** Supporting two-handed chorded shortcuts (e.g., left hand for modifiers like Ctrl/Shift, right hand for navigation).
- **System Tray Integration:** Ability to run the engine entirely in the background as a Windows Service.
- **Voice-Gesture Fusion:** Integrating voice commands (Whisper AI) to complement hand gestures for a truly multimodal HCI experience.

---

## 🏢 Use Cases

- **Presentation Spaces:** Control digital slideshows naturally while standing significantly away from your laptop.
- **Accessible Workstations:** Interact with standard UIs without requiring physical peripheral grips or fine motor mouse control.
- **Hands-Free Prototyping:** A foundation for integration into smart-mirrors, interactive art, and sterile manufacturing environments.

---

## 📄 License & Usage

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
  <i>Control space, not screens. Developed by the GestureWave AI Team.</i>
</div>
