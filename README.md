# GestureWave AI 🖐️
### Touch-Free Gesture Control System

**GestureWave AI v2.1** is a real-time, touch-free hand gesture recognition engine that maps your hand movements to your PC cursor and actions using computer vision—no extra hardware required.

## Key Features
- **Premium GUI Dashboard**: Full settings menu to control smoothing, dead zones, click sensitivity, and live event logs smoothly wrapped in a Dark Mode Tkinter UI.
- **8 Native Gestures**: 
  - ☝️ **Move cursor**
  - 🤏 **Left click**
  - 🤌 **Right click**
  - ⚡ **Double click**
  - 🤏→ **Drag and Drop**
  - ✌️ **Scroll up/down**
  - 🔍 **Zoom in/out**
  - ✋ **Pause tracking**
- **Custom App Shortcuts**: Press `R` in front of your camera to record custom hand poses. (Currently, `Gesture_1` automatically launches LinkedIn!)
- **Windows Installer**: One-click `.exe` builder via GitHub actions, entirely deployable. 
- **Next.js Frontend**: Included documentation, feature showcase, and community routing inside the `/frontend-app/` directory.

## Tech Stack
- **AI/Vision**: Python, MediaPipe, OpenCV, NumPy
- **Desktop Control**: PyAutoGUI, Tkinter
- **Continuous Integration**: GitHub Actions, PyInstaller, Inno Setup (ISCC)
- **Frontend App**: Next.js, React, Tailwind CSS, Framer Motion

## Installation & Usage

**Method 1: Desktop Installer (Easiest)**
1. Check the GitHub Actions artifacts or your repository releases.
2. Download `GestureWaveAI_Installer.exe` and follow the standard Windows setup wizard.
3. Launch directly from your start menu.

**Method 2: Run from Source (For Developers)**
```bash
# Safely restrict NumPy to prevent MediaPipe architecture clashes
pip install "numpy<2" -r requirements.txt

# Start the premium GUI Launcher
python app.py
```
