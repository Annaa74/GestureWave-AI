# GestureWave AI 🖐️
![GestureWave Banner](assets/banner.png)

### Touch-Free Gesture Control System

**GestureWave AI v2.1** is a real-time, touch-free hand gesture recognition engine that uses computer vision to control your PC cursor and actions with only a webcam.

## Overview

GestureWave AI is designed as a desktop gesture-control system that recognizes hand landmarks in real time and maps them to mouse interactions. The project includes:

- a **Python gesture engine**
- a **Tkinter-based desktop launcher**
- a **frontend showcase app** inside `frontend-app/`
- packaging support for Windows installer generation

## Current Stable Features

The most reliable features in the current build are:

- ☝️ **Index finger only** → Move cursor
- 🤏 **Thumb + index pinch** → Left click
- 🤏→ **Hold thumb + index pinch** → Drag and drop
- ✋ **Open palm** → Pause / resume tracking

## Experimental / In-Progress Gestures

The following gesture paths exist in the project but may still need tuning depending on camera quality, lighting, and runtime conditions:

- 🤌 **Middle + thumb pinch** → Right click
- ⚡ **Quick double pinch** → Double click
- ✌️ **Peace sign** → Scroll
- 🔍 **Two fingers spread** → Zoom in / out
- 🎯 **Custom recorded gesture** → Action trigger

## Tech Stack

- **AI / Vision**: Python, MediaPipe, OpenCV, NumPy
- **Desktop Control**: PyAutoGUI, Tkinter
- **Packaging / CI**: GitHub Actions, PyInstaller, Inno Setup
- **Frontend**: Next.js, React, Tailwind CSS, Framer Motion

## Project Structure

```text
GestureWave-AI/
├── app.py                 # Desktop launcher
├── main.py                # Gesture engine runtime
├── core/
│   ├── __init__.py
│   └── config.py          # Runtime configuration
├── gesture_registry.py    # Custom gesture storage / recognition
├── gesture_utils.py       # Landmark normalization helpers
├── feature_extraction.py  # Feature helpers
├── assets/                # Banner and static visuals
├── frontend-app/          # Next.js frontend
├── installer.iss          # Windows installer config
└── requirements.txt       # Python dependencies
Installation
1. Clone the repository
git clone https://github.com/Annaa74/GestureWave-AI.git
cd GestureWave-AI
2. Create and activate a virtual environment
Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Running the Project
Option 1: Run the desktop launcher
python app.py

Use this for the normal user flow. From the launcher, click Start Tracking.

Option 2: Run the gesture engine directly
python main.py

Use this when testing the gesture engine directly without the launcher UI.

Important Usage Note

Do not manually run app.py and main.py at the same time.

Recommended flow:

run python app.py
click Start Tracking from the launcher

OR

run python main.py directly for engine-only testing
Gesture Guide
Stable Core Gestures
Index finger only → Move cursor
Thumb + index pinch → Left click
Hold thumb + index pinch → Drag and drop
Open palm → Pause / resume tracking
Additional Gestures Present in the Codebase
Middle + thumb pinch → Right click
Quick repeated pinch → Double click
Peace sign → Scroll
Two fingers spread → Zoom
Press R → Record a custom gesture
Press Esc → Exit tracking
Recommended Testing Flow

For best results:

Use a well-lit environment
Keep only one hand in frame
Start with cursor movement first
Then test click
Then test drag
Test advanced gestures only after stable movement works
Known Issues
Gesture accuracy can vary with lighting and background clutter
Advanced gestures may need more tuning than core cursor controls
Running tracking alongside other heavy apps may reduce FPS
Webcam resolution and laptop performance can affect smoothness
Future Improvements
More reliable gesture classification pipeline
Better gesture debouncing and hysteresis
Persistent user settings
Cleaner module separation for gesture detection logic
Improved support for media control and shortcuts
License

This project is licensed under the terms of the included LICENSE
.