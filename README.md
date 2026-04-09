# GestureWave AI 🖐️  
![GestureWave Banner](assets/banner.png)

## **Touch-Free Gesture Control System**

**GestureWave AI v2.1** is a real-time **computer-vision-based gesture control system** that allows users to interact with their PC using only a webcam and hand gestures — no extra sensors, gloves, or hardware required.

It combines **MediaPipe hand tracking**, **OpenCV-based video processing**, and **PyAutoGUI desktop actions** to turn hand poses into live cursor movement and gesture-triggered actions.

---

# **✨ Project Highlights**

## **What GestureWave AI does**
GestureWave AI detects and tracks hand landmarks in real time, interprets selected hand gestures, and maps them to desktop controls such as:

- **Cursor movement**
- **Left click**
- **Right click**
- **Double click**
- **Drag and drop**
- **Scroll**
- **Zoom**
- **Pause / resume tracking**

It also includes a **desktop launcher GUI**, **custom gesture support**, and a **frontend showcase app**.

---

# **🚀 Core Features**

## **1. Real-Time Hand Tracking**
Uses **MediaPipe Hands** to detect and track a single hand in real time using a webcam feed.

## **2. Touch-Free Cursor Control**
Maps hand motion to cursor movement with smoothing and motion filtering.

## **3. Native Gesture Actions**
Supports gesture-triggered desktop interactions such as:

- ☝️ **Index finger only** → Move cursor  
- 🤏 **Thumb + index pinch** → Left click  
- 🤌 **Middle + thumb pinch** → Right click  
- ⚡ **Quick repeated pinch** → Double click  
- 🤏→ **Hold pinch + move** → Drag and drop  
- ✌️ **Peace sign** → Scroll  
- 🔍 **Two-finger spread** → Zoom in / out  
- ✋ **Open palm** → Pause / resume tracking  

## **4. Desktop Launcher**
Includes a **Tkinter-based control dashboard** with:
- status display
- tracking controls
- gesture reference
- settings access
- live interaction feedback

## **5. Custom Gesture Recording**
Users can press **`R`** during runtime to record and store custom gestures for later action mapping.

## **6. Windows Packaging Support**
Includes installer and deployment files for building a Windows executable workflow.

## **7. Frontend Showcase**
A **Next.js frontend** is included in the `frontend-app/` folder for product presentation and project showcase.

---

# **🛠️ Tech Stack**

## **AI / Vision**
- **Python**
- **MediaPipe**
- **OpenCV**
- **NumPy**

## **Desktop Interaction**
- **PyAutoGUI**
- **Tkinter**

## **Packaging / Automation**
- **GitHub Actions**
- **PyInstaller**
- **Inno Setup**

## **Frontend**
- **Next.js**
- **React**
- **Tailwind CSS**
- **Framer Motion**

---

# **📁 Project Structure**

```text
GestureWave-AI/
├── app.py                 # Desktop launcher UI
├── main.py                # Gesture engine runtime
├── core/
│   ├── __init__.py
│   └── config.py          # Runtime configuration
├── gesture_registry.py    # Custom gesture storage and matching
├── gesture_utils.py       # Landmark normalization helpers
├── feature_extraction.py  # Gesture feature utilities
├── assets/                # Images, banner, visuals
├── frontend-app/          # Next.js frontend showcase
├── installer.iss          # Windows installer configuration
├── requirements.txt       # Python dependencies
└── README.md
⚙️ Installation
Prerequisites

Before running the project, make sure you have:

Python 3.11 installed
A working webcam
Windows PowerShell or terminal access
Internet connection for dependency installation
Step 1 — Clone the Repository
git clone https://github.com/Annaa74/GestureWave-AI.git
cd GestureWave-AI
Step 2 — Create a Virtual Environment
Windows PowerShell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

After activation, your terminal should show something like:

(venv) PS C:\path\to\GestureWave-AI>
Step 3 — Install Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Note:
If you face dependency conflicts related to NumPy or OpenCV, make sure the versions in requirements.txt match the project’s tested runtime setup.

▶️ Running the Project
Option 1 — Run the Desktop Launcher

Use this for the normal application flow:

python app.py

Then click Start Tracking from the launcher.

Option 2 — Run the Gesture Engine Directly

Use this if you want to test the camera tracking engine without the launcher UI:

python main.py
❗ Important Runtime Note

Do not run app.py and main.py manually at the same time.

Use one of these flows:

Launcher Flow
python app.py

Then press Start Tracking

Engine-Only Flow
python main.py

Running both separately at once can cause:

camera conflicts
duplicate tracking windows
unstable behavior
lower FPS
🖐️ Gesture Reference
Stable / Primary Gestures

These are the most important gestures to understand first:

Gesture	Action
Index finger only	Move cursor
Thumb + index pinch	Left click
Hold thumb + index pinch	Drag and drop
Open palm	Pause / resume tracking
Additional Gestures Available

These are supported in the project logic, though some may require extra tuning depending on environment and runtime conditions:

Gesture	Action
Middle + thumb pinch	Right click
Quick repeated pinch	Double click
Peace sign	Scroll
Two fingers spread	Zoom in / out
Press R	Record custom gesture
Press Esc	Exit tracking
🧪 Recommended Testing Flow

For the best testing experience:

1. Start simple

First test only:

cursor movement
left click
drag
2. Use a clean camera environment

For better tracking:

keep the background simple
avoid low light
avoid strong backlight behind your hand
3. Keep one hand in frame

The system is currently optimized for single-hand interaction.

4. Test advanced gestures later

Only after movement and clicking feel stable should you test:

right click
scroll
zoom
custom gestures
📌 Known Limitations

GestureWave AI is functional, but still evolving. Current limitations may include:

reduced accuracy in poor lighting
lower FPS on weaker systems
advanced gestures requiring more tuning than core gestures
occasional gesture overlap depending on hand angle and distance from camera
webcam quality affecting performance and stability
📈 Future Improvements

Planned improvements include:

better gesture classification accuracy
reduced false positives
stronger debouncing and hysteresis logic
persistent runtime settings
cleaner modular architecture
improved media control gesture support
more stable custom gesture mapping
💡 Use Cases

GestureWave AI can be useful for:

accessibility-focused input systems
touch-free computer interaction demos
gesture-based HCI experiments
AI / CV project showcases
desktop automation prototypes
📜 License

This project is licensed under the terms of the included LICENSE
.

🙌 Acknowledgment

Built as a real-time hand-gesture interaction system using modern computer vision and desktop automation tools.