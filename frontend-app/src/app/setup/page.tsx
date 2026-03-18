"use client";

import { motion } from "framer-motion";
import { Terminal, Package, Play, AlertTriangle, CheckCircle2, Copy, Monitor, Camera, Cpu, Fingerprint } from "lucide-react";
import { useState } from "react";

const steps = [
  {
    number: "01",
    icon: <Package className="w-5 h-5" />,
    title: "Clone the Repository",
    description: "Get the source code from GitHub onto your local machine.",
    code: `git clone https://github.com/yourusername/GestureWave-AI.git
cd GestureWave-AI`,
    lang: "bash",
  },
  {
    number: "02",
    icon: <Terminal className="w-5 h-5" />,
    title: "Create a Virtual Environment",
    description: "Isolate dependencies using a Python virtual environment.",
    code: `# Windows
python -m venv venv
venv\\Scripts\\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate`,
    lang: "bash",
  },
  {
    number: "03",
    icon: <Package className="w-5 h-5" />,
    title: "Install Dependencies",
    description: "Install all required Python packages with pip.",
    code: `pip install -r requirements.txt

# Or install manually:
pip install mediapipe opencv-python pyautogui numpy`,
    lang: "bash",
  },
  {
    number: "04",
    icon: <Play className="w-5 h-5" />,
    title: "Run GestureWave AI",
    description: "Launch the hand tracking engine and start controlling your cursor.",
    code: `python main.py`,
    lang: "bash",
  },
  {
    number: "05",
    icon: <Fingerprint className="w-5 h-5" />,
    title: "Record Custom Gestures",
    description: "Press 'R' in the camera window to record your own custom hand poses.",
    code: `# 1. Run main.py
# 2. Press 'R' on keyboard
# 3. Hold pose for 3 seconds
# 4. Gesture saved!`,
    lang: "bash",
  },
];

const requirements = [
  { icon: <Camera className="w-4 h-4" />, label: "Webcam", detail: "720p or higher recommended" },
  { icon: <Cpu className="w-4 h-4" />, label: "Python 3.8+", detail: "3.10 or 3.11 preferred" },
  { icon: <Monitor className="w-4 h-4" />, label: "OS", detail: "Windows 10/11, macOS 11+, Ubuntu 20.04+" },
];

const tips = [
  {
    type: "tip",
    title: "Good Lighting = Better Tracking",
    body: "Use a well-lit room with no strong backlight behind you. Front-facing light is ideal for clean hand detection.",
  },
  {
    type: "warning",
    title: "Permission Required on macOS",
    body: "On macOS, you must grant Accessibility permissions to your terminal app under System Settings → Privacy & Security.",
  },
  {
    type: "tip",
    title: "Camera Distance",
    body: "Position your hand approximately 30–70cm from the camera for optimal landmark accuracy.",
  },
  {
    type: "warning",
    title: "PyAutoGUI Failsafe",
    body: "Move your mouse to any screen corner to immediately halt cursor control if GestureWave AI becomes unresponsive.",
  },
];

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative mt-4 group">
      <div className="bg-gray-900 rounded-xl overflow-hidden border border-gray-800">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-800 bg-gray-950">
          <span className="text-xs text-gray-500 font-mono">{lang}</span>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            <Copy className="w-3.5 h-3.5" />
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <pre className="p-5 text-sm text-green-400 font-mono leading-relaxed overflow-x-auto whitespace-pre">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
}

export default function SetupPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen">
      <div className="max-w-4xl mx-auto px-6 py-20">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-50 border border-green-100 text-green-700 text-sm font-medium mb-6">
            <CheckCircle2 className="w-4 h-4" />
            Get Running in Under 3 Minutes
          </span>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6">
            Setup Guide
          </h1>
          <p className="text-lg text-gray-600 max-w-xl mx-auto leading-relaxed">
            GestureWave AI is a Python-based tool. Follow these steps to install it on Windows, macOS, or Linux.
          </p>
        </motion.div>

        {/* Requirements */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="mb-14 p-6 bg-white border border-gray-100 rounded-2xl shadow-sm"
        >
          <h2 className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-4">Requirements</h2>
          <div className="grid md:grid-cols-3 gap-4">
            {requirements.map((req, idx) => (
              <div key={idx} className="flex items-center gap-3 p-4 bg-gray-50 rounded-xl">
                <div className="w-9 h-9 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600">
                  {req.icon}
                </div>
                <div>
                  <p className="font-bold text-sm text-gray-900">{req.label}</p>
                  <p className="text-xs text-gray-500">{req.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Steps */}
        <div className="space-y-8 mb-16">
          {steps.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 + idx * 0.1, duration: 0.5 }}
              className="relative pl-8"
            >
              {/* Step connector line */}
              {idx < steps.length - 1 && (
                <div className="absolute left-4 top-12 bottom-0 w-0.5 bg-gradient-to-b from-blue-200 to-transparent" />
              )}

              <div className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all">
                <div className="flex items-start gap-4 mb-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-600 text-white flex items-center justify-center shrink-0 shadow-md shadow-blue-200">
                    {step.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-bold text-blue-400 tracking-widest font-mono">STEP {step.number}</span>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mt-0.5">{step.title}</h3>
                    <p className="text-gray-500 text-sm mt-1">{step.description}</p>
                  </div>
                </div>
                <CodeBlock code={step.code} lang={step.lang} />
              </div>
            </motion.div>
          ))}
        </div>

        {/* Tips & Warnings */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            Tips & Troubleshooting
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {tips.map((tip, idx) => (
              <div
                key={idx}
                className={`p-5 rounded-2xl border ${
                  tip.type === "warning"
                    ? "bg-amber-50 border-amber-100"
                    : "bg-blue-50 border-blue-100"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  {tip.type === "warning"
                    ? <AlertTriangle className="w-4 h-4 text-amber-600" />
                    : <CheckCircle2 className="w-4 h-4 text-blue-600" />
                  }
                  <p className={`font-bold text-sm ${tip.type === "warning" ? "text-amber-800" : "text-blue-800"}`}>
                    {tip.title}
                  </p>
                </div>
                <p className={`text-sm leading-relaxed ${tip.type === "warning" ? "text-amber-700" : "text-blue-700"}`}>
                  {tip.body}
                </p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Gesture Reference */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="bg-gray-900 text-white rounded-3xl p-8"
        >
          <h2 className="text-2xl font-bold mb-6 text-center">Gesture Reference</h2>
          <div className="grid md:grid-cols-3 gap-6 text-center">
            {[
              { gesture: "☝️ Index Finger", action: "Move Cursor" },
              { gesture: "🤏 Index + Thumb", action: "Left Click" },
              { gesture: "🤌 Middle + Thumb", action: "Right Click" },
              { gesture: "✌️ Peace Sign", action: "Scroll Up/Down" },
              { gesture: "✋ Open Palm", action: "Pause / Resume" },
              { gesture: "✨ Custom Pose", action: "Press 'R' to Record" },
            ].map((item, idx) => (
              <div key={idx} className="bg-white/5 border border-white/10 rounded-2xl p-5 hover:bg-white/10 transition-all">
                <div className="text-3xl mb-3">{item.gesture.split(" ")[0]}</div>
                <p className="text-xs text-gray-400 mb-1 font-mono">{item.gesture.replace(item.gesture.split(" ")[0] + " ", "")}</p>
                <p className="text-sm font-bold text-white">{item.action}</p>
              </div>
            ))}
          </div>
        </motion.div>

      </div>
    </div>
  );
}
