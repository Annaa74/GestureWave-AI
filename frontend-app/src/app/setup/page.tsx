"use client";

import { motion } from "framer-motion";
import {
  Download, Monitor, Camera, Waves,
  MousePointer2, MousePointerClick, Scroll,
  Pause, Maximize, Save, ArrowRight,
  Info, CheckCircle2, AlertTriangle, ShieldCheck
} from "lucide-react";
import Link from "next/link";

const mainSteps = [
  {
    number: "01",
    icon: <Download className="w-6 h-6" />,
    title: "Download for Windows",
    description: "Download the GestureWave AI v2.1 installer. It's fully bundled — no extra software or Python needed.",
    button: {
      text: "Download Installer",
      link: "https://github.com/Annaa74/GestureWave-AI/releases/latest/download/GestureWaveAI_Installer.exe",
      primary: true
    }
  },
  {
    number: "02",
    icon: <Monitor className="w-6 h-6" />,
    title: "One-Click Install",
    description: "Run the GestureWaveAI_Installer.exe and follow the wizard. It takes less than 30 seconds to set up.",
    button: null
  },
  {
    number: "03",
    icon: <Camera className="w-6 h-6" />,
    title: "Point your Webcam",
    description: "Launch the app from your Start Menu. Ensure your hand is visible to your webcam at a natural distance.",
    button: null
  },
];

const gestureGuides = [
  {
    icon: <MousePointer2 className="w-6 h-6 text-blue-600" />,
    gesture: "☝️ Index Finger",
    action: "Move Cursor",
    description: "Point your index finger and move it in space to guide the cursor across your screen.",
    color: "bg-blue-50 border-blue-100"
  },
  {
    icon: <MousePointerClick className="w-6 h-6 text-indigo-600" />,
    gesture: "🤏 Index + Thumb",
    action: "Left Click",
    description: "Pinch index and thumb to trigger a left click on your screen.",
    color: "bg-indigo-50 border-indigo-100"
  },
  {
    icon: <MousePointerClick className="w-6 h-6 text-violet-600" />,
    gesture: "🤌 Middle + Thumb",
    action: "Right Click",
    description: "Pinch middle finger and thumb to trigger a right-click context menu.",
    color: "bg-violet-50 border-violet-100"
  },
  {
    icon: <Scroll className="w-6 h-6 text-emerald-600" />,
    gesture: "✌️ Peace Sign",
    action: "Scroll Up / Down",
    description: "Show a peace sign. Move it slightly up to scroll up, or down to scroll down.",
    color: "bg-emerald-50 border-emerald-100"
  },
  {
    icon: <Maximize className="w-6 h-6 text-orange-600" />,
    gesture: "🔍 Two Fingers Spread",
    action: "Zoom In / Out",
    description: "Spread your index and middle fingers apart to zoom into browsers and photos.",
    color: "bg-orange-50 border-orange-100"
  },
  {
    icon: <Pause className="w-6 h-6 text-gray-600" />,
    gesture: "✋ Open Palm",
    action: "Pause / Resume",
    description: "Raise an open palm to temporarily pause tracking. Raising it again resumes control.",
    color: "bg-gray-100 border-gray-200"
  },
];

const tips = [
  {
    icon: <Info className="w-4 h-4 text-blue-600" />,
    title: "Emergency Stop",
    body: "If the cursor goes wild, simply move your real mouse to any screen corner to trigger the failsafe."
  },
  {
    icon: <ShieldCheck className="w-4 h-4 text-green-600" />,
    title: "Privacy First",
    body: "GestureWave AI runs 100% offline. No camera feeds are ever recorded or sent to the cloud."
  },
  {
    icon: <Save className="w-4 h-4 text-purple-600" />,
    title: "Custom Shortcuts",
    body: "Press 'R' in the live window to record a custom hand pose. Use it to launch apps like LinkedIn!"
  }
];

export default function SetupPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen selection:bg-blue-100">
      <div className="max-w-6xl mx-auto px-6 py-20">

        {/* ── Header ─────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-xs font-bold uppercase tracking-wider mb-6">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Zero Coding Required
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            Ready to <span className="text-blue-600">Wave?</span>
          </h1>
          <p className="text-lg md:text-xl text-gray-500 max-w-2xl mx-auto leading-relaxed">
            Follow this simple guide to get GestureWave AI running on your Windows PC and learn the gestural language of the future.
          </p>
        </motion.div>

        {/* ── Quick Start Steps ───────────────────────────────────────── */}
        <div className="grid md:grid-cols-3 gap-8 mb-32">
          {mainSteps.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="relative p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-xl hover:shadow-blue-900/5 transition-all group"
            >
              <div className="absolute top-0 right-0 p-6 text-6xl font-black text-gray-50 opacity-10 group-hover:opacity-20 transition-opacity">
                {step.number}
              </div>
              <div className="w-14 h-14 rounded-2xl bg-black text-white flex items-center justify-center mb-8 shadow-lg shadow-black/10 group-hover:scale-110 transition-transform">
                {step.icon}
              </div>
              <h3 className="text-xl font-bold mb-3">{step.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed mb-8">
                {step.description}
              </p>
              {step.button && (
                <Link href={step.button.link}>
                  <button className="w-full py-3 rounded-xl bg-blue-600 text-white font-bold text-sm hover:bg-blue-700 transition-colors flex items-center justify-center gap-2">
                    <Download className="w-4 h-4" />
                    {step.button.text}
                  </button>
                </Link>
              )}
            </motion.div>
          ))}
        </div>

        {/* ── Gesture Interaction (The Core Focus) ──────────────────── */}
        <div className="mb-32">
          <div className="flex flex-col md:flex-row items-end justify-between gap-6 mb-12">
            <div className="max-w-xl">
              <h2 className="text-3xl md:text-4xl font-bold mb-4 flex items-center gap-3">
                <Waves className="w-8 h-8 text-blue-600" />
                Interacting with GestureWave
              </h2>
              <p className="text-gray-500">
                Master the native gestures included in v2.1. These are designed to feel natural and work consistently across all apps.
              </p>
            </div>
            <div className="flex gap-2">
              <div className="px-4 py-2 rounded-full bg-green-50 text-green-700 text-xs font-bold border border-green-100">
                8 Native Gestures
              </div>
              <div className="px-4 py-2 rounded-full bg-indigo-50 text-indigo-700 text-xs font-bold border border-indigo-100">
                100% Customizable
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {gestureGuides.map((g, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: idx * 0.05 }}
                className={`p-7 rounded-3xl border ${g.color} transition-all hover:scale-[1.02] cursor-default group`}
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="p-3 bg-white rounded-xl shadow-sm group-hover:rotate-6 transition-transform">
                    {g.icon}
                  </div>
                  <div className="text-xs font-bold uppercase tracking-widest text-gray-400">
                    Gesture
                  </div>
                </div>
                <h4 className="text-sm font-bold text-gray-500 mb-1">{g.gesture}</h4>
                <h3 className="text-2xl font-bold mb-3">{g.action}</h3>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {g.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* ── Tips & Failsafes ───────────────────────────────────────── */}
        <div className="grid md:grid-cols-3 gap-6 items-stretch">
          {tips.map((tip, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              className="p-6 bg-white border border-gray-100 rounded-2xl shadow-sm flex flex-col items-center text-center"
            >
              <div className="w-10 h-10 rounded-full bg-gray-50 border border-gray-100 flex items-center justify-center mb-4">
                {tip.icon}
              </div>
              <h5 className="font-bold text-sm mb-2">{tip.title}</h5>
              <p className="text-xs text-gray-500 leading-relaxed">
                {tip.body}
              </p>
            </motion.div>
          ))}
        </div>

        {/* ── Final CTA ──────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="mt-32 p-12 bg-black rounded-3xl text-center text-white overflow-hidden relative"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-indigo-600/20 pointer-events-none" />
          <h2 className="text-4xl md:text-5xl font-bold mb-6 relative">Still have questions?</h2>
          <p className="text-gray-400 max-w-xl mx-auto mb-10 text-lg relative">
            Our community is here to help. Check out the community hub for tutorials, custom gesture packs, and developer logs.
          </p>
          <div className="flex flex-wrap justify-center gap-4 relative">
             <Link href="/community">
              <button className="px-10 py-4 bg-white text-black font-bold rounded-full hover:scale-105 transition-all flex items-center gap-2">
                Join Community <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
