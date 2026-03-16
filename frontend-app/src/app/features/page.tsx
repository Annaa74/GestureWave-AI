"use client";

import { motion } from "framer-motion";
import {
  Waves, MousePointer2, ScanEye, Zap, Shield, Cpu, Hand,
  MonitorSmartphone, Layers, GitMerge, ArrowRight, Check
} from "lucide-react";
import Link from "next/link";

const coreFeatures = [
  {
    icon: <Hand className="w-7 h-7 text-blue-600" />,
    title: "21-Point Hand Landmarks",
    description: "MediaPipe tracks 21 precise hand landmark points in real-time 3D space, giving you sub-millimeter precision for every gesture.",
    badge: "Core",
  },
  {
    icon: <MousePointer2 className="w-7 h-7 text-indigo-600" />,
    title: "Smooth Cursor Mapping",
    description: "Your index fingertip is mapped directly to screen coordinates using adaptive scaling — no jitter, no lag, just fluid control.",
    badge: "Control",
  },
  {
    icon: <ScanEye className="w-7 h-7 text-violet-600" />,
    title: "Pinch Gesture Clicks",
    description: "Pinch your index finger and thumb together to trigger left-clicks. A separate pinch pattern activates right-click for full mouse emulation.",
    badge: "Gesture",
  },
  {
    icon: <Waves className="w-7 h-7 text-blue-500" />,
    title: "Scroll Detection",
    description: "Two-finger vertical swipes trigger OS-level scroll events. Works in browsers, documents, and any scrollable interface.",
    badge: "Navigation",
  },
  {
    icon: <Zap className="w-7 h-7 text-amber-500" />,
    title: "< 50ms Latency",
    description: "Optimized OpenCV pipeline with frame skipping and gesture debouncing ensures near-instantaneous response across all hardware.",
    badge: "Performance",
  },
  {
    icon: <Shield className="w-7 h-7 text-green-600" />,
    title: "100% Offline & Private",
    description: "All processing happens locally on your machine. No data leaves your device, no cloud dependency, no tracking whatsoever.",
    badge: "Privacy",
  },
];

const techStack = [
  { name: "MediaPipe", role: "Hand landmark detection & 3D pose estimation", color: "blue" },
  { name: "OpenCV", role: "Camera capture, frame processing & visual overlay", color: "indigo" },
  { name: "PyAutoGUI", role: "OS-level cursor movement & click simulation", color: "violet" },
  { name: "NumPy", role: "Coordinate math, smoothing & gesture thresholds", color: "purple" },
];

const comparisons = [
  { feature: "Works without hardware peripherals", gw: true, traditional: false },
  { feature: "Contactless interaction", gw: true, traditional: false },
  { feature: "Real-time 3D hand tracking", gw: true, traditional: false },
  { feature: "100% local & private", gw: true, traditional: true },
  { feature: "Sub-50ms latency", gw: true, traditional: true },
  { feature: "Multi-gesture support", gw: true, traditional: false },
];

export default function FeaturesPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen">
      <div className="max-w-6xl mx-auto px-6 py-20">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-20"
        >
          <span className="inline-block px-4 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium mb-6">
            Platform Capabilities
          </span>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-tight">
            Everything you need to<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
              control without touching.
            </span>
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            GestureWave AI is engineered from the ground up for precision, speed, and privacy. Here&apos;s what&apos;s under the hood.
          </p>
        </motion.div>

        {/* Core Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-24">
          {coreFeatures.map((feature, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + idx * 0.08, duration: 0.5 }}
              className="p-7 rounded-2xl bg-white border border-gray-100 hover:shadow-lg hover:border-blue-100 transition-all group"
            >
              <div className="flex items-start justify-between mb-5">
                <div className="w-14 h-14 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center group-hover:scale-110 transition-transform">
                  {feature.icon}
                </div>
                <span className="text-xs font-semibold px-2.5 py-1 bg-gray-100 text-gray-600 rounded-full">
                  {feature.badge}
                </span>
              </div>
              <h3 className="text-lg font-bold mb-3 text-gray-900">{feature.title}</h3>
              <p className="text-gray-500 text-sm leading-relaxed">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="mb-24"
        >
          <div className="text-center mb-12">
            <div className="flex items-center justify-center gap-2 mb-4">
              <Cpu className="w-5 h-5 text-blue-600" />
              <span className="text-sm font-semibold text-blue-600 uppercase tracking-widest">Tech Stack</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold">Built on battle-tested libraries</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {techStack.map((tech, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + idx * 0.1 }}
                className="flex items-center gap-5 p-5 bg-white border border-gray-100 rounded-2xl hover:shadow-md transition-all"
              >
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm shrink-0">
                  {tech.name.slice(0, 2)}
                </div>
                <div>
                  <p className="font-bold text-gray-900">{tech.name}</p>
                  <p className="text-sm text-gray-500">{tech.role}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Comparison Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="mb-24"
        >
          <div className="text-center mb-12">
            <div className="flex items-center justify-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-indigo-600" />
              <span className="text-sm font-semibold text-indigo-600 uppercase tracking-widest">Comparison</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold">GestureWave vs. Traditional Input</h2>
          </div>
          <div className="bg-white border border-gray-100 rounded-2xl overflow-hidden shadow-sm">
            <div className="grid grid-cols-3 bg-gray-50 border-b border-gray-100 px-6 py-4 text-sm font-bold text-gray-700">
              <span>Feature</span>
              <span className="text-center text-blue-600">GestureWave AI</span>
              <span className="text-center">Traditional Mouse</span>
            </div>
            {comparisons.map((row, idx) => (
              <div
                key={idx}
                className={`grid grid-cols-3 px-6 py-4 border-b border-gray-50 last:border-0 items-center ${idx % 2 === 0 ? "bg-white" : "bg-gray-50/30"}`}
              >
                <span className="text-sm text-gray-700">{row.feature}</span>
                <div className="flex justify-center">
                  {row.gw
                    ? <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center"><Check className="w-3.5 h-3.5 text-green-600" /></div>
                    : <div className="w-6 h-6 rounded-full bg-red-100 flex items-center justify-center"><span className="text-red-500 text-xs font-bold">✕</span></div>
                  }
                </div>
                <div className="flex justify-center">
                  {row.traditional
                    ? <div className="w-6 h-6 rounded-full bg-green-100 flex items-center justify-center"><Check className="w-3.5 h-3.5 text-green-600" /></div>
                    : <div className="w-6 h-6 rounded-full bg-red-100 flex items-center justify-center"><span className="text-red-500 text-xs font-bold">✕</span></div>
                  }
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="text-center bg-gradient-to-br from-blue-600 to-indigo-700 rounded-3xl p-14 text-white"
        >
          <GitMerge className="w-10 h-10 mx-auto mb-5 opacity-80" />
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to try it out?</h2>
          <p className="text-blue-100 mb-8 max-w-md mx-auto">Follow our setup guide to install GestureWave AI on your machine in under 3 minutes.</p>
          <Link href="/setup">
            <button className="px-8 py-3.5 bg-white text-blue-700 font-bold rounded-full hover:scale-105 active:scale-95 transition-all shadow-xl flex items-center gap-2 mx-auto">
              Setup Guide <ArrowRight className="w-4 h-4" />
            </button>
          </Link>
        </motion.div>

      </div>
    </div>
  );
}
