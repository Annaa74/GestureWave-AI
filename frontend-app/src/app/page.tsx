"use client";

import { motion } from "framer-motion";
import {
  MousePointer2, Waves, ScanEye, GitBranch,
  ArrowRight, CheckCircle2, Download, Monitor,
  Zap, Lock, Users, ChevronRight
} from "lucide-react";
import Link from "next/link";

const features = [
  {
    icon: <Waves className="w-6 h-6 text-blue-600" />,
    title: "Hand Tracking",
    description: "Real-time 21-point hand landmark detection using MediaPipe for precise 3D spatial mapping.",
  },
  {
    icon: <MousePointer2 className="w-6 h-6 text-indigo-600" />,
    title: "Cursor Control",
    description: "EMA-smoothed cursor movement with velocity-adaptive dampening — no jitter, no lag.",
  },
  {
    icon: <ScanEye className="w-6 h-6 text-violet-600" />,
    title: "8 Gesture Types",
    description: "Click, double-click, right-click, drag, scroll, zoom, and pause — all contactless.",
  },
];

const trustStats = [
  { value: "<50ms", label: "Response latency" },
  { value: "8",     label: "Gesture types" },
  { value: "100%",  label: "Offline & private" },
  { value: "3 min", label: "To get started" },
];

const steps = [
  { num: "1", title: "Download", body: "Get the GestureWave AI installer for your platform." },
  { num: "2", title: "Install", body: "Run the installer — no Python or dependencies needed." },
  { num: "3", title: "Wave",    body: "Launch, point your webcam, and start controlling your PC." },
];

export default function Home() {
  return (
    <div className="bg-[#fafafa] text-gray-900 selection:bg-blue-100">

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <div className="max-w-6xl mx-auto px-6 py-20 md:py-32">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-16 mb-24">
          <div className="flex-1 text-center lg:text-left">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium mb-6"
            >
              <CheckCircle2 className="w-4 h-4" />
              GestureWave AI v2.1 is now live
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.5 }}
              className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 leading-[1.1]"
            >
              Control your PC with <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
                just a wave.
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto lg:mx-0 mb-10 leading-relaxed"
            >
              GestureWave AI maps your hand movements to your cursor using computer vision — no hardware, no subscriptions, fully offline.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="flex flex-wrap items-center justify-center lg:justify-start gap-4"
            >
              <Link href="/setup">
                <button className="px-8 py-3.5 rounded-full bg-black text-white font-semibold hover:scale-105 active:scale-95 transition-all shadow-xl shadow-black/10 flex items-center gap-2">
                  <Download className="w-4 h-4" />
                  Download for Windows
                </button>
              </Link>
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                <button className="px-8 py-3.5 rounded-full bg-white border border-gray-200 hover:bg-gray-50 font-medium transition-all flex items-center gap-2 text-gray-700">
                  <GitBranch className="w-4 h-4" />
                  View Source
                </button>
              </a>
            </motion.div>

            {/* trust line */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="flex flex-wrap items-center gap-4 mt-8 justify-center lg:justify-start"
            >
              {[
                <><Lock className="w-3.5 h-3.5" /> 100% offline</>,
                <><Zap className="w-3.5 h-3.5" /> No account needed</>,
                <><Monitor className="w-3.5 h-3.5" /> Windows 10 / 11</>,
              ].map((item, i) => (
                <span key={i} className="flex items-center gap-1.5 text-xs text-gray-500 font-medium">
                  {item}
                </span>
              ))}
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4, duration: 0.7 }}
            className="flex-1 relative w-full aspect-square max-w-md mx-auto"
          >
            <div className="absolute inset-0 rounded-3xl bg-white border border-gray-100 overflow-hidden flex items-center justify-center shadow-2xl shadow-blue-900/5">
              <div className="relative w-full h-full p-8 flex flex-col items-center justify-center bg-gradient-to-br from-blue-50/50 to-indigo-50/50">
                <div className="w-48 h-48 rounded-full border-2 border-blue-100 flex items-center justify-center mb-8 relative bg-white shadow-sm">
                  <div className="absolute inset-0 rounded-full border border-blue-300 animate-ping opacity-20" style={{ animationDuration: "3s" }} />
                  <Waves className="w-20 h-20 text-blue-600 opacity-80" />
                </div>
                <div className="bg-white px-5 py-3 rounded-xl border border-gray-100 shadow-sm text-sm text-gray-600 font-mono flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  Hand detected. Tracking active.
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* ── Stats row ──────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.55, duration: 0.5 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-24"
        >
          {trustStats.map((s, i) => (
            <div key={i} className="text-center p-6 bg-white border border-gray-100 rounded-2xl shadow-sm">
              <p className="text-3xl font-bold text-gray-900 mb-1">{s.value}</p>
              <p className="text-xs text-gray-500">{s.label}</p>
            </div>
          ))}
        </motion.div>

        {/* ── How it works ───────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mb-24"
        >
          <div className="text-center mb-12">
            <span className="text-sm font-bold uppercase tracking-widest text-blue-600 mb-3 block">Get Running in 3 Steps</span>
            <h2 className="text-3xl md:text-4xl font-bold">Simple as it gets</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {steps.map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.65 + i * 0.1 }}
                className="relative p-7 bg-white border border-gray-100 rounded-2xl shadow-sm group hover:shadow-md transition-all"
              >
                {i < steps.length - 1 && (
                  <ChevronRight className="hidden md:block absolute -right-3 top-1/2 -translate-y-1/2 w-6 h-6 text-gray-300 z-10" />
                )}
                <div className="w-12 h-12 rounded-2xl bg-black text-white flex items-center justify-center text-xl font-bold mb-5 shadow-md group-hover:scale-110 transition-transform">
                  {step.num}
                </div>
                <h3 className="text-xl font-bold mb-2">{step.title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{step.body}</p>
              </motion.div>
            ))}
          </div>
          <div className="text-center mt-8">
            <Link href="/setup">
              <button className="inline-flex items-center gap-2 text-sm font-semibold text-blue-600 hover:underline">
                Full Setup Guide <ArrowRight className="w-4 h-4" />
              </button>
            </Link>
          </div>
        </motion.div>

        {/* ── Feature cards ──────────────────────────────────────── */}
        <div className="grid md:grid-cols-3 gap-8 mb-24">
          {features.map((feature, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 + idx * 0.1, duration: 0.5 }}
              className="p-8 rounded-2xl bg-white border border-gray-100 hover:shadow-lg transition-all shadow-sm group"
            >
              <div className="w-14 h-14 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-900">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* ── Community CTA strip ────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.5 }}
          className="flex flex-col md:flex-row items-center justify-between gap-6 p-7 bg-gradient-to-r from-indigo-600 to-blue-600 rounded-3xl text-white"
        >
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-white/10 border border-white/20 rounded-xl flex items-center justify-center">
              <Users className="w-6 h-6" />
            </div>
            <div>
              <p className="font-bold text-lg">Join the Community</p>
              <p className="text-blue-100 text-sm">Subscribe for release updates, feature previews, and community highlights.</p>
            </div>
          </div>
          <Link href="/community">
            <button className="shrink-0 px-7 py-3 bg-white text-blue-700 font-bold rounded-full hover:scale-105 active:scale-95 transition-all flex items-center gap-2 text-sm shadow-xl">
              Subscribe <ArrowRight className="w-4 h-4" />
            </button>
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
