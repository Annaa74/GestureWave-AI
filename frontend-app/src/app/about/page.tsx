"use client";

import { motion } from "framer-motion";
import { Github, Heart, Lightbulb, Code2, Globe, Star, GitFork, ArrowRight } from "lucide-react";

const milestones = [
  {
    year: "2024 Q1",
    title: "The Idea",
    description: "Born from frustration — a desire to control a computer without ever touching a mouse during a presentation gone wrong.",
  },
  {
    year: "2024 Q2",
    title: "First Prototype",
    description: "MediaPipe integrated with OpenCV. First successful cursor movement tracked from a webcam. Shaky, but it worked.",
  },
  {
    year: "2024 Q3",
    title: "Gesture Engine v0.5",
    description: "Added click detection via pinch gestures. Smoothing algorithms introduced to eliminate jitter.",
  },
  {
    year: "2024 Q4",
    title: "Open Source Release",
    description: "GestureWave AI v1.0 released on GitHub. Community contributions begin. Scroll and double-click support added.",
  },
  {
    year: "2025",
    title: "You Are Here",
    description: "Next phase: multi-hand support, custom gesture macros, and integration with accessibility APIs.",
  },
];

const values = [
  {
    icon: <Heart className="w-6 h-6 text-red-500" />,
    title: "Accessibility First",
    description: "GestureWave was built to empower people who struggle with traditional input devices. Accessibility is not a feature — it's the mission.",
  },
  {
    icon: <Lightbulb className="w-6 h-6 text-amber-500" />,
    title: "Open Innovation",
    description: "All code is public. We believe the best solutions are built collaboratively and transparently.",
  },
  {
    icon: <Code2 className="w-6 h-6 text-blue-500" />,
    title: "Code Quality",
    description: "Clean, readable Python with comprehensive documentation so anyone can understand, fork, and improve it.",
  },
  {
    icon: <Globe className="w-6 h-6 text-green-500" />,
    title: "Privacy by Default",
    description: "No cloud. No telemetry. No accounts. Your camera feed never leaves your machine.",
  },
];

const stats = [
  { value: "21", label: "Hand landmarks tracked" },
  { value: "<50ms", label: "Average response latency" },
  { value: "6+", label: "Gesture types supported" },
  { value: "3 min", label: "Average setup time" },
];

export default function AboutPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen">
      <div className="max-w-5xl mx-auto px-6 py-20">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-20"
        >
          <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-50 border border-purple-100 text-purple-700 text-sm font-medium mb-6">
            <Heart className="w-4 h-4" />
            Passion Project
          </span>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-tight">
            Built with purpose.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">
              Shared with the world.
            </span>
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            GestureWave AI is an open-source project dedicated to making computers more accessible, intuitive, and futuristic — one gesture at a time.
          </p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-24"
        >
          {stats.map((stat, idx) => (
            <div key={idx} className="text-center p-6 bg-white border border-gray-100 rounded-2xl shadow-sm">
              <p className="text-3xl md:text-4xl font-bold text-gray-900 mb-1">{stat.value}</p>
              <p className="text-xs text-gray-500 leading-snug">{stat.label}</p>
            </div>
          ))}
        </motion.div>

        {/* Mission + Creator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="grid md:grid-cols-2 gap-10 mb-24 items-center"
        >
          <div>
            <span className="text-sm font-bold uppercase tracking-widest text-purple-600 mb-3 block">The Mission</span>
            <h2 className="text-3xl font-bold mb-5">Human-computer interaction, reimagined</h2>
            <p className="text-gray-600 leading-relaxed mb-4">
              We&apos;ve used mice and keyboards for over 40 years. GestureWave AI asks a simple question: what if you could just... wave?
            </p>
            <p className="text-gray-600 leading-relaxed mb-4">
              By leveraging Google&apos;s MediaPipe and OpenCV, we&apos;ve created a real-time hand tracking pipeline that maps your natural movements directly to cursor input — no specialized hardware required.
            </p>
            <p className="text-gray-600 leading-relaxed">
              The project is particularly meaningful for users with limited mobility, presenters who need hands-free control, and technologists pushing the boundaries of HCI.
            </p>
          </div>

          <div className="bg-gradient-to-br from-purple-600 to-indigo-700 rounded-3xl p-8 text-white">
            <div className="w-16 h-16 rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center mb-6">
              <Code2 className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold mb-2">Open Source & Free</h3>
            <p className="text-purple-100 text-sm leading-relaxed mb-6">
              GestureWave AI is MIT licensed. Use it, modify it, build on it. No strings attached.
            </p>
            <div className="flex items-center gap-4">
              <a
                href="https://github.com/Annaa74/GestureWave-AI"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-4 py-2.5 bg-white text-purple-700 font-bold text-sm rounded-full hover:scale-105 transition-all"
              >
                <Github className="w-4 h-4" />
                View on GitHub
              </a>
              <div className="flex items-center gap-3 text-purple-200 text-sm">
                <a href="https://github.com/Annaa74/GestureWave-AI" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-white transition-colors">
                  <Star className="w-4 h-4" /> Star us
                </a>
                <a href="https://github.com/Annaa74/GestureWave-AI/fork" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-white transition-colors">
                  <GitFork className="w-4 h-4" /> Fork it
                </a>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Values */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mb-24"
        >
          <div className="text-center mb-12">
            <span className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3 block">What We Stand For</span>
            <h2 className="text-3xl md:text-4xl font-bold">Our core values</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            {values.map((value, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + idx * 0.1 }}
                className="p-6 bg-white border border-gray-100 rounded-2xl hover:shadow-md transition-all group"
              >
                <div className="w-12 h-12 rounded-xl bg-gray-50 border border-gray-100 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  {value.icon}
                </div>
                <h3 className="font-bold text-lg mb-2">{value.title}</h3>
                <p className="text-gray-500 text-sm leading-relaxed">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Timeline */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mb-24"
        >
          <div className="text-center mb-12">
            <span className="text-sm font-bold uppercase tracking-widest text-gray-500 mb-3 block">Project History</span>
            <h2 className="text-3xl md:text-4xl font-bold">The story so far</h2>
          </div>
          <div className="relative">
            <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-purple-400 via-indigo-300 to-transparent" />
            <div className="space-y-8">
              {milestones.map((milestone, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + idx * 0.1 }}
                  className="flex gap-6 pl-2"
                >
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 text-white flex items-center justify-center text-xs font-bold shrink-0 shadow-md shadow-purple-200 z-10">
                    {idx + 1}
                  </div>
                  <div className="pb-2">
                    <span className="text-xs font-bold text-purple-500 font-mono">{milestone.year}</span>
                    <h3 className="text-lg font-bold text-gray-900 mt-0.5 mb-1">{milestone.title}</h3>
                    <p className="text-gray-500 text-sm leading-relaxed">{milestone.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 0.5 }}
          className="text-center py-16 border-t border-gray-100"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Join the movement</h2>
          <p className="text-gray-500 mb-8 max-w-md mx-auto">
            Whether you want to use it, contribute to it, or just follow along — we&apos;d love to have you.
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <a
              href="https://github.com/Annaa74/GestureWave-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-7 py-3.5 bg-black text-white font-bold rounded-full hover:scale-105 transition-all shadow-xl shadow-black/10"
            >
              <Github className="w-4 h-4" />
              Star on GitHub
            </a>
            <a
              href="/setup"
              className="flex items-center gap-2 px-7 py-3.5 bg-white border border-gray-200 text-gray-700 font-semibold rounded-full hover:bg-gray-50 transition-all"
            >
              Get Started <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
