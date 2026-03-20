"use client";

import { motion } from "framer-motion";
import { BookOpen, Camera, Hand, AlertTriangle, Lightbulb, CheckCircle2, Search, Zap, Code } from "lucide-react";

export default function DocsPage() {
  const sections = [
    {
      id: "installation",
      icon: <Zap className="w-5 h-5 text-blue-600" />,
      title: "Quick Start & Installation",
      content: (
        <div className="space-y-4 text-gray-600 text-sm leading-relaxed">
          <p>
            The easiest way to get started is by downloading the <strong>Windows Installer (.exe)</strong> from the Community page or Setup Guide. It bundles everything you need—no coding required.
          </p>
          <div className="bg-gray-50 p-4 rounded-xl border border-gray-100 font-mono text-xs">
            1. Download GestureWaveAI_Installer.exe
            <br />
            2. Run the installer and agree to terms
            <br />
            3. Launch "GestureWave AI" from your start menu or desktop shortcut!
          </div>
          <p>
            For advanced Python users running from source, please clone the GitHub repository and run <code className="bg-gray-100 px-1 py-0.5 rounded text-red-500">pip install -r requirements.txt</code>. Then execute <code className="bg-gray-100 px-1 py-0.5 rounded text-red-500">python app.py</code> in your terminal.
          </p>
        </div>
      )
    },
    {
      id: "custom-gestures",
      icon: <Hand className="w-5 h-5 text-indigo-600" />,
      title: "Recording Custom Gestures",
      content: (
        <div className="space-y-4 text-gray-600 text-sm leading-relaxed">
          <p>
            GestureWave AI allows you to record custom hand poses which bind to system actions. Currently, <strong>Gesture_1</strong> is mapped to instantly open your default web browser to LinkedIn.
          </p>
          <ol className="list-decimal list-outside ml-4 space-y-2">
            <li>Click <strong>Start Tracking</strong> in the main launcher tab.</li>
            <li>Hold your hand up to the camera in the new pose you want to store (e.g., three fingers up).</li>
            <li>While holding still, press the <kbd className="font-mono bg-white border border-gray-200 px-1.5 py-0.5 rounded shadow-sm">r</kbd> key on your keyboard.</li>
            <li>Wait for the 3-second countdown on screen to complete.</li>
            <li>Your pose is now saved. Next time you make that exact pose, LinkedIn will launch!</li>
          </ol>
        </div>
      )
    },
    {
      id: "troubleshoot-camera",
      icon: <Camera className="w-5 h-5 text-amber-600" />,
      title: "Troubleshooting: Webcam Not Found",
      content: (
        <div className="space-y-4 text-gray-600 text-sm leading-relaxed">
          <p>
            If the application starts but the camera preview window does not open, or if the status says "No Hand Detected" even when your hand is up:
          </p>
          <ul className="list-disc list-outside ml-4 space-y-2">
            <li>Ensure no other applications (like Zoom or Teams) are currently locking your webcam.</li>
            <li>Navigate to the <strong>Settings</strong> tab inside the Launcher.</li>
            <li>Change the <strong>Camera Index</strong> from <code className="bg-gray-100 px-1 py-0.5 rounded">0</code> to <code className="bg-gray-100 px-1 py-0.5 rounded">1</code> or <code className="bg-gray-100 px-1 py-0.5 rounded">2</code> based on your USB configuration.</li>
            <li>Click <strong>Apply Settings</strong> and restart the tracking.</li>
          </ul>
        </div>
      )
    },
    {
      id: "troubleshoot-crash",
      icon: <AlertTriangle className="w-5 h-5 text-red-600" />,
      title: "Troubleshooting: App Immediate Crash",
      content: (
        <div className="space-y-4 text-gray-600 text-sm leading-relaxed">
          <p>
            If you are running the project from source and receive an <code className="bg-gray-100 px-1 py-0.5 rounded text-red-500">ImportError: numpy.core.multiarray</code>, this happens if your system's NumPy version is incompatible with MediaPipe's compilation.
          </p>
          <div className="bg-red-50 p-4 rounded-xl border border-red-100">
            <div className="flex items-center gap-2 mb-2 font-bold text-red-800">
              <CheckCircle2 className="w-4 h-4" /> Solution
            </div>
            <p className="text-red-700">
              We highly recommend strictly pinning your NumPy. Execute this command in your terminal manually: <br/><br/>
              <code className="bg-white px-2 py-1 rounded shadow-sm text-red-600">pip install "numpy&lt;2"</code><br/><br/>
              This will safely downgrade your numpy installation below v2.0 to resolve the architecture conflict.
            </p>
          </div>
        </div>
      )
    }
  ];

  return (
    <div className="bg-[#fafafa] min-h-screen">
      {/* Header */}
      <div className="bg-white border-b border-gray-100 pt-32 pb-16">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium mb-6 shadow-sm"
          >
            <BookOpen className="w-4 h-4" /> Official Documentation
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className="text-4xl md:text-5xl font-extrabold tracking-tight text-gray-900 mb-6"
          >
            How can we help you today?
          </motion.h1>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="relative max-w-2xl mx-auto"
          >
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input 
              type="text" 
              placeholder="Search for installation, troubleshooting, gestures..." 
              className="w-full pl-12 pr-4 py-4 rounded-2xl border border-gray-200 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-lg shadow-gray-200/20 text-gray-800"
            />
          </motion.div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-6 py-20 flex flex-col md:flex-row gap-12 items-start">
        
        {/* Sidebar Nav */}
        <div className="md:w-64 shrink-0 top-32 sticky hidden md:block">
          <h3 className="font-bold text-gray-900 mb-4 uppercase text-xs tracking-wider">On this page</h3>
          <ul className="space-y-3 border-l-2 border-gray-100">
            {sections.map(section => (
              <li key={section.id}>
                <a href={`#${section.id}`} className="block pl-4 text-sm text-gray-500 hover:text-blue-600 hover:border-l-2 hover:-ml-[2px] hover:border-blue-600 transition-all">
                  {section.title}
                </a>
              </li>
            ))}
          </ul>
        </div>

        {/* Content Body */}
        <div className="flex-1 space-y-12">
          {sections.map((section, idx) => (
            <motion.div 
              id={section.id} 
              key={section.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + (idx * 0.1), duration: 0.5 }}
              className="scroll-mt-32 p-8 bg-white border border-gray-100 rounded-3xl shadow-sm"
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gray-50 flex items-center justify-center border border-gray-100 shadow-sm">
                  {section.icon}
                </div>
                <h2 className="text-xl font-bold text-gray-900">{section.title}</h2>
              </div>
              <div className="w-full h-px bg-gray-50 mb-6" />
              {section.content}
            </motion.div>
          ))}
          
          <div className="mt-16 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-3xl p-8 text-white shadow-xl shadow-blue-900/10 flex flex-col md:flex-row items-center justify-between gap-6">
            <div>
              <h3 className="text-xl font-bold mb-2 flex items-center gap-2"><Lightbulb className="w-6 h-6"/> Still need help?</h3>
              <p className="text-blue-100 text-sm max-w-md">Our community forum and developers are highly active and waiting to help you optimize your gesture engine setup.</p>
            </div>
            <a href="/community" className="px-6 py-3 bg-white text-blue-700 font-bold text-sm rounded-full hover:scale-105 active:scale-95 transition-all shadow-md shrink-0">
              Visit Community Forum
            </a>
          </div>
        </div>
        
      </div>
    </div>
  );
}
