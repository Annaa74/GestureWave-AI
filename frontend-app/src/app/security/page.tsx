"use client";

import { motion } from "framer-motion";
import { ShieldCheck, Lock, Activity, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function SecurityPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen">
      <div className="max-w-4xl mx-auto px-6 py-20">
        <Link href="/" className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-black mb-12 transition-colors">
          <ArrowLeft className="w-4 h-4" /> Back to Home
        </Link>
        <motion.div
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           className="mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight">Security Standards</h1>
          <p className="text-lg text-gray-500 leading-relaxed max-w-2xl">
            GestureWave AI is built with transparency and safe local computing in mind. Our code is 100% open-source for full auditability.
          </p>
        </motion.div>

        <div className="space-y-8 mb-16">
          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Lock className="w-5 h-5 text-indigo-600" />
              Local-Only Environment
            </h3>
            <p className="text-gray-500 leading-relaxed">
              GestureWave AI requires no network access to function. Our executable does not attempt to connect to the internet, ensuring that your system activity and camera feed are always private and protected from interception.
            </p>
          </div>

          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-600" />
              Sandboxed Processing
            </h3>
            <p className="text-gray-500 leading-relaxed">
              The hand tracking engine runs in a separate process and only communicates with your system's cursor controller through standard system calls. We do not modify system registries or core drivers.
            </p>
          </div>

          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-600" />
              Resource Optimization
            </h3>
            <p className="text-gray-500 leading-relaxed">
              We've optimized our algorithm to maintain a high level of security without overwhelming your CPU. This prevents system freezes during intensive usage.
            </p>
          </div>
        </div>

        <section className="p-8 bg-gray-900 rounded-3xl text-white">
          <h4 className="text-xl font-bold mb-2">Vulnerability Reporting</h4>
          <p className="text-gray-400 text-sm leading-relaxed mb-6">
            If you find any security issues in our source code, please report them directly via our GitHub issues or reach out via our community forum.
          </p>
          <a href="https://github.com/Annaa74/GestureWave-AI" target="_blank" rel="noopener noreferrer">
            <button className="px-6 py-3 bg-white text-black font-bold text-sm rounded-full hover:scale-105 transition-all">
              Audit the Code
            </button>
          </a>
        </section>
      </div>
    </div>
  );
}
