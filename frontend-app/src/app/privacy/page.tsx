"use client";

import { motion } from "framer-motion";
import { Shield, Lock, Eye, Server, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function PrivacyPage() {
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
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-6">Privacy Policy</h1>
          <p className="text-lg text-gray-500 leading-relaxed">
            Your privacy is our highest priority. GestureWave AI is designed from the ground up to ensure that your data stays exactly where it belongs: with you.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 mb-16">
          <div className="p-6 bg-white border border-gray-100 rounded-3xl shadow-sm">
            <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center mb-6">
              <Eye className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold mb-3">No Cloud Storage</h3>
            <p className="text-sm text-gray-500 leading-relaxed">
              We do not store, upload, or transmit any images or videos from your webcam to any servers. All processing happens locally on your machine.
            </p>
          </div>
          <div className="p-6 bg-white border border-gray-100 rounded-3xl shadow-sm">
            <div className="w-12 h-12 rounded-xl bg-green-50 flex items-center justify-center mb-6">
              <Lock className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-xl font-bold mb-3">Offline Processing</h3>
            <p className="text-sm text-gray-500 leading-relaxed">
              GestureWave AI can operate entirely without an internet connection. Your camera feed never leaves your local environment.
            </p>
          </div>
        </div>

        <div className="prose prose-blue max-w-none text-gray-600 space-y-8">
          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">1. Information We Collect</h2>
            <p>
              GestureWave AI does not collect any personal information. The application processes video frames in real-time to detect hand landmarks. These frames are discarded immediately after processing and are never saved to disk or transmitted over a network.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">2. Local Configuration</h2>
            <p>
              The application may save small configuration files (such as <code>gesture_registry.json</code>) locally on your computer to store your custom gesture preferences. This data remains on your device.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">3. Third-Party Libraries</h2>
            <p>
              We use Google MediaPipe for hand landmark detection. MediaPipe operates locally and follows strict privacy standards. No data is sent to Google through our implementation.
            </p>
          </section>

          <section className="p-8 bg-blue-600 rounded-3xl text-white">
            <h2 className="text-2xl font-bold mb-4">Security Commitment</h2>
            <p className="opacity-90 leading-relaxed">
              As an open-source project, our code is fully transparent. You can audit our source code on GitHub at any time to verify our privacy claims.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
