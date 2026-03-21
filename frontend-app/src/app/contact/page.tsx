"use client";

import { motion } from "framer-motion";
import { Mail, MessageCircle, Github, MapPin, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function ContactPage() {
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
          <h1 className="text-4xl md:text-5xl font-extrabold mb-6 tracking-tight">Contact Us</h1>
          <p className="text-lg text-gray-500 leading-relaxed max-w-2xl">
            Have questions or issues? Our community and developers are here to help. Reach out through any of the channels below.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 mb-16">
          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-md transition-all group">
            <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
              <Mail className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold mb-3">Support Email</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-6">
              For common support issues and questions, send us a direct email. We aim to respond within 48 hours.
            </p>
            <a href="mailto:support@gesturewaveai.com" className="text-blue-600 font-bold text-sm hover:underline">
              support@gesturewaveai.com
            </a>
          </div>

          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-md transition-all group">
            <div className="w-12 h-12 rounded-xl bg-indigo-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
              <MessageCircle className="w-6 h-6 text-indigo-600" />
            </div>
            <h3 className="text-xl font-bold mb-3">Community Hub</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-6">
              Join the conversation, find tutorials, and discuss new feature ideas with fellow users.
            </p>
            <Link href="/community" className="text-indigo-600 font-bold text-sm hover:underline">
              Visit Community
            </Link>
          </div>

          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-md transition-all group">
            <div className="w-12 h-12 rounded-xl bg-gray-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
              <Github className="w-6 h-6 text-gray-900" />
            </div>
            <h3 className="text-xl font-bold mb-3">GitHub Issues</h3>
            <p className="text-sm text-gray-500 leading-relaxed mb-6">
              Find a bug? Open an issue on our official repository to inform the core developers.
            </p>
            <a href="https://github.com/Annaa74/GestureWave-AI/issues" target="_blank" rel="noopener noreferrer" className="text-gray-900 font-bold text-sm hover:underline">
              Open an Issue
            </a>
          </div>

          <div className="p-8 bg-white border border-gray-100 rounded-3xl shadow-sm hover:shadow-md transition-all group">
            <div className="w-12 h-12 rounded-xl bg-green-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform shadow-sm">
              <MapPin className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-xl font-bold mb-3">Base of Operations</h3>
            <p className="text-sm text-gray-400 mb-1 font-mono uppercase text-xs">Origin</p>
            <p className="text-sm text-gray-900 font-bold">Built with Heart in India</p>
          </div>
        </div>

        <div className="bg-gray-50 p-10 rounded-3xl border border-gray-100 text-center">
          <h2 className="text-2xl font-bold mb-4">Want to Contribute?</h2>
          <p className="text-gray-500 mb-8 max-w-sm mx-auto text-sm leading-relaxed">
            GestureWave AI is a community-driven project. We're always looking for contributors to help with code, documentation, or design.
          </p>
          <a href="https://github.com/Annaa74/GestureWave-AI" target="_blank" rel="noopener noreferrer">
            <button className="px-8 py-3 bg-black text-white font-bold rounded-full text-sm hover:scale-105 transition-all">
              Join the Developers
            </button>
          </a>
        </div>
      </div>
    </div>
  );
}
