"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import {
  Mail, Bell, Users, Rss, Star, GitBranch,
  CheckCircle2, Zap, Shield, Package, ArrowRight,
  MessageSquare, Sparkles, ChevronRight
} from "lucide-react";

// ── Changelog data ────────────────────────────────────────────────────────────
const changelog = [
  {
    version: "v2.1",
    date: "Mar 2026",
    tag: "Latest",
    tagColor: "bg-green-100 text-green-700 border-green-200",
    entries: [
      { type: "feat", text: "Premium dark-mode desktop GUI with tabbed settings panel" },
      { type: "feat", text: "Live gesture log and real-time session timer in launcher" },
      { type: "feat", text: "Per-setting sliders: smoothing, dead zone, click threshold, scroll speed" },
      { type: "fix",  text: "Camera index selection (supports multi-webcam setups)" },
    ],
  },
  {
    version: "v2.0",
    date: "Mar 2026",
    tag: "Stable",
    tagColor: "bg-blue-100 text-blue-700 border-blue-200",
    entries: [
      { type: "feat", text: "EMA cursor smoothing + velocity-adaptive dampening" },
      { type: "feat", text: "Dead zone to eliminate tremor and micro-jitter" },
      { type: "feat", text: "New: drag & drop, double click, peace-sign scroll, zoom, pause/resume" },
      { type: "feat", text: "Full gesture state machine with debouncing and per-gesture cooldowns" },
      { type: "feat", text: "On-screen HUD with FPS counter, gesture name, and state indicator" },
      { type: "improve", text: "Camera set to 1280×720 @ 30fps for better tracking accuracy" },
    ],
  },
  {
    version: "v1.0",
    date: "Early 2024",
    tag: "Initial",
    tagColor: "bg-gray-100 text-gray-600 border-gray-200",
    entries: [
      { type: "feat", text: "Initial release: index-finger cursor control via MediaPipe" },
      { type: "feat", text: "Index+thumb pinch → left click, middle+thumb → right click" },
      { type: "feat", text: "Two-finger proximity scroll" },
    ],
  },
];

// ── Roadmap ───────────────────────────────────────────────────────────────────
const roadmap = [
  { icon: <Package className="w-5 h-5" />, title: "Windows Installer (.exe)", status: "In Progress", color: "amber" },
  { icon: <Sparkles className="w-5 h-5" />, title: "Custom Gesture Macros", status: "Planned", color: "blue" },
  { icon: <Shield className="w-5 h-5" />, title: "Multi-hand Support", status: "Planned", color: "blue" },
  { icon: <Zap className="w-5 h-5" />, title: "Mobile Companion App (Android)", status: "Future", color: "violet" },
  { icon: <Zap className="w-5 h-5" />, title: "Tablet Support (iPad / Android)", status: "Future", color: "violet" },
  { icon: <MessageSquare className="w-5 h-5" />, title: "Voice + Gesture Combo Mode", status: "Concept", color: "gray" },
];

const typeColor: Record<string, string> = {
  feat:    "bg-blue-100 text-blue-700",
  fix:     "bg-red-100 text-red-700",
  improve: "bg-amber-100 text-amber-700",
};
const typeLabel: Record<string, string> = {
  feat: "feat", fix: "fix", improve: "improv",
};

const statusConfig: Record<string, { dot: string; badge: string }> = {
  "In Progress": { dot: "bg-amber-400",  badge: "bg-amber-50 text-amber-700 border-amber-200" },
  "Planned":     { dot: "bg-blue-400",   badge: "bg-blue-50 text-blue-700 border-blue-200" },
  "Future":      { dot: "bg-violet-400", badge: "bg-violet-50 text-violet-700 border-violet-200" },
  "Concept":     { dot: "bg-gray-400",   badge: "bg-gray-100 text-gray-600 border-gray-200" },
};

// ── Newsletter form ───────────────────────────────────────────────────────────
function NewsletterForm() {
  const [email, setEmail]       = useState("");
  const [name, setName]         = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.includes("@")) {
      setError("Please enter a valid email address.");
      return;
    }
    setLoading(true);
    setError("");

    // ── Integration point ──────────────────────────────────────────────────
    // Replace this with your preferred email service:
    //   • Formspree:    POST to https://formspree.io/f/YOUR_ID
    //   • EmailJS:      emailjs.send(...)
    //   • Resend:       POST to /api/subscribe
    // ──────────────────────────────────────────────────────────────────────
    try {
      await new Promise(r => setTimeout(r, 1200)); // simulated delay
      setSubmitted(true);
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="text-center py-10"
      >
        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 className="w-8 h-8 text-green-600" />
        </div>
        <h3 className="text-2xl font-bold mb-2">You&apos;re on the list!</h3>
        <p className="text-gray-500">
          We&apos;ll send updates about new releases, features, and community news to<br />
          <span className="font-semibold text-gray-700">{email}</span>
        </p>
      </motion.div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-bold text-gray-700 mb-1.5">Name (optional)</label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Your name"
            className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-gray-400"
          />
        </div>
        <div>
          <label className="block text-sm font-bold text-gray-700 mb-1.5">Email address *</label>
          <input
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
            className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all placeholder-gray-400"
          />
        </div>
      </div>

      {error && (
        <p className="text-sm text-red-600 flex items-center gap-1.5">
          <span className="w-4 h-4 rounded-full bg-red-100 flex items-center justify-center text-xs">!</span>
          {error}
        </p>
      )}

      <div className="flex items-center justify-between pt-2">
        <p className="text-xs text-gray-400">
          No spam. Unsubscribe anytime. Updates only.
        </p>
        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 px-7 py-3 bg-blue-600 text-white font-bold text-sm rounded-full hover:bg-blue-700 active:scale-95 transition-all disabled:opacity-60 disabled:cursor-wait shadow-lg shadow-blue-200"
        >
          {loading ? (
            <span className="animate-spin">⟳</span>
          ) : (
            <>
              <Mail className="w-4 h-4" />
              Subscribe
            </>
          )}
        </button>
      </div>
    </form>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function CommunityPage() {
  return (
    <div className="bg-[#fafafa] text-gray-900 min-h-screen">
      <div className="max-w-4xl mx-auto px-6 py-20">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-20"
        >
          <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-blue-700 text-sm font-medium mb-6">
            <Users className="w-4 h-4" />
            Community & Updates
          </span>
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-6 leading-tight">
            Stay in the loop.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
              Shape what&apos;s next.
            </span>
          </h1>
          <p className="text-lg text-gray-600 max-w-xl mx-auto leading-relaxed">
            Subscribe for release notes, feature previews, and community highlights — delivered straight to your inbox.
          </p>
        </motion.div>

        {/* Newsletter Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="bg-white border border-gray-100 rounded-3xl p-8 shadow-sm mb-10"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center">
              <Bell className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Get Release Updates</h2>
              <p className="text-sm text-gray-500">New versions, feature drops, community news</p>
            </div>
          </div>
          <NewsletterForm />
        </motion.div>

        {/* What you'll receive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="grid md:grid-cols-3 gap-4 mb-16"
        >
          {[
            { icon: <Rss className="w-5 h-5 text-blue-600" />, title: "Release Notes", body: "Get notified the moment a new version ships, with full changelog." },
            { icon: <Sparkles className="w-5 h-5 text-violet-600" />, title: "Feature Previews", body: "Early access to beta features before they go public." },
            { icon: <Star className="w-5 h-5 text-amber-500" />, title: "Community Picks", body: "Best use cases, demos, and setups from other GestureWave users." },
          ].map((card, i) => (
            <div key={i} className="p-5 bg-white border border-gray-100 rounded-2xl shadow-sm">
              <div className="w-10 h-10 bg-gray-50 rounded-xl flex items-center justify-center mb-3">
                {card.icon}
              </div>
              <h3 className="font-bold mb-1">{card.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">{card.body}</p>
            </div>
          ))}
        </motion.div>

        {/* Roadmap */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="mb-16"
        >
          <div className="flex items-center gap-2 mb-8">
            <GitBranch className="w-5 h-5 text-indigo-600" />
            <h2 className="text-2xl font-bold">Public Roadmap</h2>
          </div>
          <div className="space-y-3">
            {roadmap.map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -16 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 + i * 0.07 }}
                className="flex items-center justify-between p-4 bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md transition-all group"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-gray-50 rounded-xl border border-gray-100 flex items-center justify-center text-gray-500 group-hover:text-blue-600 transition-colors">
                    {item.icon}
                  </div>
                  <span className="font-semibold text-gray-900">{item.title}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-bold ${statusConfig[item.status].badge}`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${statusConfig[item.status].dot}`} />
                    {item.status}
                  </span>
                  <ChevronRight className="w-4 h-4 text-gray-300 group-hover:text-gray-500 transition-colors" />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Changelog */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="mb-16"
        >
          <div className="flex items-center gap-2 mb-8">
            <Rss className="w-5 h-5 text-blue-600" />
            <h2 className="text-2xl font-bold">Changelog</h2>
          </div>
          <div className="space-y-6">
            {changelog.map((release, i) => (
              <div key={i} className="bg-white border border-gray-100 rounded-2xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <span className="text-xl font-bold text-gray-900 font-mono">{release.version}</span>
                    <span className={`px-2.5 py-0.5 rounded-full border text-xs font-bold ${release.tagColor}`}>
                      {release.tag}
                    </span>
                  </div>
                  <span className="text-sm text-gray-400 font-mono">{release.date}</span>
                </div>
                <ul className="space-y-2">
                  {release.entries.map((entry, j) => (
                    <li key={j} className="flex items-start gap-3 text-sm">
                      <span className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-xs font-bold font-mono ${typeColor[entry.type]}`}>
                        {typeLabel[entry.type]}
                      </span>
                      <span className="text-gray-600 leading-relaxed">{entry.text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Forum / Discord teaser */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="rounded-3xl bg-gradient-to-br from-indigo-600 to-blue-700 text-white p-10 text-center"
        >
          <MessageSquare className="w-10 h-10 mx-auto mb-4 opacity-80" />
          <h2 className="text-2xl font-bold mb-3">Community Forum — Coming Soon</h2>
          <p className="text-blue-100 mb-6 max-w-md mx-auto text-sm leading-relaxed">
            A dedicated space to share your gesture setups, request features, report bugs, and connect with other GestureWave users. Subscribe above to get first access.
          </p>
          <div className="flex items-center justify-center gap-4 flex-wrap">
            <div className="flex items-center gap-2 bg-white/10 border border-white/20 rounded-full px-4 py-2 text-sm">
              <Users className="w-4 h-4" /> Community Q&A
            </div>
            <div className="flex items-center gap-2 bg-white/10 border border-white/20 rounded-full px-4 py-2 text-sm">
              <Star className="w-4 h-4" /> Showcase your setup
            </div>
            <div className="flex items-center gap-2 bg-white/10 border border-white/20 rounded-full px-4 py-2 text-sm">
              <Sparkles className="w-4 h-4" /> Vote on features
            </div>
          </div>
          <a href="/setup" className="inline-flex items-center gap-2 mt-8 px-7 py-3 bg-white text-blue-700 font-bold rounded-full hover:scale-105 transition-all text-sm">
            Get started now <ArrowRight className="w-4 h-4" />
          </a>
        </motion.div>

      </div>
    </div>
  );
}
