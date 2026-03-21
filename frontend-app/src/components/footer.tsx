import Link from "next/link";
import { Waves, Twitter, Github, Linkedin, MapPin, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-24">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="col-span-1 md:col-span-1">
            <Link href="/" className="flex items-center gap-2 font-bold text-xl tracking-tighter text-gray-900 mb-6">
              <div className="w-8 h-8 rounded-lg bg-black flex items-center justify-center shadow-md shadow-blue-900/10">
                <Waves className="w-5 h-5 text-white" />
              </div>
              GestureWave<span className="text-gray-400">AI</span>
            </Link>
            <p className="text-sm text-gray-500 leading-relaxed mb-6">
              Pioneering the future of spatial computing and contactless interaction. Built with passion for a more accessible web.
            </p>
            <div className="flex gap-4">
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-blue-500 transition-colors">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="https://github.com/Annaa74/GestureWave-AI" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-black transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-blue-700 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Platform</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="/features" className="hover:text-black transition-colors">Features</Link></li>
              <li><Link href="/setup" className="hover:text-black transition-colors">Setup Guide</Link></li>
              <li><Link href="/docs" className="hover:text-black transition-colors">Documentation</Link></li>
              <li><Link href="/community" className="hover:text-black transition-colors">Community Hub</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Explore</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="/about" className="hover:text-black transition-colors">Our Story</Link></li>
              <li><Link href="/community" className="hover:text-black transition-colors flex items-center gap-2">Showcase <span className="bg-blue-100 text-blue-700 font-medium px-2 py-0.5 rounded-full text-xs animate-pulse">Waitlist</span></Link></li>
              <li><Link href="https://github.com/Annaa74/GestureWave-AI/issues" className="hover:text-black transition-colors">Report Issues</Link></li>
              <li><Link href="/contact" className="hover:text-black transition-colors">Contact Support</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Legal</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="/privacy" className="hover:text-black transition-colors">Privacy Policy</Link></li>
              <li><Link href="/terms" className="hover:text-black transition-colors">Terms of Service</Link></li>
              <li><Link href="/security" className="hover:text-black transition-colors">Security</Link></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-100 mt-16 pt-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col items-center md:items-start gap-1">
            <p className="text-sm text-gray-500">
              © {new Date().getFullYear()} GestureWave AI. All rights reserved.
            </p>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <MapPin className="w-3 h-3 text-red-400" />
              Built with <Heart className="w-3 h-3 text-red-500 inline fill-red-500" /> in India
            </div>
          </div>
          <div className="flex items-center gap-6 text-xs font-medium text-gray-400">
            <Link href="/terms" className="hover:text-gray-600">Terms</Link>
            <Link href="/privacy" className="hover:text-gray-600">Privacy</Link>
            <Link href="/security" className="hover:text-gray-600">Security</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
