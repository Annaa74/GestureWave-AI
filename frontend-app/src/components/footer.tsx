import Link from "next/link";
import { Waves, Twitter, Github, Linkedin, Briefcase } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-24">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12">
          {/* Brand */}
          <div className="col-span-1 md:col-span-1">
            <Link href="/" className="flex items-center gap-2 font-bold text-xl tracking-tighter text-gray-900 mb-6">
              <div className="w-8 h-8 rounded-lg bg-black flex items-center justify-center shadow-md">
                <Waves className="w-5 h-5 text-white" />
              </div>
              GestureWave<span className="text-gray-400">AI</span>
            </Link>
            <p className="text-sm text-gray-500 leading-relaxed mb-6">
              Pioneering the future of spatial computing and contactless interaction out of our headquarters in San Francisco, CA.
            </p>
            <div className="flex gap-4">
              <a href="https://twitter.com" className="text-gray-400 hover:text-blue-500 transition-colors">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="https://github.com" className="text-gray-400 hover:text-black transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://linkedin.com" className="text-gray-400 hover:text-blue-700 transition-colors">
                <Linkedin className="w-5 h-5" />
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Platform</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="/features" className="hover:text-black transition-colors">Features</Link></li>
              <li><Link href="/setup" className="hover:text-black transition-colors">Documentation</Link></li>
              <li><Link href="/setup" className="hover:text-black transition-colors">Setup Guide</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Changelog</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Company</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="/about" className="hover:text-black transition-colors">About Us</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors flex items-center gap-2">Careers <span className="bg-blue-100 text-blue-700 font-medium px-2 py-0.5 rounded-full text-xs">Hiring</span></Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Press</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Contact</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-gray-900 mb-6">Legal</h3>
            <ul className="flex flex-col gap-4 text-sm text-gray-500">
              <li><Link href="#" className="hover:text-black transition-colors">Privacy Policy</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Terms of Service</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Cookie Policy</Link></li>
              <li><Link href="#" className="hover:text-black transition-colors">Security</Link></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-100 mt-16 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-gray-500">
            © {new Date().getFullYear()} GestureWave AI, Inc. All rights reserved.
          </p>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Briefcase className="w-4 h-4" />
            Designed in California
          </div>
        </div>
      </div>
    </footer>
  );
}
