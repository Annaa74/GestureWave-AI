"use client";

import { motion } from "framer-motion";
import { Waves, Sparkles, Users } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/",          label: "Platform"    },
  { href: "/features",  label: "Features"    },
  { href: "/setup",     label: "Setup Guide" },
  { href: "/community", label: "Community"   },
  { href: "/about",     label: "About"       },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/85 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-6xl mx-auto px-6 h-20 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl tracking-tighter text-gray-900">
          <div className="w-8 h-8 rounded-lg bg-black flex items-center justify-center shadow-md">
            <Waves className="w-5 h-5 text-white" />
          </div>
          GestureWave<span className="text-gray-400">AI</span>
        </Link>

        <div className="hidden md:flex items-center gap-7 font-medium text-sm text-gray-600">
          {NAV_LINKS.map(link => (
            <Link
              key={link.href}
              href={link.href}
              className={`transition-colors hover:text-black relative py-1 ${
                pathname === link.href ? "text-black" : ""
              }`}
            >
              {link.label}
              {pathname === link.href && (
                <motion.span
                  layoutId="nav-underline"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 rounded-full"
                />
              )}
              {link.label === "Community" && (
                <span className="ml-1.5 inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700 text-[10px] font-bold leading-none">
                  NEW
                </span>
              )}
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <Link href="/community">
            <button className="hidden md:flex px-4 py-2 text-sm font-semibold text-blue-600 border border-blue-200 bg-blue-50 hover:bg-blue-100 rounded-full transition-all items-center gap-1.5">
              <Users className="w-3.5 h-3.5" />
              Community
            </button>
          </Link>
          <Link href="/setup">
            <button className="px-5 py-2.5 text-sm font-semibold bg-black text-white hover:bg-gray-800 rounded-full transition-all flex items-center gap-2 shadow-lg shadow-black/10">
              <Sparkles className="w-4 h-4" />
              Download
            </button>
          </Link>
        </div>
      </div>
    </nav>
  );
}
