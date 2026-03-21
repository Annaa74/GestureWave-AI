"use client";

import { motion } from "framer-motion";
import { Scale, Book, Shield, ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function TermsPage() {
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
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-6 text-gray-900">Terms of Service</h1>
          <p className="text-lg text-gray-500 max-w-2xl leading-relaxed">
            Please read these terms carefully before using GestureWave AI. By using the software, you agree to follow these rules.
          </p>
        </motion.div>

        <div className="prose prose-blue max-w-none text-gray-600 space-y-12">
          <div className="p-8 bg-blue-50 border border-blue-100 rounded-3xl">
            <h2 className="text-2xl font-bold text-blue-900 mb-4 flex items-center gap-3">
              <Scale className="w-6 h-6" /> Open Source License
            </h2>
            <p className="text-blue-800 leading-relaxed">
              GestureWave AI is released under the <strong>MIT License</strong>. You can use, modify, and distribute the project freely, as long as the original copyright notice and this permission notice are included in all copies or substantial portions of the Software.
            </p>
          </div>

          <section>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">1. Use of Software</h3>
            <p>
              GestureWave AI is an experimental tool for hand-gesture interaction. You are responsible for ensuring that your computer hardware (webcam, mouse) is compatible and that the use of this software does not interfere with critical system functions.
            </p>
          </section>

          <section>
            <h3 className="text-2xl font-bold text-gray-900 mb-4">2. Liability</h3>
            <p>
              The software is provided "as is", without warranty of any kind. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
            </p>
          </section>

          <section>
             <h3 className="text-2xl font-bold text-gray-900 mb-4">3. Updates & Maintenance</h3>
             <p>
               While we strive for stability, updates may change hardware requirements or gesture configurations. We are not responsible for software behavior changes resulting from library updates (MediaPipe, OpenCV, etc.).
             </p>
          </section>

          <section className="bg-white p-8 rounded-3xl border border-gray-100 shadow-sm text-sm italic">
            Last Updated: March 2026
          </section>
        </div>
      </div>
    </div>
  );
}
