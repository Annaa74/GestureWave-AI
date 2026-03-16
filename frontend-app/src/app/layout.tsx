import type { Metadata } from "next";
import { Special_Elite } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";

const specialElite = Special_Elite({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-retro",
  display: "swap",
});

export const metadata: Metadata = {
  title: "GestureWave AI - Hand Tracking Cursor",
  description: "Control your PC cursor with hand gestures using MediaPipe and OpenCV.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={specialElite.variable}>
      <body
        className="antialiased bg-[#fafafa] min-h-screen flex flex-col"
        style={{ fontFamily: "var(--font-retro), 'Courier New', monospace" }}
      >
        <Navbar />
        <main className="flex-1 mt-20">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}
