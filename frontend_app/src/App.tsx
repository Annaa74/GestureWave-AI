import React, { useEffect } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { Download, MonitorPlay, MousePointer2, Settings, Zap, Shield, ArrowRight, Github, Twitter, Linkedin } from 'lucide-react';
import heroImage from '../../frontend/assets/hero.png';

const fadeIn = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } }
};

const staggerContainer = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15 } }
};

function App() {
    const { scrollY } = useScroll();
    const y1 = useTransform(scrollY, [0, 1000], [0, 200]);
    const y2 = useTransform(scrollY, [0, 1000], [0, -100]);

    // Smooth scroll progress
    const scaleX = useSpring(useTransform(scrollY, [0, 3000], [0, 1]), {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    return (
        <div className="relative overflow-hidden bg-background">
            {/* Scroll Progress Bar */}
            <motion.div
                className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary to-secondary z-[1001] origin-left"
                style={{ scaleX }}
            />

            {/* Background Animated Orbs */}
            <div className="bg-orb w-[600px] h-[600px] bg-[radial-gradient(circle,rgba(37,99,235,0.15)_0%,transparent_70%)] top-[-100px] right-[-100px]" />
            <div className="bg-orb w-[500px] h-[500px] bg-[radial-gradient(circle,rgba(139,92,246,0.15)_0%,transparent_70%)] bottom-[20%] left-[-100px] [animation-delay:-5s]" />

            {/* Navigation */}
            <motion.nav
                initial={{ y: -100 }}
                animate={{ y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="fixed top-0 w-full z-[1000] transition-all duration-300 gloss-effect bg-white/70 backdrop-blur-xl border-b border-white/50"
            >
                <div className="max-w-7xl mx-auto px-6 lg:px-8 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3 font-display font-extrabold text-2xl tracking-tighter text-text-main">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary shadow-[inset_0_2px_0_rgba(255,255,255,0.4)]" />
                        <span>GestureWave <span className="text-primary">AI</span></span>
                    </div>
                    <div className="hidden md:flex gap-10 font-medium text-text-muted">
                        <a href="#overview" className="hover:text-text-main transition-colors">Overview</a>
                        <a href="#features" className="hover:text-text-main transition-colors">Features</a>
                        <a href="#technology" className="hover:text-text-main transition-colors">Technology</a>
                    </div>
                    <div className="flex gap-4">
                        <a href="#docs" className="btn btn-secondary hidden sm:inline-flex">Documentation</a>
                        <a href="https://github.com/Annaa74/GestureWave-AI" className="btn btn-primary">
                            <Download size={18} /> Download
                        </a>
                    </div>
                </div>
            </motion.nav>

            {/* Hero Section */}
            <main className="pt-32 pb-16 lg:pt-48 lg:pb-32 px-6 lg:px-8 max-w-7xl mx-auto min-h-screen flex flex-col justify-center">
                <div className="grid lg:grid-cols-2 gap-16 items-center relative z-10">
                    <motion.div
                        initial="hidden"
                        animate="visible"
                        variants={staggerContainer}
                        className="max-w-2xl"
                    >
                        <motion.div variants={fadeIn} className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-black/5 rounded-full text-sm font-semibold text-text-muted mb-8 shadow-sm">
                            <span className="w-2 h-2 rounded-full bg-primary shadow-[0_0_10px_rgba(37,99,235,0.5)] animate-pulse" />
                            v1.0 Windows Release Available
                        </motion.div>
                        <motion.h1 variants={fadeIn} className="text-5xl lg:text-7xl font-extrabold leading-[1.05] tracking-tight mb-6">
                            Master Your <br />
                            <span className="gradient-text pb-2">Digital Workspace</span>
                        </motion.h1>
                        <motion.p variants={fadeIn} className="text-lg text-text-muted mb-10 max-w-lg leading-relaxed">
                            Experience frictionless computing. Control your OS with enterprise-grade hand tracking precision. No specialized hardware needed.
                        </motion.p>
                        <motion.div variants={fadeIn} className="flex flex-wrap items-center gap-4">
                            <a href="https://github.com/Annaa74/GestureWave-AI" className="btn btn-primary px-8 py-4 text-lg rounded-2xl">
                                Get Started <ArrowRight size={20} />
                            </a>
                            <a href="#demo" className="btn btn-secondary px-8 py-4 text-lg rounded-2xl bg-white">
                                <MonitorPlay size={20} className="text-text-muted" /> Watch Demo
                            </a>
                        </motion.div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, rotateY: 15 }}
                        animate={{ opacity: 1, scale: 1, rotateY: 0 }}
                        transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
                        className="relative lg:h-[600px] flex items-center justify-center [perspective:1000px]"
                    >
                        <motion.div
                            style={{ y: y1 }}
                            whileHover={{ rotateY: 0, rotateX: 0, scale: 1.02 }}
                            className="gloss-panel w-full max-w-md transform -rotate-y-12 rotate-x-6 transition-transform duration-500 bg-white"
                        >
                            <div className="border-b border-black/5 p-4 flex gap-2">
                                <div className="w-3 h-3 rounded-full bg-[#ff5f56]" />
                                <div className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
                                <div className="w-3 h-3 rounded-full bg-[#27c93f]" />
                            </div>
                            <img src={heroImage} alt="AI Hand Tracking Visualization" className="w-full h-auto rounded-b-3xl" />
                        </motion.div>

                        {/* Parallax Floating Widgets */}
                        <motion.div
                            style={{ y: y2 }}
                            className="absolute -right-6 -top-6 gloss-panel bg-white/90 backdrop-blur-xl p-4 md:p-6 rounded-2xl flex items-center gap-4 shadow-2xl z-20"
                        >
                            <div className="w-12 h-12 rounded-xl bg-background flex items-center justify-center text-primary">
                                <Zap size={24} />
                            </div>
                            <div>
                                <h4 className="font-bold text-sm">Zero Latency</h4>
                                <p className="text-xs text-text-muted">{'<'} 15ms Response</p>
                            </div>
                        </motion.div>

                        <motion.div
                            animate={{ y: [0, -15, 0] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                            className="absolute -left-6 bottom-12 gloss-panel bg-white/90 backdrop-blur-xl p-4 md:p-6 rounded-2xl flex items-center gap-4 shadow-2xl z-20"
                        >
                            <div className="w-12 h-12 rounded-xl bg-background flex items-center justify-center text-secondary">
                                <Settings size={24} />
                            </div>
                            <div>
                                <h4 className="font-bold text-sm">Pixel Perfect</h4>
                                <p className="text-xs text-text-muted">100% Tracking</p>
                            </div>
                        </motion.div>
                    </motion.div>
                </div>
            </main>

            {/* Features Section */}
            <section id="features" className="py-24 bg-surface relative z-10 border-y border-black/5">
                <div className="max-w-7xl mx-auto px-6 lg:px-8">
                    <motion.div
                        initial="hidden"
                        whileInView="visible"
                        viewport={{ once: true, margin: "-100px" }}
                        variants={fadeIn}
                        className="text-center max-w-2xl mx-auto mb-20"
                    >
                        <h2 className="text-4xl font-bold mb-4">Designed for Professionals</h2>
                        <p className="text-lg text-text-muted">Every gesture is calibrated for maximum ergonomic efficiency and seamless integration into your daily workflow.</p>
                    </motion.div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {[
                            {
                                icon: <MousePointer2 size={24} />,
                                title: 'Fluid Navigation',
                                desc: 'Drive your cursor directly with index finger tracking. Algorithms smooth out micro-tremors for pristine cursor placement.'
                            },
                            {
                                icon: <Zap size={24} />,
                                title: 'Contextual Clicks',
                                desc: 'Perform left and right clicks with natural thumb-to-finger pinches. Haptic-like visual feedback confirms every action.'
                            },
                            {
                                icon: <Settings size={24} />,
                                title: 'Intelligent Scroll',
                                desc: 'Bind fingers together to instantly activate the scroll plane. Read documents and scrub timelines entirely hands-free.'
                            }
                        ].map((feature, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true, margin: "-50px" }}
                                transition={{ duration: 0.6, delay: i * 0.15 }}
                                whileHover={{ y: -8 }}
                                className="gloss-panel p-10 bg-white"
                            >
                                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary to-secondary text-white flex items-center justify-center mb-6 shadow-[inset_0_2px_0_rgba(255,255,255,0.3),0_8px_20px_rgba(37,99,235,0.25)]">
                                    {feature.icon}
                                </div>
                                <h3 className="text-2xl font-bold mb-3">{feature.title}</h3>
                                <p className="text-text-muted leading-relaxed">{feature.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Technology Section */}
            <section id="technology" className="py-32 relative z-10 pb-48">
                <div className="max-w-5xl mx-auto px-6 lg:px-8">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.8 }}
                        className="gloss-panel bg-gradient-to-br from-white to-gray-50 p-12 lg:p-16 flex flex-col md:flex-row items-center gap-12 lg:gap-20"
                    >
                        <div className="flex-1">
                            <h2 className="text-4xl font-bold mb-6">Powered by Embedded AI</h2>
                            <p className="text-lg text-text-muted mb-8 leading-relaxed">
                                GestureWave runs on a locally compiled, highly optimized lightweight neural network. Absolutely zero data is sent to the cloud. Your webcam feed is processed directly in memory.
                            </p>
                            <ul className="space-y-4">
                                {[
                                    '100% Offline Processing',
                                    'Minimal CPU Overhead',
                                    'Complete Privacy Guarantee'
                                ].map((item, i) => (
                                    <li key={i} className="flex items-center gap-3 font-medium">
                                        <span className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-sm">
                                            <Shield size={14} />
                                        </span>
                                        {item}
                                    </li>
                                ))}
                            </ul>
                        </div>
                        <div className="w-full md:w-1/2 flex justify-center">
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                                className="w-64 h-64 lg:w-80 lg:h-80 rounded-full bg-gradient-to-br from-primary to-secondary shadow-[inset_-20px_-20px_60px_rgba(0,0,0,0.5),inset_10px_10px_40px_rgba(255,255,255,0.8),0_20px_50px_rgba(37,99,235,0.4)]"
                            />
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Professional Footer */}
            <footer className="bg-white border-t border-black/5 pt-24 pb-12 px-6 lg:px-8 relative z-20">
                <div className="max-w-7xl mx-auto">
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-12 mb-16">
                        <div className="col-span-2 lg:col-span-2">
                            <div className="flex items-center gap-2 font-display font-extrabold text-2xl tracking-tighter text-text-main mb-6">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-secondary shadow-[inset_0_2px_0_rgba(255,255,255,0.4)]" />
                                <span>GestureWave <span className="text-primary">AI</span></span>
                            </div>
                            <p className="text-text-muted pr-12 max-w-sm mb-8">
                                Pioneering the next generation of human-computer interaction through advanced locally-run machine learning.
                            </p>
                        </div>

                        <div>
                            <h4 className="font-bold mb-6 text-text-main">Product</h4>
                            <ul className="space-y-4 text-text-muted">
                                <li><a href="#" className="hover:text-primary transition-colors">Download Window UI</a></li>
                                <li><a href="#" className="hover:text-primary transition-colors">Release Notes</a></li>
                                <li><a href="#" className="hover:text-primary transition-colors">Security</a></li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="font-bold mb-6 text-text-main">Resources</h4>
                            <ul className="space-y-4 text-text-muted">
                                <li><a href="#" className="hover:text-primary transition-colors">Documentation</a></li>
                                <li><a href="#" className="hover:text-primary transition-colors">API Reference</a></li>
                                <li><a href="https://github.com/Annaa74/GestureWave-AI" className="hover:text-primary transition-colors">Open Source</a></li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="font-bold mb-6 text-text-main">Company</h4>
                            <ul className="space-y-4 text-text-muted">
                                <li><a href="#" className="hover:text-primary transition-colors">About</a></li>
                                <li><a href="#" className="hover:text-primary transition-colors">Privacy Policy</a></li>
                                <li><a href="#" className="hover:text-primary transition-colors">Terms of Service</a></li>
                            </ul>
                        </div>
                    </div>

                    <div className="pt-8 border-t border-black/5 flex flex-col md:flex-row justify-between items-center gap-4 text-text-muted text-sm">
                        <p>© 2026 Annaa74 Technologies. All rights reserved.</p>
                        <div className="flex gap-6">
                            <a href="#" className="hover:text-primary transition-colors"><Github size={20} /></a>
                            <a href="#" className="hover:text-primary transition-colors"><Twitter size={20} /></a>
                            <a href="#" className="hover:text-primary transition-colors"><Linkedin size={20} /></a>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default App;
