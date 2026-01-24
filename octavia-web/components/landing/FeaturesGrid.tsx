"use client";

import { motion } from "framer-motion";
import { Video, Mic, Languages, Wand2, Zap, Globe } from "lucide-react";

const features = [
    {
        icon: Video,
        title: "Video Translation",
        description:
            "Translate videos with AI-powered lip synchronization. The AI adjusts mouth movements to match the new language perfectly.",
        color: "text-primary-purple-bright",
    },
    {
        icon: Mic,
        title: "Voice Cloning",
        description:
            "Keep the original speaker's voice and emotion. Our engine clones the unique vocal characteristics for a natural result.",
        color: "text-accent-cyan",
    },
    {
        icon: Languages,
        title: "Smart Subtitles",
        description:
            "Generate frame-perfect subtitles in SRT, VTT, or burned-in formats. Edit them in real-time with our advanced editor.",
        color: "text-accent-pink",
    },
    {
        icon: Wand2,
        title: "Magic Mode",
        description:
            "One-click enhancement. Automatically remove background noise, enhance speech clarity, and balance audio levels.",
        color: "text-yellow-400",
    },
    {
        icon: Zap,
        title: "Lightning Fast",
        description:
            "Process hours of content in minutes. Our distributed GPU network ensures you never wait for a translation.",
        color: "text-blue-400",
    },
    {
        icon: Globe,
        title: "Global Reach",
        description:
            "Support for 30+ languages including Japanese, Mandarin, Spanish, French, German, and more.",
        color: "text-accent-cyan",
    },
];

export function FeaturesGrid() {
    return (
        <section id="features" className="py-24 relative">
            <div className="container mx-auto px-6">
                <div className="text-center mb-16">
                    <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
                        Everything you need to <br />
                        <span className="text-gradient-purple">go global.</span>
                    </h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Octavia combines cutting-edge AI models to deliver the most realistic
                        dubbing and translation experience available.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {features.map((feature, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            className="glass-panel-glow hover-lift interactive group relative overflow-hidden rounded-2xl p-8"
                        >
                            <div className="glass-shine"></div>

                            {/* Localized Glow on Hover */}
                            <div className={`absolute -top-1/2 left-1/2 -translate-x-1/2 w-64 h-64 opacity-0 group-hover:opacity-40 transition-opacity duration-500 blur-[80px] rounded-full ${feature.color.includes("cyan") ? "bg-accent-cyan" :
                                feature.color.includes("pink") ? "bg-accent-pink" :
                                    "bg-primary-purple"
                                }`} />

                            <div className="relative z-10">
                                <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-white/10 transition-colors border border-white/10 group-hover:border-white/20">
                                    <feature.icon className={`w-6 h-6 ${feature.color}`} />
                                </div>
                                <h3 className="text-xl font-bold text-white mb-3 text-glow-purple">
                                    {feature.title}
                                </h3>
                                <p className="text-gray-400 leading-relaxed text-sm">
                                    {feature.description}
                                </p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
