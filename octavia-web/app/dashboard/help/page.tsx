"use client";

import { motion } from "framer-motion";
import { HelpCircle, Video, FileText, Book, MessageCircle, ExternalLink } from "lucide-react";

const helpTopics = [
    {
        title: "Getting Started",
        description: "Learn the basics of using Octavia for video and audio translation",
        icon: Book,
        articles: ["First translation", "Understanding Magic Mode", "Voice cloning basics"]
    },
    {
        title: "Video Translation",
        description: "Master video dubbing with voice cloning and lip-sync",
        icon: Video,
        articles: ["Upload video files", "Select voices", "Review translations", "Export dubbed videos"]
    },
    {
        title: "Subtitle Tools",
        description: "Generate, translate, and convert subtitles efficiently",
        icon: FileText,
        articles: ["Auto-generate subtitles", "Translate subtitle files", "Subtitle to audio", "Format conversion"]
    },
    {
        title: "Troubleshooting",
        description: "Common issues and how to resolve them",
        icon: HelpCircle,
        articles: ["Audio quality issues", "Sync problems", "File format errors", "API limits"]
    },
];

export default function HelpPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-wrap justify-between items-center gap-4">
                <div className="flex items-center gap-5">
                    <div className="flex flex-col">
                        <h2 className="text-white text-2xl font-black leading-tight bg-gradient-to-r from-white via-primary-purple-bright to-white bg-clip-text text-transparent drop-shadow-[0_0_10px_rgba(168,85,247,0.4)]">
                            Octavia
                        </h2>
                        <p className="text-[10px] font-bold leading-tight tracking-[0.2em] bg-gradient-to-r from-primary-purple-bright via-accent-cyan to-primary-purple-bright bg-clip-text text-transparent text-glow-purple">
                            RISE BEYOND LANGUAGE
                        </p>
                    </div>

                    <div className="h-10 w-[1px] bg-white/10 hidden sm:block mx-1" />

                    <div>
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Help Center</h1>
                    </div>
                </div>
            </div>

            {/* Search Bar */}
            <div className="glass-panel p-4">
                <div className="relative">
                    <input
                        type="text"
                        placeholder="Search for help..."
                        className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-primary-purple/50 focus:ring-1 focus:ring-primary-purple/50"
                    />
                    <HelpCircle className="absolute right-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                </div>
            </div>

            {/* Help Topics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {helpTopics.map((topic, index) => {
                    const Icon = topic.icon;

                    return (
                        <motion.div
                            key={topic.title}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                            className="glass-panel-glow p-5"
                        >
                            <div className="glass-shine" />
                            <div className="relative z-10">
                                <div className="flex items-center gap-3 mb-3">
                                    <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary-purple/10 border border-primary-purple/20">
                                        <Icon className="w-5 h-5 text-primary-purple-bright" />
                                    </div>
                                    <h3 className="text-white font-bold text-base">{topic.title}</h3>
                                </div>
                                <p className="text-slate-400 text-sm mb-4">{topic.description}</p>

                                <div className="space-y-2">
                                    {topic.articles.map((article, idx) => (
                                        <a
                                            key={idx}
                                            href="#"
                                            className="flex items-center gap-2 text-sm text-slate-300 hover:text-primary-purple-bright transition-colors group"
                                        >
                                            <span className="w-1 h-1 rounded-full bg-primary-purple-bright" />
                                            <span className="group-hover:underline">{article}</span>
                                        </a>
                                    ))}
                                </div>
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            {/* Quick Links */}
            <div className="glass-panel p-6">
                <h2 className="text-xl font-bold text-white mb-4">Quick Links</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <a href="#" className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                        <Video className="w-5 h-5 text-primary-purple-bright" />
                        <div>
                            <p className="text-white text-sm font-medium">Video Tutorials</p>
                            <p className="text-slate-500 text-xs">Watch step-by-step guides</p>
                        </div>
                        <ExternalLink className="w-4 h-4 text-slate-500 ml-auto group-hover:text-white" />
                    </a>

                    <a href="#" className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                        <Book className="w-5 h-5 text-accent-cyan" />
                        <div>
                            <p className="text-white text-sm font-medium">Documentation</p>
                            <p className="text-slate-500 text-xs">Complete API reference</p>
                        </div>
                        <ExternalLink className="w-4 h-4 text-slate-500 ml-auto group-hover:text-white" />
                    </a>

                    <a href="#" className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                        <MessageCircle className="w-5 h-5 text-accent-pink" />
                        <div>
                            <p className="text-white text-sm font-medium">Community Forum</p>
                            <p className="text-slate-500 text-xs">Ask questions & share tips</p>
                        </div>
                        <ExternalLink className="w-4 h-4 text-slate-500 ml-auto group-hover:text-white" />
                    </a>
                </div>
            </div>
        </div>
    );
}
