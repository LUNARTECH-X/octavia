"use client";

import { motion } from "framer-motion";
import { LifeBuoy, Mail, MessageCircle, Send, Clock } from "lucide-react";

export default function SupportPage() {
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
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Support</h1>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Contact Form */}
                <div className="lg:col-span-2">
                    <div className="glass-panel p-6">
                        <h2 className="text-xl font-bold text-white mb-4">Contact Support</h2>

                        <form className="space-y-4">
                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Subject</label>
                                <input
                                    type="text"
                                    placeholder="Brief description of your issue"
                                    className="glass-input w-full"
                                />
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Category</label>
                                <select className="glass-select w-full">
                                    <option>Select a category</option>
                                    <option>Technical Issue</option>
                                    <option>Billing Question</option>
                                    <option>Feature Request</option>
                                    <option>Account Problem</option>
                                    <option>Other</option>
                                </select>
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Priority</label>
                                <select className="glass-select w-full">
                                    <option>Normal</option>
                                    <option>High</option>
                                    <option>Urgent</option>
                                </select>
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Description</label>
                                <textarea
                                    rows={6}
                                    placeholder="Please provide as much detail as possible..."
                                    className="glass-input w-full resize-none"
                                />
                            </div>

                            <div>
                                <label className="text-white text-sm font-semibold mb-2 block">Attachments (optional)</label>
                                <div className="glass-input flex items-center justify-center py-8 cursor-pointer hover:bg-white/10 transition-colors">
                                    <div className="text-center">
                                        <p className="text-slate-400 text-sm">Click to upload files</p>
                                        <p className="text-slate-600 text-xs mt-1">Images, videos, or documents</p>
                                    </div>
                                </div>
                            </div>

                            <button type="submit" className="btn-border-beam w-full group">
                                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                                    <Send className="w-4 h-4" />
                                    <span>Submit Ticket</span>
                                </div>
                            </button>
                        </form>
                    </div>
                </div>

                {/* Support Info Sidebar */}
                <div className="space-y-4">
                    {/* Response Time */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="glass-panel-glow p-5"
                    >
                        <div className="glass-shine" />
                        <div className="relative z-10">
                            <div className="flex items-center gap-3 mb-3">
                                <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary-purple/10 border border-primary-purple/20">
                                    <Clock className="w-5 h-5 text-primary-purple-bright" />
                                </div>
                                <h3 className="text-white font-bold text-sm">Response Time</h3>
                            </div>
                            <p className="text-slate-400 text-sm mb-2">We typically respond within:</p>
                            <ul className="space-y-1 text-sm">
                                <li className="flex items-center gap-2">
                                    <span className="w-1 h-1 rounded-full bg-accent-cyan" />
                                    <span className="text-slate-300">Urgent: <span className="text-white font-semibold">1-2 hours</span></span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <span className="w-1 h-1 rounded-full bg-yellow-400" />
                                    <span className="text-slate-300">High: <span className="text-white font-semibold">4-6 hours</span></span>
                                </li>
                                <li className="flex items-center gap-2">
                                    <span className="w-1 h-1 rounded-full bg-blue-400" />
                                    <span className="text-slate-300">Normal: <span className="text-white font-semibold">24 hours</span></span>
                                </li>
                            </ul>
                        </div>
                    </motion.div>

                    {/* Contact Methods */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 }}
                        className="glass-panel p-5"
                    >
                        <h3 className="text-white font-bold text-sm mb-4">Other Ways to Reach Us</h3>
                        <div className="space-y-3">
                            <a href="mailto:support@octavia.ai" className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                                <Mail className="w-5 h-5 text-primary-purple-bright" />
                                <div>
                                    <p className="text-white text-sm font-medium">Email</p>
                                    <p className="text-slate-500 text-xs">support@octavia.ai</p>
                                </div>
                            </a>

                            <a href="#" className="flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group">
                                <MessageCircle className="w-5 h-5 text-accent-cyan" />
                                <div>
                                    <p className="text-white text-sm font-medium">Live Chat</p>
                                    <p className="text-slate-500 text-xs">Available 9am-5pm EST</p>
                                </div>
                            </a>
                        </div>
                    </motion.div>

                    {/* Support Status */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                        className="glass-panel p-5 relative overflow-hidden"
                    >
                        <div className="relative z-10">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="w-1 h-1 rounded-full bg-accent-cyan" />
                                Live Support Available
                            </div>
                            <div className="text-3xl font-bold text-white mb-2">~5 Min</div>
                            <p className="text-slate-400 text-sm">Average response time</p>
                        </div>
                        <div className="absolute top-6 right-6 flex items-center justify-center">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-cyan opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-accent-cyan"></span>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
