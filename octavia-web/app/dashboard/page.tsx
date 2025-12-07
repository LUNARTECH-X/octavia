// app/dashboard/page.tsx - Updated with localStorage check
"use client";

import { motion } from "framer-motion";
import { Video, Mic, FileText, Languages, AudioWaveform, Sparkles } from "lucide-react";
import Link from "next/link";
import { useUser } from "@/contexts/UserContext";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

const features = [
    {
        title: "Video Translation",
        subtitle: "Magic Mode",
        description: "Translate full-length videos with perfectly synced dubbed audio",
        icon: Video,
        color: "primary-purple",
        href: "/dashboard/video",
        highlights: ["10-hour videos", "Voice cloning", "Multi-speaker"]
    },
    {
        title: "Audio Translation",
        subtitle: "Voice Cloning",
        description: "Translate podcasts and audio files with voice cloning",
        icon: Mic,
        color: "accent-cyan",
        href: "/dashboard/audio",
        highlights: ["Long-form content", "Voice cloning", "Parallel processing"]
    },
    {
        title: "Subtitle Generation",
        subtitle: "Auto-Detect",
        description: "Auto-generate accurate subtitles from video or audio",
        icon: FileText,
        color: "accent-pink",
        href: "/dashboard/subtitles",
        highlights: ["50+ languages", "Auto-detect", "Word-level sync"]
    },
    {
        title: "Subtitle Translation",
        subtitle: "Context-Aware AI",
        description: "Translate SRT/VTT files with context-aware AI",
        icon: Languages,
        color: "green-500",
        href: "/dashboard/subtitles/translate",
        highlights: ["GPT-4/Claude", "Context-aware", "Format preservation"]
    },
    {
        title: "Subtitle to Audio",
        subtitle: "Multi-Voice",
        description: "Convert written subtitles into natural-sounding speech",
        icon: AudioWaveform,
        color: "orange-500",
        href: "/dashboard/audio/subtitle-to-audio",
        highlights: ["Multi-voice", "Natural speech", "Custom parameters"]
    },
    {
        title: "My Voices",
        subtitle: "Voice Library",
        description: "Manage your custom voice clones and voice profiles",
        icon: Sparkles,
        color: "purple-400",
        href: "/dashboard/voices",
        highlights: ["Voice library", "Custom clones", "Quick access"]
    }
];

const getColorClasses = (color: string) => {
    const colorMap: Record<string, { bg: string, border: string, text: string, glow: string }> = {
        "primary-purple": {
            bg: "bg-primary-purple/10",
            border: "border-primary-purple/20 group-hover:border-primary-purple/40",
            text: "text-primary-purple-bright",
            glow: "glow-purple"
        },
        "accent-cyan": {
            bg: "bg-accent-cyan/10",
            border: "border-accent-cyan/20 group-hover:border-accent-cyan/40",
            text: "text-accent-cyan",
            glow: "glow-cyan"
        },
        "accent-pink": {
            bg: "bg-accent-pink/10",
            border: "border-accent-pink/20 group-hover:border-accent-pink/40",
            text: "text-accent-pink",
            glow: "glow-pink"
        },
        "green-500": {
            bg: "bg-green-500/10",
            border: "border-green-500/20 group-hover:border-green-500/40",
            text: "text-green-400",
            glow: "glow-green"
        },
        "orange-500": {
            bg: "bg-orange-500/10",
            border: "border-orange-500/20 group-hover:border-orange-500/40",
            text: "text-orange-400",
            glow: "glow-orange"
        },
        "purple-400": {
            bg: "bg-purple-400/10",
            border: "border-purple-400/20 group-hover:border-purple-400/40",
            text: "text-purple-400",
            glow: "glow-purple"
        }
    };
    return colorMap[color] || colorMap["primary-purple"];
};

interface UserData {
    email: string;
    name: string;
    token: string;
    credits: number;
    verified: boolean;
}

export default function DashboardPage() {
    const router = useRouter();
    const { user, setUser } = useUser();
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Check if user is authenticated
        const checkAuth = () => {
            setIsLoading(true);
            
            try {
                // Check UserContext first (if using context)
                if (user) {
                    setIsLoading(false);
                    return;
                }
                
                // Check localStorage as fallback
                const userData = localStorage.getItem('octavia_user');
                
                if (userData) {
                    try {
                        const parsedUser: UserData = JSON.parse(userData);
                        
                        // Validate the user data structure
                        if (parsedUser.email && parsedUser.token) {
                            // Update context if setUser function exists
                            if (setUser) {
                                setUser(parsedUser);
                            }
                            setIsLoading(false);
                            return;
                        } else {
                            // Invalid user data structure
                            console.error("Invalid user data structure");
                            redirectToLogin();
                        }
                    } catch (parseError) {
                        console.error("Error parsing user data:", parseError);
                        redirectToLogin();
                    }
                } else {
                    // No user data found
                    redirectToLogin();
                }
            } catch (error) {
                console.error("Auth check error:", error);
                redirectToLogin();
            }
        };

        const redirectToLogin = () => {
            localStorage.removeItem('octavia_user');
            router.push('/login');
        };

        checkAuth();
    }, [router, user, setUser]);

    // Show loading state
    if (isLoading) {
        return (
            <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 border-4 border-primary-purple/30 border-t-primary-purple rounded-full animate-spin" />
                    <p className="text-slate-400">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    // Get user data from localStorage if context doesn't have it
    const currentUser = user || (() => {
        try {
            const userData = localStorage.getItem('octavia_user');
            return userData ? JSON.parse(userData) : null;
        } catch {
            return null;
        }
    })();

    // Double check - if still no user, redirect
    if (!currentUser) {
        router.push('/login');
        return null;
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-wrap justify-between gap-3">
                <div>
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple">Hub</h1>
                    <p className="text-slate-400 text-sm">Choose a workflow to get started with AI-powered translation</p>
                </div>
                
                {currentUser && (
                    <div className="flex items-center gap-4">
                        <div className="glass-card px-4 py-2">
                            <div className="text-sm text-slate-400">Available Credits</div>
                            <div className="text-xl font-bold text-primary-purple-bright">{currentUser.credits || 0}</div>
                        </div>
                    </div>
                )}
            </div>

            {/* Welcome Message */}
            {currentUser && (
                <div className="glass-panel p-6">
                    <h2 className="text-xl font-bold text-white mb-2">Welcome back, {currentUser.name}!</h2>
                    <p className="text-slate-400">
                        {currentUser.verified 
                            ? "Your email is verified and ready for translation magic! ✨"
                            : "Please verify your email to access all features."}
                    </p>
                    <div className="flex items-center gap-2 mt-2">
                        <span className={`text-xs px-2 py-1 rounded-full ${currentUser.verified ? 'bg-green-500/20 text-green-400 border border-green-500/30' : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'}`}>
                            {currentUser.verified ? '✅ Verified' : '⚠️ Not Verified'}
                        </span>
                        <span className="text-xs text-slate-500">
                            {currentUser.email}
                        </span>
                    </div>
                </div>
            )}

                        {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 auto-rows-fr">
                {features.map((feature, index) => {
                    const colors = getColorClasses(feature.color);
                    const Icon = feature.icon;

                    return (
                        <Link key={feature.href} href={feature.href}>
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.1 }}
                                whileHover={{ y: -4 }}
                                className="glass-panel-glow p-4 relative group cursor-pointer h-full"
                            >
                                <div className="glass-shine" />
                                <div className="relative z-10 flex flex-col">
                                    {/* Icon */}
                                    <div className={`flex items-center justify-center w-12 h-12 rounded-xl ${colors.bg} border ${colors.border} mb-3 group-hover:scale-110 transition-all duration-300`}>
                                        <Icon className={`w-6 h-6 ${colors.text}`} />
                                    </div>

                                    {/* Title */}
                                    <div className="mb-2">
                                        <h2 className="text-white text-base font-bold leading-tight">{feature.title}</h2>
                                        {feature.subtitle && (
                                            <span className={`text-xs font-semibold ${colors.text}`}>{feature.subtitle}</span>
                                        )}
                                    </div>

                                    {/* Description */}
                                    <p className="text-slate-400 text-xs font-normal leading-relaxed mb-3">
                                        {feature.description}
                                    </p>

                                    {/* Highlights */}
                                    <div className="flex flex-wrap gap-1.5">
                                        {feature.highlights.map((highlight, idx) => (
                                            <span
                                                key={idx}
                                                className={`text-[10px] px-2 py-0.5 rounded-full ${colors.bg} border ${colors.border} ${colors.text}`}
                                            >
                                                {highlight}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                                {/* Localized Glow */}
                                <div className={colors.glow} style={{ width: "250px", height: "250px", top: "-50%", left: "50%", transform: "translateX(-50%)" }} />
                            </motion.div>
                        </Link>
                    );
                })}
            </div>
        </div>
    );
}