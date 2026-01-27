"use client";

import { motion } from "framer-motion";
import { Volume2, Search, X, Info, Check, Globe, Play } from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import { useUser } from "@/contexts/UserContext";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Voice {
    id: number;
    name: string;
    language: string;
    language_code: string;
    voice_id: string;
    sample_text?: string;
}

interface LanguageEntry {
    language: string;
    locale: string;
    voice: Voice;
}

export default function MyVoicesPage() {
    const { user, refreshCredits } = useUser();
    const [languages, setLanguages] = useState<LanguageEntry[]>([]);
    const [filteredLanguages, setFilteredLanguages] = useState<LanguageEntry[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedVoice, setSelectedVoice] = useState<Voice | null>(null);
    const [previewText, setPreviewText] = useState("Hello! This is a preview of the voice for your translated content.");
    const [isPlayingPreview, setIsPlayingPreview] = useState(false);
    const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [error, setError] = useState<string | null>(null);

    const getToken = (): string | null => {
        if (typeof window === 'undefined') return null;
        const userStr = localStorage.getItem('octavia_user');
        if (userStr) {
            try {
                const user = JSON.parse(userStr);
                return user.token || null;
            } catch {
                return null;
            }
        }
        return null;
    };

    const isDemoUser = useMemo(() => {
        return user?.email === "demo@octavia.com" || false;
    }, [user]);

    useEffect(() => {
        fetchVoices();
    }, []);

    useEffect(() => {
        filterVoices();
    }, [languages, searchQuery]);

    const fetchVoices = async () => {
        try {
            setIsLoading(true);
            setError(null);
            const token = getToken();

            const response = await fetch(`${API_BASE_URL}/api/voices/all`, {
                headers: token ? {
                    'Authorization': `Bearer ${token}`
                } : {},
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch voices: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.success && data.voices_by_language) {
                const entries: LanguageEntry[] = [];
                let voiceId = 1;

                const languageNames: Record<string, string> = {
                    'en': 'English',
                    'es': 'Spanish',
                    'fr': 'French',
                    'de': 'German',
                    'it': 'Italian',
                    'pt': 'Portuguese',
                    'ru': 'Russian',
                    'ja': 'Japanese',
                    'ko': 'Korean',
                    'zh-cn': 'Chinese (Simplified)',
                    'ar': 'Arabic',
                    'hi': 'Hindi',
                    'nl': 'Dutch',
                    'pl': 'Polish',
                    'tr': 'Turkish',
                    'sv': 'Swedish'
                };

                Object.entries(data.voices_by_language as Record<string, any[]>).forEach(([langCode, voiceList]) => {
                    if (voiceList.length > 0) {
                        const voice = voiceList[0];
                        entries.push({
                            language: languageNames[langCode] || langCode,
                            locale: langCode,
                            voice: {
                                id: voiceId++,
                                name: voice.name,
                                language: languageNames[langCode] || langCode,
                                language_code: langCode,
                                voice_id: voice.voice_id,
                                sample_text: voice.sample_text
                            }
                        });
                    }
                });

                setLanguages(entries);

                if (entries.length > 0) {
                    setSelectedVoice(entries[0].voice);
                }
            } else {
                throw new Error("Invalid response format");
            }
        } catch (err) {
            console.error("Error fetching voices:", err);
            setError(err instanceof Error ? err.message : "Failed to load voices");
        } finally {
            setIsLoading(false);
        }
    };

    const filterVoices = () => {
        if (!languages.length) return;

        const query = searchQuery.toLowerCase();
        const filtered = languages.filter(entry => {
            return query === "" ||
                entry.language.toLowerCase().includes(query) ||
                entry.locale.toLowerCase().includes(query) ||
                entry.voice.name.toLowerCase().includes(query);
        });

        setFilteredLanguages(filtered);
    };

    const handlePreviewVoice = async (voice: Voice) => {
        try {
            setSelectedVoice(voice);

            if (!isDemoUser && (user?.credits || 0) < 1) {
                alert(`Insufficient credits. You need 1 credit but only have ${user?.credits || 0}.`);
                return;
            }

            setIsPlayingPreview(true);

            const token = getToken();
            if (!token && !isDemoUser) {
                alert("Please log in to preview voices");
                setIsPlayingPreview(false);
                return;
            }

            const formData = new FormData();
            formData.append('voice_id', voice.voice_id);
            formData.append('text', previewText);
            formData.append('language', voice.language_code);

            const headers: Record<string, string> = {};
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }

            const response = await fetch(`${API_BASE_URL}/api/voices/preview`, {
                method: 'POST',
                headers,
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorData = JSON.parse(errorText);
                    throw new Error(errorData.detail || errorData.error || "Preview generation failed");
                } catch {
                    throw new Error(`Preview generation failed (Status: ${response.status})`);
                }
            }

            const data = await response.json();

            if (data.success && data.preview_url) {
                const audioUrl = `${API_BASE_URL}${data.preview_url}`;
                const audio = new Audio(audioUrl);

                audio.onended = () => {
                    setIsPlayingPreview(false);
                };

                audio.onerror = () => {
                    setIsPlayingPreview(false);
                    alert("Failed to play preview audio");
                };

                setAudioElement(audio);
                audio.play().catch((err) => {
                    setIsPlayingPreview(false);
                    alert(`Failed to play audio: ${err.message}`);
                });

                refreshCredits();
            }
        } catch (err) {
            console.error("Preview error:", err);
            setIsPlayingPreview(false);
            const errorMessage = err instanceof Error ? err.message : "Unknown error";
            alert(`Preview failed: ${errorMessage}`);
        }
    };

    const stopPreview = () => {
        if (audioElement) {
            audioElement.pause();
            audioElement.currentTime = 0;
            setIsPlayingPreview(false);
        }
    };

    useEffect(() => {
        return () => {
            if (audioElement) {
                audioElement.pause();
                audioElement.currentTime = 0;
            }
        };
    }, [audioElement]);

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
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Voice Library</h1>
                    </div>
                </div>
            </div>

            {/* Search */}
            <div className="relative max-w-md">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                    type="text"
                    placeholder="Search languages..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="glass-input w-full pl-10 pr-4 py-2"
                />
                {searchQuery && (
                    <button
                        onClick={() => setSearchQuery("")}
                        className="absolute right-3 top-1/2 -translate-y-1/2"
                    >
                        <X className="w-4 h-4 text-slate-500 hover:text-white" />
                    </button>
                )}
            </div>

            {/* Error State */}
            {error && (
                <div className="glass-panel p-6 border-red-500/20 bg-red-500/10">
                    <div className="flex items-center gap-3 text-red-400">
                        <Info className="w-5 h-5" />
                        <span>{error}</span>
                    </div>
                    <button
                        onClick={fetchVoices}
                        className="mt-4 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm"
                    >
                        Retry
                    </button>
                </div>
            )}

            {/* Preview Section */}
            {selectedVoice && (
                <div className="glass-panel p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-xl bg-primary-purple/20 flex items-center justify-center">
                                <Volume2 className="w-6 h-6 text-primary-purple-bright" />
                            </div>
                            <div>
                                <h2 className="text-white font-bold text-lg">{selectedVoice.name}</h2>
                                <div className="flex items-center gap-2 text-sm text-slate-400">
                                    <Globe className="w-3 h-3" />
                                    <span>{selectedVoice.language}</span>
                                </div>
                            </div>
                        </div>
                        <div className="text-right text-sm text-slate-500">
                            <span className="font-mono bg-white/5 px-2 py-1 rounded">{selectedVoice.voice_id}</span>
                        </div>
                    </div>

                    <div className="mb-4">
                        <label className="text-white text-sm font-medium mb-2 block">Preview Text</label>
                        <textarea
                            value={previewText}
                            onChange={(e) => setPreviewText(e.target.value)}
                            className="glass-input w-full h-24 p-3"
                            placeholder="Enter text to preview..."
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <p className="text-slate-500 text-sm">Voice preview for translation</p>
                        <div className="flex items-center gap-3">
                            {isPlayingPreview ? (
                                <button onClick={stopPreview} className="btn-border-beam">
                                    <div className="btn-border-beam-inner flex items-center gap-2 px-6 py-2.5">
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                        <span>Stop</span>
                                    </div>
                                </button>
                            ) : (
                                <button onClick={() => handlePreviewVoice(selectedVoice)} className="btn-border-beam">
                                    <div className="btn-border-beam-inner flex items-center gap-2 px-6 py-2.5">
                                        <Play className="w-4 h-4" />
                                        <span>Preview Voice</span>
                                    </div>
                                </button>
                            )}
                        </div>
                    </div>

                    {isPlayingPreview && (
                        <div className="mt-4 p-3 bg-primary-purple/10 border border-primary-purple/20 rounded-lg">
                            <p className="text-primary-purple-bright text-sm flex items-center gap-2">
                                <Volume2 className="w-4 h-4" />
                                Playing preview... {isDemoUser ? "(Demo: Free)" : "(1 credit)"}
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Voices Grid */}
            {isLoading ? (
                <div className="text-center py-12">
                    <div className="inline-block w-8 h-8 border-2 border-primary-purple border-t-transparent rounded-full animate-spin"></div>
                    <p className="mt-4 text-slate-400">Loading voices...</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {filteredLanguages.map((entry, idx) => (
                        <motion.div
                            key={entry.locale}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className={`glass-panel p-5 cursor-pointer transition-all hover:ring-2 ${selectedVoice?.id === entry.voice.id
                                ? 'ring-2 ring-primary-purple bg-primary-purple/5'
                                : 'hover:bg-white/5'
                                }`}
                            onClick={() => setSelectedVoice(entry.voice)}
                        >
                            <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-lg bg-primary-purple/10 flex items-center justify-center">
                                        <Globe className="w-5 h-5 text-primary-purple-bright" />
                                    </div>
                                    <div>
                                        <h3 className="text-white font-bold">{entry.language}</h3>
                                        <span className="text-slate-500 text-xs font-mono">{entry.locale}</span>
                                    </div>
                                </div>
                                {selectedVoice?.id === entry.voice.id && (
                                    <span className="px-2 py-1 rounded-full bg-primary-purple/20 text-primary-purple-bright text-xs">
                                        Selected
                                    </span>
                                )}
                            </div>

                            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
                                <span className="text-slate-500 text-sm">{entry.voice.name}</span>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handlePreviewVoice(entry.voice);
                                    }}
                                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary-purple/20 hover:bg-primary-purple/30 text-primary-purple-bright text-xs transition-colors"
                                >
                                    <Play className="w-3 h-3" />
                                    Preview
                                </button>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* No Results */}
            {!isLoading && filteredLanguages.length === 0 && !error && (
                <div className="text-center py-12">
                    <Globe className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                    <p className="text-slate-400">No languages found.</p>
                    <button
                        onClick={() => setSearchQuery("")}
                        className="mt-4 text-primary-purple-bright hover:text-primary-purple text-sm"
                    >
                        Clear search
                    </button>
                </div>
            )}

            {/* Info Card */}
            <div className="glass-card p-4">
                <h3 className="text-white font-semibold mb-2 flex items-center gap-2">
                    <Info className="w-4 h-4 text-primary-purple-bright" />
                    About Voice Selection
                </h3>
                <ul className="text-slate-400 text-sm space-y-2 pl-6 list-disc">
                    <li>Each language has one high-quality voice</li>
                    <li>Preview voices with custom text before translation</li>
                    <li>The selected voice will be used for your translated content</li>
                </ul>
            </div>
        </div>
    );
}
