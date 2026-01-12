"use client";

import { motion } from "framer-motion";
import { Plus, Mic, Play, MoreVertical, Trash2, Edit2, Globe, Volume2 } from "lucide-react";
import { useState, useEffect } from "react";
import { useUser } from "@/contexts/UserContext";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Voice {
    id: number;
    name: string;
    type: string;
    language: string;
    language_code: string;
    voice_id: string;
    gender: string;
    sample_text?: string;
    date: string;
}

export default function MyVoicesPage() {
    const { user, refreshCredits } = useUser();
    const [voices, setVoices] = useState<Voice[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [selectedVoice, setSelectedVoice] = useState<Voice | null>(null);
    const [previewText, setPreviewText] = useState("Hello! This is a preview of the voice.");
    const [isPlayingPreview, setIsPlayingPreview] = useState(false);
    const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

    // Get authentication token
    const getToken = (): string | null => {
        if (typeof window === 'undefined') return null;
        const userStr = localStorage.getItem('octavia_user');
        if (userStr) {
            try {
                const user = JSON.parse(userStr);
                return user.token || null;
            } catch (error) {
                return null;
            }
        }
        return null;
    };

    // Fetch available voices from backend
    useEffect(() => {
        fetchVoices();
    }, []);

    const fetchVoices = async () => {
        try {
            setIsLoading(true);
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

            console.log("API Response:", data); // Debug log

            if (data.success && data.voices_by_language) {
                // Transform the data to match our frontend structure
                const allVoices: Voice[] = [];
                let voiceId = 1;

                Object.entries(data.voices_by_language as Record<string, any[]>).forEach(([language, voiceList]) => {
                    voiceList.forEach((voice: any) => {
                        allVoices.push({
                            id: voiceId++,
                            name: voice.name,
                            type: voice.type || "Synthetic",
                            language: language,
                            language_code: voice.language_code,
                            voice_id: voice.voice_id || voice.id,
                            gender: voice.gender,
                            sample_text: voice.sample_text,
                            date: "Available"
                        });
                    });
                });

                setVoices(allVoices);

                // Select first voice by default
                if (allVoices.length > 0) {
                    setSelectedVoice(allVoices[0]);
                }

                console.log("Processed voices:", allVoices); // Debug log
            } else {
                // Use fallback data if response structure is unexpected
                console.warn("Unexpected API response structure, using fallback data");
                const fallbackVoices = [
                    { id: 1, name: "Aria (Female)", type: "Synthetic", language: "English", language_code: "en", gender: "Female", voice_id: "aria_female", date: "Available" },
                    { id: 2, name: "David (Male)", type: "Synthetic", language: "English", language_code: "en", gender: "Male", voice_id: "david_male", date: "Available" },
                    { id: 3, name: "Elena (Female)", type: "Synthetic", language: "Spanish", language_code: "es", gender: "Female", voice_id: "elena_female", date: "Available" },
                    { id: 4, name: "Alvaro (Male)", type: "Synthetic", language: "Spanish", language_code: "es", gender: "Male", voice_id: "alvaro_male", date: "Available" },
                    { id: 5, name: "Denise (Female)", type: "Synthetic", language: "French", language_code: "fr", gender: "Female", voice_id: "denise_female", date: "Available" },
                    { id: 6, name: "Henri (Male)", type: "Synthetic", language: "French", language_code: "fr", gender: "Male", voice_id: "henri_male", date: "Available" },
                ];

                setVoices(fallbackVoices);

                if (fallbackVoices.length > 0) {
                    setSelectedVoice(fallbackVoices[0]);
                }
            }
        } catch (error) {
            console.error("Error fetching voices:", error);
            // Fallback to static data if API fails
            const fallbackVoices = [
                { id: 1, name: "Aria (Female)", type: "Synthetic", language: "English", language_code: "en", gender: "Female", voice_id: "aria_female", date: "Available" },
                { id: 2, name: "David (Male)", type: "Synthetic", language: "English", language_code: "en", gender: "Male", voice_id: "david_male", date: "Available" },
                { id: 3, name: "Elena (Female)", type: "Synthetic", language: "Spanish", language_code: "es", gender: "Female", voice_id: "elena_female", date: "Available" },
                { id: 4, name: "Alvaro (Male)", type: "Synthetic", language: "Spanish", language_code: "es", gender: "Male", voice_id: "alvaro_male", date: "Available" },
                { id: 5, name: "Denise (Female)", type: "Synthetic", language: "French", language_code: "fr", gender: "Female", voice_id: "denise_female", date: "Available" },
                { id: 6, name: "Henri (Male)", type: "Synthetic", language: "French", language_code: "fr", gender: "Male", voice_id: "henri_male", date: "Available" },
            ];

            setVoices(fallbackVoices);

            if (fallbackVoices.length > 0) {
                setSelectedVoice(fallbackVoices[0]);
            }
        } finally {
            setIsLoading(false);
        }
    };

    // Handle voice preview
    const handlePreviewVoice = async (voice: Voice) => {
        try {
            setSelectedVoice(voice);

            // Check if user has enough credits (1 credit per preview)
            if ((user?.credits || 0) < 1) {
                alert(`Insufficient credits. You need 1 credit but only have ${user?.credits || 0}.`);
                return;
            }

            setIsPlayingPreview(true);

            const token = getToken();
            if (!token) {
                alert("Please log in to preview voices");
                setIsPlayingPreview(false);
                return;
            }

            const formData = new FormData();
            formData.append('voice_id', voice.voice_id);
            formData.append('text', previewText);
            formData.append('language', voice.language_code);

            const response = await fetch(`${API_BASE_URL}/api/voices/preview`, {
                method: 'POST',
                headers: token ? {
                    'Authorization': `Bearer ${token}`
                } : {},
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Preview API error:', response.status, errorText);
                try {
                    const errorData = JSON.parse(errorText);
                    throw new Error(errorData.error || errorData.detail || "Preview generation failed");
                } catch {
                    throw new Error(`Preview generation failed (Status: ${response.status})`);
                }
            }

            const data = await response.json();

            if (data.success && data.preview_url) {
                // Create audio element and play
                const audioUrl = `${API_BASE_URL}${data.preview_url}`;
                console.log("Audio URL:", audioUrl);
                const audio = new Audio(audioUrl);

                audio.onloadedmetadata = () => {
                    console.log("Audio metadata loaded:", audio.duration, "seconds");
                };

                audio.onloadeddata = () => {
                    console.log("Audio data loaded");
                };

                audio.oncanplay = () => {
                    console.log("Audio can play");
                };

                audio.onended = () => {
                    console.log("Audio playback ended");
                    setIsPlayingPreview(false);
                };

                audio.onerror = (e) => {
                    console.error("Audio error:", e);
                    console.error("Audio error code:", audio.error?.code);
                    console.error("Audio error message:", audio.error?.message);
                    setIsPlayingPreview(false);

                    let errorMsg = "Failed to play preview audio";
                    if (audio.error?.message) {
                        errorMsg += `: ${audio.error.message}`;
                    } else if (audio.error?.code === 4) {
                        errorMsg = "Audio file format error. The voice preview service may be temporarily unavailable. Please try again later.";
                    } else if (audio.error?.code === 3) {
                        errorMsg = "Audio decoding error. Please try a different voice.";
                    } else if (audio.error?.code === 2) {
                        errorMsg = "Network error loading audio. Please check your connection.";
                    }

                    alert(errorMsg);
                };

                setAudioElement(audio);
                audio.play().catch((err) => {
                    console.error("Failed to play audio:", err);
                    setIsPlayingPreview(false);
                    alert(`Failed to play audio: ${err.message}`);
                });

                console.log("Preview credits remaining:", data.credits_remaining);

                // Refresh credits after preview
                refreshCredits();
            }
        } catch (error) {
            console.error("Preview error:", error);
            setIsPlayingPreview(false);

            const errorMessage = error instanceof Error ? error.message : "Unknown error";

            // Show detailed error message
            if (errorMessage.includes("Voice preview service is currently unavailable")) {
                alert(
                    "Voice Preview Service Unavailable\n\n" +
                    "The Microsoft TTS service is experiencing connectivity issues.\n" +
                    "Please try again later or use the Subtitle-to-Audio feature which uses a different TTS service."
                );
            } else if (errorMessage.includes("No audio data received")) {
                alert(
                    "Audio Generation Failed\n\n" +
                    "No audio was generated from the TTS service.\n" +
                    "This may be a network issue. Please try again."
                );
            } else if (errorMessage.includes("Not enough credits")) {
                alert(errorMessage);
            } else {
                alert(`Preview failed: ${errorMessage}`);
            }
        }
    };

    // Stop preview
    const stopPreview = () => {
        if (audioElement) {
            audioElement.pause();
            audioElement.currentTime = 0;
            setIsPlayingPreview(false);
        }
    };

    // Clean up audio on unmount
    useEffect(() => {
        return () => {
            if (audioElement) {
                audioElement.pause();
                audioElement.currentTime = 0;
            }
        };
    }, [audioElement]);

    // Group voices by language
    const voicesByLanguage = voices.reduce((groups: Record<string, Voice[]>, voice) => {
        const lang = voice.language;
        if (!groups[lang]) {
            groups[lang] = [];
        }
        groups[lang].push(voice);
        return groups;
    }, {});

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Voice Library</h1>
                    <p className="text-slate-400 text-sm">Preview and manage available AI voices</p>
                </div>
                <div className="flex items-center gap-4">
                    <div className="glass-card px-4 py-2">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-primary-purple-bright animate-pulse"></div>
                            <span className="text-white text-sm">Credits: <span className="font-bold">{user?.credits || 0}</span></span>
                        </div>
                        <p className="text-slate-400 text-xs mt-1">1 credit per preview</p>
                    </div>
                    <div className="text-sm text-slate-400">
                        {isLoading ? "Loading..." : `${voices.length} voices available`}
                    </div>
                </div>
            </div>

            {/* Preview Section */}
            {selectedVoice && (
                <div className="glass-panel p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-white font-bold text-lg">Voice Preview</h2>
                        <div className="flex items-center gap-2 text-sm text-slate-400">
                            <Globe className="w-4 h-4" />
                            <span>{selectedVoice.language}</span>
                            <span className="px-2 py-0.5 rounded-full bg-white/5 text-xs">
                                {selectedVoice.gender}
                            </span>
                        </div>
                    </div>

                    <div className="mb-4">
                        <label className="text-white text-sm font-medium mb-2 block">Preview Text</label>
                        <textarea
                            value={previewText}
                            onChange={(e) => setPreviewText(e.target.value)}
                            className="glass-input w-full h-24 p-3"
                            placeholder="Enter text to preview the voice..."
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-white font-bold">{selectedVoice.name}</h3>
                            <p className="text-slate-400 text-sm">{selectedVoice.type} Voice</p>
                        </div>

                        <button
                            onClick={() => handlePreviewVoice(selectedVoice)}
                            disabled={isPlayingPreview}
                            className="btn-border-beam"
                        >
                            <div className="btn-border-beam-inner flex items-center gap-2 px-6 py-2.5">
                                {isPlayingPreview ? (
                                    <>
                                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                        <span>Playing...</span>
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4" />
                                        <span>Preview Voice</span>
                                    </>
                                )}
                            </div>
                        </button>
                    </div>

                    {isPlayingPreview && (
                        <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                            <p className="text-blue-400 text-sm flex items-center gap-2">
                                <Volume2 className="w-4 h-4" />
                                Playing preview... (1 credit will be deducted)
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Voices by Language */}
            {isLoading ? (
                <div className="text-center py-12">
                    <div className="inline-block w-8 h-8 border-2 border-primary-purple border-t-transparent rounded-full animate-spin"></div>
                    <p className="mt-4 text-slate-400">Loading voices...</p>
                </div>
            ) : voices.length === 0 ? (
                <div className="text-center py-12">
                    <p className="text-slate-400">No voices available. Please check your connection or try again later.</p>
                </div>
            ) : (
                Object.entries(voicesByLanguage).map(([language, langVoices]) => (
                    <div key={language} className="space-y-4">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-lg bg-primary-purple/10 flex items-center justify-center">
                                <Globe className="w-4 h-4 text-primary-purple-bright" />
                            </div>
                            <h2 className="text-white font-bold text-xl">{language}</h2>
                            <span className="text-slate-500 text-sm">({langVoices.length} voices)</span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                            {langVoices.map((voice) => (
                                <motion.div
                                    key={voice.id}
                                    whileHover={{ y: -4 }}
                                    className={`glass-panel-glow p-5 relative group cursor-pointer transition-all ${selectedVoice?.id === voice.id ? 'ring-2 ring-primary-purple' : ''
                                        }`}
                                    onClick={() => setSelectedVoice(voice)}
                                >
                                    <div className="glass-shine" />

                                    <div className="flex justify-between items-start mb-4 relative z-10">
                                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center border ${voice.gender === 'Female'
                                            ? 'bg-pink-500/10 border-pink-500/20'
                                            : 'bg-blue-500/10 border-blue-500/20'
                                            }`}>
                                            <Mic className={`w-5 h-5 ${voice.gender === 'Female'
                                                ? 'text-pink-500'
                                                : 'text-blue-500'
                                                }`} />
                                        </div>
                                        <button className="text-slate-500 hover:text-white transition-colors">
                                            <MoreVertical className="w-4 h-4" />
                                        </button>
                                    </div>

                                    <div className="relative z-10">
                                        <h3 className="text-white font-bold text-lg mb-1">{voice.name}</h3>
                                        <div className="flex items-center gap-2 mb-4">
                                            <span className={`text-xs px-2 py-0.5 rounded-full ${voice.type === 'Cloned'
                                                ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                                                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                                }`}>
                                                {voice.type}
                                            </span>
                                            <span className={`text-xs px-2 py-0.5 rounded-full ${voice.gender === 'Female'
                                                ? 'bg-pink-500/10 text-pink-400 border border-pink-500/20'
                                                : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                                }`}>
                                                {voice.gender}
                                            </span>
                                        </div>

                                        <div className="flex items-center gap-2 mt-4 pt-4 border-t border-white/5">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handlePreviewVoice(voice);
                                                }}
                                                className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-xs font-medium text-white transition-colors"
                                            >
                                                <Play className="w-3 h-3" />
                                                Preview
                                            </button>
                                            <button
                                                onClick={(e) => e.stopPropagation()}
                                                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                                            >
                                                <Edit2 className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </div>
                ))
            )}

            {/* Instructions */}
            <div className="glass-card p-4 mt-8">
                <h3 className="text-white font-semibold mb-2">How to use voices:</h3>
                <ol className="text-slate-400 text-sm space-y-2 pl-4 list-decimal">
                    <li>Select a voice from the library to preview it</li>
                    <li>Customize the preview text to hear how it sounds with your content</li>
                    <li>Click "Preview Voice" to generate a sample (1 credit per preview)</li>
                    <li>When translating subtitles or audio, choose your preferred voice from the dropdown</li>
                    <li>Different voices work better for different content types (narration, dialogue, etc.)</li>
                </ol>
                <div className="mt-4 p-3 bg-primary-purple/10 border border-primary-purple/20 rounded-lg">
                    <p className="text-primary-purple-bright text-sm">
                        <strong>Note:</strong> Each voice preview costs 1 credit. Voice selection is available in subtitle-to-audio and audio translation features.
                    </p>
                </div>
            </div>
        </div>
    );
}