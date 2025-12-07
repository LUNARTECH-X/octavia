"use client";

import { motion } from "framer-motion";
import { FileText, Languages, Upload, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { useState, useCallback } from "react";
import { useUser } from "@/contexts/UserContext";
import { api } from "@/lib/api";

export default function SubtitleTranslatePage() {
    const { user } = useUser();
    const [sourceLanguage, setSourceLanguage] = useState("en");
    const [targetLanguage, setTargetLanguage] = useState("es");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isTranslating, setIsTranslating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const handleFileSelect = useCallback((file: File) => {
        // Check file type
        const validExtensions = ['.srt', '.vtt', '.ass', '.ssa'];
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();

        if (!validExtensions.includes(fileExt)) {
            setError('Please upload a valid subtitle file (SRT, VTT, ASS, SSA)');
            return;
        }

        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            setError('File size must be less than 10MB');
            return;
        }

        setSelectedFile(file);
        setError(null);
        setSuccess(null);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    }, [handleFileSelect]);

    const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    }, [handleFileSelect]);

    const handleTranslate = async () => {
        if (!selectedFile) {
            setError('Please select a subtitle file first');
            return;
        }

        if (!sourceLanguage || !targetLanguage) {
            setError('Please select both source and target languages');
            return;
        }

        if (sourceLanguage === targetLanguage) {
            setError('Source and target languages must be different');
            return;
        }

        setIsTranslating(true);
        setError(null);
        setSuccess(null);

        try {
            // Check user credits first (reduced to 5 credits for subtitle translation)
            const creditsResponse = await api.getUserCredits();
            if (!creditsResponse.success || !creditsResponse.data || creditsResponse.data.credits < 5) {
                setError('Insufficient credits. You need at least 5 credits to translate subtitles.');
                return;
            }

            // Start translation
            const result = await api.translateSubtitleFile({
                file: selectedFile,
                sourceLanguage,
                targetLanguage
            });

            if (result.success) {
                setSuccess('Translation completed! Your file should download automatically.');
                setSelectedFile(null); // Reset file selection

                // Auto-download the translated file if available
                if (result.download_url) {
                    // The backend will handle the download automatically
                    console.log('Translation successful, download URL:', result.download_url);
                }
            } else {
                setError(result.error || 'Translation failed');
            }
        } catch (err) {
            console.error('Translation error:', err);
            // Better error handling for API response objects
            let errorMessage = 'An unexpected error occurred';
            if (err instanceof Error) {
                errorMessage = err.message;
            } else if (typeof err === 'object' && err !== null) {
                // Handle API response error objects - use type assertion for known properties
                const errorObj = err as any;
                if (errorObj.error) {
                    errorMessage = errorObj.error;
                } else if (errorObj.message) {
                    errorMessage = errorObj.message;
                } else if (errorObj.detail) {
                    errorMessage = errorObj.detail;
                } else {
                    // Convert object to string for debugging
                    errorMessage = `Translation failed: ${JSON.stringify(err)}`;
                }
            } else if (typeof err === 'string') {
                errorMessage = err;
            }
            setError(errorMessage);
        } finally {
            setIsTranslating(false);
        }
    };

    const languageOptions = [
        { value: 'en', label: 'English' },
        { value: 'es', label: 'Spanish' },
        { value: 'fr', label: 'French' },
        { value: 'de', label: 'German' },
        { value: 'it', label: 'Italian' },
        { value: 'pt', label: 'Portuguese' },
        { value: 'ru', label: 'Russian' },
        { value: 'ja', label: 'Japanese' },
        { value: 'ko', label: 'Korean' },
        { value: 'zh', label: 'Chinese' },
        { value: 'ar', label: 'Arabic' },
        { value: 'hi', label: 'Hindi' },
    ];

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-col gap-2">
                <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle Translation</h1>
                <p className="text-slate-400 text-sm">Translate existing subtitle files to another language using AI</p>
            </div>

            {/* Status Messages */}
            {error && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel border-red-500/30 bg-red-500/10 p-4"
                >
                    <div className="flex items-center gap-3">
                        <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                        <p className="text-red-400 text-sm">{error}</p>
                    </div>
                </motion.div>
            )}

            {success && (
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-panel border-green-500/30 bg-green-500/10 p-4"
                >
                    <div className="flex items-center gap-3">
                        <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                        <p className="text-green-400 text-sm">{success}</p>
                    </div>
                </motion.div>
            )}

            {/* Upload Zone */}
            <motion.div
                whileHover={{ scale: selectedFile ? 1 : 1.01 }}
                className={`glass-panel relative border-2 ${selectedFile ? 'border-primary-purple-bright/50' : 'border-dashed border-primary-purple/30'} hover:border-primary-purple/50 transition-all cursor-pointer mb-6 overflow-hidden ${isDragging ? 'scale-[1.02]' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => !selectedFile && document.getElementById('file-input')?.click()}
            >
                <div className="glass-shine" />
                <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

                <div className="relative z-20 py-12 px-6">
                    {selectedFile ? (
                        <div className="flex flex-col items-center justify-center gap-3 text-center">
                            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-green-500/10 border border-green-500/30">
                                <FileText className="w-8 h-8 text-green-400" />
                            </div>
                            <div>
                                <h3 className="text-white text-lg font-bold mb-1 text-glow-green">{selectedFile.name}</h3>
                                <p className="text-slate-400 text-sm">
                                    {(selectedFile.size / 1024).toFixed(1)} KB • Subtitle file
                                </p>
                            </div>
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setSelectedFile(null);
                                }}
                                className="text-sm text-red-400 hover:text-red-300 mt-2"
                            >
                                Remove file
                            </button>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center gap-3 text-center">
                            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-purple/10 border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform">
                                {isTranslating ? <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" /> : <Upload className="w-8 h-8 text-primary-purple-bright" />}
                            </div>
                            <div>
                                <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                                    {isDragging ? 'Drop subtitle file here' : 'Drop subtitle file here'}
                                </h3>
                                <p className="text-slate-400 text-sm">or click to browse files • SRT, VTT, ASS, SSA supported • Max 10MB</p>
                            </div>
                        </div>
                    )}
                </div>

                <input
                    id="file-input"
                    type="file"
                    accept=".srt,.vtt,.ass,.ssa"
                    onChange={handleFileInput}
                    className="hidden"
                />
            </motion.div>

            {/* Configuration */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block flex items-center gap-2">
                        <span>Source Language</span>
                        {selectedFile && (
                            <span className="text-xs text-slate-500">(Detected from file content)</span>
                        )}
                    </label>
                    <select
                        className="glass-select w-full"
                        value={sourceLanguage}
                        onChange={(e) => setSourceLanguage(e.target.value)}
                        disabled={isTranslating}
                    >
                        <option value="">Auto-detect</option>
                        {languageOptions.map((lang) => (
                            <option key={lang.value} value={lang.value}>
                                {lang.label}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="glass-card p-4">
                    <label className="text-white text-sm font-semibold mb-2 block">Target Language</label>
                    <select
                        className="glass-select w-full"
                        value={targetLanguage}
                        onChange={(e) => setTargetLanguage(e.target.value)}
                        disabled={isTranslating}
                    >
                        {languageOptions.map((lang) => (
                            <option key={lang.value} value={lang.value}>
                                {lang.label}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Start Button */}
            <button
                onClick={handleTranslate}
                disabled={!selectedFile || isTranslating || !user}
                className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
            >
                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                    {isTranslating ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Translating Subtitles...</span>
                        </>
                    ) : (
                        <>
                            <Languages className="w-5 h-5" />
                            <span>{selectedFile ? 'Translate Subtitles' : 'Select a file to translate'}</span>
                        </>
                    )}
                </div>
            </button>

            {/* Info Panel */}
            <div className="glass-card p-4">
                <h4 className="text-white font-semibold mb-2">How it works:</h4>
                <ul className="text-slate-400 text-sm space-y-1">
                    <li>• Upload any SRT, VTT, ASS, or SSA subtitle file</li>
                    <li>• Select source and target languages</li>
                    <li>• AI translates the text while preserving timing and formatting</li>
                    <li>• Download the translated subtitle file instantly</li>
                    <li>• Cost: 5 credits per translation</li>
                </ul>
            </div>
        </div>
    );
}
