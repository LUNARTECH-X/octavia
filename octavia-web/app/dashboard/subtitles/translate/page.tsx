"use client";

import { motion } from "framer-motion";
import { FileText, Languages, Upload, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { useState, useCallback, useEffect } from "react";
import { useUser } from "@/contexts/UserContext";
import { useRouter } from "next/navigation";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/lib/api";

export default function SubtitleTranslatePage() {
    const { toast } = useToast();
    const { user, refreshCredits } = useUser();
    const router = useRouter();
    const [sourceLanguage, setSourceLanguage] = useState("en");
    const [targetLanguage, setTargetLanguage] = useState("es");
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isTranslating, setIsTranslating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const [jobProgress, setJobProgress] = useState<any>(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [projectId, setProjectId] = useState<string | null>(null);

    // Check for project context on mount
    useEffect(() => {
        // Attempt to recover active job from localStorage
        const savedJobId = localStorage.getItem('current_subtitle_translate_job');
        if (savedJobId) {
            setActiveJobId(savedJobId);
            setIsTranslating(true);
        }

        const projectContext = localStorage.getItem('octavia_project_context');
        if (projectContext) {
            try {
                const context = JSON.parse(projectContext);
                if (context.fileUrl && context.projectType === 'Subtitle Translation') {
                    console.log("Found project context, loading file:", context.fileName);

                    // Fetch the blob and create a File object
                    fetch(context.fileUrl)
                        .then(response => response.blob())
                        .then(blob => {
                            const projectFile = new File([blob], context.fileName, { type: context.fileType });
                            setSelectedFile(projectFile);
                            if (context.projectId) {
                                setProjectId(context.projectId);
                            }
                            setError(null);
                            setSuccess(null);

                            toast({
                                title: "File Loaded",
                                description: `Successfully loaded ${context.fileName} from project.`,
                            });

                            // Clear the project context after using it
                            localStorage.removeItem('octavia_project_context');
                        })
                        .catch(error => {
                            console.error('Failed to load file from project context:', error);
                            localStorage.removeItem('octavia_project_context');
                        });
                }
            } catch (error) {
                console.error('Failed to parse project context:', error);
                localStorage.removeItem('octavia_project_context');
            }
        }
    }, []);

    // Poll for job progress
    useEffect(() => {
        if (!activeJobId || !isTranslating) return;

        const pollProgress = async () => {
            try {
                const response = await api.getSubtitleJobStatus(activeJobId);

                if (response.success || (response as any).job_id) {
                    const data = response.data || (response as any);
                    setJobProgress(data);

                    if (data.progress !== undefined) {
                        setUploadProgress(data.progress);
                    }

                    if (data.status === 'completed' || data.status === 'failed') {
                        setIsTranslating(false);
                        localStorage.removeItem('current_subtitle_translate_job');
                        if (data.status === 'completed') {
                            setSuccess('Subtitle translation completed! Redirecting...');
                            setTimeout(() => {
                                router.push(`/dashboard/subtitles/review?jobId=${activeJobId}`);
                            }, 2000);
                        } else {
                            setError(data.error || 'Subtitle translation failed');
                        }
                    }
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        };

        const interval = setInterval(pollProgress, 2000);
        return () => clearInterval(interval);
    }, [activeJobId, isTranslating, router]);

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

    const handleDownload = async () => {
        if (!downloadUrl) return;

        setIsDownloading(true);
        try {
            // Extract file ID from download URL
            const fileIdMatch = downloadUrl.match(/\/download\/subtitles\/([^\/]+)/);
            if (!fileIdMatch) {
                setError('Invalid download URL');
                return;
            }

            const fileId = fileIdMatch[1];

            // Download the file using the API service - use full backend URL
            const fullUrl = `http://localhost:8000${downloadUrl}`;
            const blob = await api.downloadFileByUrl(fullUrl);
            api.saveFile(blob, `translated_subtitles_${fileId}.srt`);
        } catch (err) {
            console.error('Download error:', err);
            let errorMessage = 'Download failed';
            if (err instanceof Error) {
                errorMessage = err.message;
            }
            setError(errorMessage);
        } finally {
            setIsDownloading(false);
        }
    };

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
        setUploadProgress(10);

        try {
            // Start translation
            const result = await api.translateSubtitleFile({
                file: selectedFile,
                sourceLanguage,
                targetLanguage,
                projectId: projectId || undefined
            });

            if (result.success && result.job_id) {
                setUploadProgress(10); // Start at a small value, let polling take over
                setActiveJobId(result.job_id);
                localStorage.setItem('current_subtitle_translate_job', result.job_id);

                // Refresh credits to show updated balance
                await refreshCredits();

                // Add job to project in localStorage if projectId exists
                if (projectId) {
                    const storedJobs = localStorage.getItem(`octavia_project_jobs_${projectId}`);
                    const projectJobs = storedJobs ? JSON.parse(storedJobs) : [];

                    const newJob = {
                        id: result.job_id,
                        type: 'subtitle_translation',
                        status: 'processing',
                        progress: 0,
                        created_at: new Date().toISOString(),
                        source_language: sourceLanguage,
                        target_language: targetLanguage
                    };

                    if (!projectJobs.find((j: any) => j.id === result.job_id)) {
                        const updatedJobs = [newJob, ...projectJobs];
                        localStorage.setItem(`octavia_project_jobs_${projectId}`, JSON.stringify(updatedJobs));

                        window.dispatchEvent(new StorageEvent('storage', {
                            key: `octavia_project_jobs_${projectId}`,
                            newValue: JSON.stringify(updatedJobs)
                        }));
                    }
                }
            } else if (result.success && result.download_url) {
                // Fallback for immediate success
                setSuccess('Translation completed! You can now download your file.');
                setDownloadUrl(result.download_url);
                setIsTranslating(false);
            } else {
                setError(result.error || 'Translation failed');
                setDownloadUrl(null);
                setIsTranslating(false);
            }
        } catch (err) {
            console.error('Translation error:', err);
            // Better error handling for API response objects
            let errorMessage = 'An unexpected error occurred';
            if (err instanceof Error) {
                errorMessage = err.message;
            } else if (typeof err === 'object' && err !== null) {
                // Handle API response error objects
                const errorObj = err as any;
                if (errorObj.error) {
                    errorMessage = errorObj.error;
                } else if (errorObj.message) {
                    errorMessage = errorObj.message;
                } else if (errorObj.detail) {
                    errorMessage = errorObj.detail;
                } else {
                    errorMessage = `Translation failed: ${JSON.stringify(err)}`;
                }
            } else if (typeof err === 'string') {
                errorMessage = err;
            }
            setError(errorMessage);
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
        <div className="space-y-8 min-h-screen">
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
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle Translation</h1>
                    </div>
                </div>
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
                    className="glass-panel border-accent-cyan/30 bg-accent-cyan/10 p-4"
                >
                    <div className="flex items-center gap-3">
                        <CheckCircle2 className="w-5 h-5 text-accent-cyan flex-shrink-0" />
                        <p className="text-accent-cyan text-sm">{success}</p>
                    </div>
                    {downloadUrl && (
                        <button
                            type="button"
                            className="btn btn-primary mt-4"
                            onClick={handleDownload}
                            disabled={isDownloading}
                        >
                            {isDownloading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                                    Downloading...
                                </>
                            ) : (
                                'Download Translated Subtitles'
                            )}
                        </button>
                    )}
                </motion.div>
            )}

            {/* Upload Zone */}
            {!activeJobId && (
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
                                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-accent-cyan/10 border border-accent-cyan/30">
                                    <FileText className="w-8 h-8 text-accent-cyan" />
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
            )}

            {/* Configuration */}
            {selectedFile && !activeJobId && !isTranslating && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
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
                                    <span>In Progress...</span>
                                </>
                            ) : (
                                <>
                                    <Languages className="w-5 h-5" />
                                    <span>{selectedFile ? 'Translate Subtitles' : 'Select a file to translate'}</span>
                                </>
                            )}
                        </div>
                    </button>
                </motion.div>
            )}

            {/* Glass Loading Section */}
            {isTranslating && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="glass-panel p-12 flex flex-col items-center justify-center space-y-8 text-center"
                >
                    <div className="relative">
                        <div className="size-24 rounded-full border-4 border-primary-purple/20 border-t-primary-purple-bright animate-spin shadow-glow" />
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="size-16 rounded-full bg-primary-purple/10 flex items-center justify-center animate-pulse">
                                <Languages className="size-8 text-primary-purple-bright" />
                            </div>
                        </div>
                        <div className="glow-purple" style={{ width: "150px", height: "150px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", opacity: 0.2 }} />
                    </div>

                    <div className="space-y-3">
                        <h2 className="text-2xl font-black text-white text-glow-purple">
                            {jobProgress?.message || "Preparing Translation..."}
                        </h2>
                        <p className="text-slate-400 max-w-md mx-auto">
                            Our AI is translating your subtitles while preserving timing. You will get the results soon.
                        </p>
                    </div>

                    <div className="flex items-center gap-2 p-3 px-6 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-slate-400">
                        <span className="w-2 h-2 rounded-full bg-primary-purple-bright animate-pulse" />
                        {activeJobId ? `Job ID: ${activeJobId.substring(0, 12)}...` : "Preparing Upload..."}
                    </div>
                </motion.div>
            )}

            {/* Info Panel */}
            {!isTranslating && !selectedFile && (
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
            )}
        </div>
    );
}
