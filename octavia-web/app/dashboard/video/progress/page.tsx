// app/dashboard/video/progress/page.tsx
"use client";

import { motion } from "framer-motion";
import { CheckCircle, Loader2, Clock, PlayCircle, ChevronDown, Terminal, AlertCircle, Download, Pause, Play, X, Volume2, VolumeX } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { AvailableChunk } from "@/lib/api";

type JobStatus = "pending" | "processing" | "completed" | "failed";
type PipelineStep = "splitting" | "transcribing" | "translating" | "dubbing" | "merging";

export default function TranslationProgressPage() {
    const [isLogsOpen, setIsLogsOpen] = useState(false);
    const [jobData, setJobData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState<string[]>([]);
    const [estimatedTime, setEstimatedTime] = useState("~12 minutes");
    const [availableChunks, setAvailableChunks] = useState<AvailableChunk[]>([]);
    const [playingChunk, setPlayingChunk] = useState<number | null>(null);
    const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

    const searchParams = useSearchParams();
    const router = useRouter();
    const jobId = searchParams.get("jobId");

    // Determine which step is active based on job data
    const getActiveStep = (): PipelineStep => {
        if (!jobData) return "splitting";

        const jobProgress = jobData.progress || 0;

        if (jobProgress < 20) return "splitting";
        if (jobProgress < 40) return "transcribing";
        if (jobProgress < 60) return "translating";
        if (jobProgress < 85) return "dubbing";
        return "merging";
    };

    const getStepStatus = (step: PipelineStep): "completed" | "active" | "queued" => {
        const activeStep = getActiveStep();
        const stepOrder: PipelineStep[] = ["splitting", "transcribing", "translating", "dubbing", "merging"];

        const currentIndex = stepOrder.indexOf(activeStep);
        const stepIndex = stepOrder.indexOf(step);

        if (stepIndex < currentIndex) return "completed";
        if (stepIndex === currentIndex) return "active";
        return "queued";
    };

    const fetchJobStatus = async () => {
        if (!jobId) return;

        try {
            console.log("Fetching job status for:", jobId);
            const response = await api.getJobStatus(jobId);
            console.log("API Response:", response);

            // Handle error responses first
            if (response.success === false && response.error) {
                console.error("API Error:", response.error);
                // If it's an auth error, we might want to handle it differently
                if (response.error.includes('Authentication')) {
                    setJobData({ status: "failed", error: "Authentication required. Please log in again." });
                    if (intervalRef.current) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                    }
                }
                return;
            }

            // Check if response has data property (ApiResponse structure) or direct data
            const jobDataResponse = response.data || response;
            console.log("Job data extracted:", jobDataResponse);

            // Validate we have usable job data
            const hasValidData = response.success ||
                jobDataResponse.job_id ||
                jobDataResponse.status ||
                typeof jobDataResponse.progress === 'number';

            if (hasValidData) {
                setJobData(jobDataResponse);
                setProgress(jobDataResponse.progress || 0);
                console.log("Updated jobData:", jobDataResponse);
                console.log("Updated progress:", jobDataResponse.progress || 0);

                // Update logs if available
                if ((jobDataResponse as any).logs) {
                    const formattedLogs = (jobDataResponse as any).logs.map((log: any) =>
                        `[${new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}] - ${log.message}`
                    );
                    setLogs(formattedLogs.slice(-20)); // Last 20 logs
                }

                // Update estimated time based on chunks
                if ((jobDataResponse as any).total_chunks && (jobDataResponse as any).processed_chunks) {
                    const remainingChunks = (jobDataResponse as any).total_chunks - (jobDataResponse as any).processed_chunks;
                    const estimatedMinutes = Math.ceil(remainingChunks * 0.5); // 30 seconds per chunk
                    setEstimatedTime(`~${estimatedMinutes} minute${estimatedMinutes !== 1 ? 's' : ''}`);
                }

                // Update available chunks for preview
                if ((jobDataResponse as any).available_chunks) {
                    setAvailableChunks((jobDataResponse as any).available_chunks);
                }

                // If job is completed, stop polling and redirect to review page after delay
                if (jobDataResponse.status === "completed") {
                    console.log("Job completed, stopping polling and redirecting...");
                    if (intervalRef.current) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                    }
                    setTimeout(() => {
                        router.push(`/dashboard/video/review?jobId=${jobId}`);
                    }, 3000);
                } else if (jobDataResponse.status === "failed") {
                    // If job failed, stop polling
                    console.log("Job failed, stopping polling");
                    if (intervalRef.current) {
                        clearInterval(intervalRef.current);
                        intervalRef.current = null;
                    }
                }
            } else {
                console.warn("Invalid or empty response structure:", response);
                // Don't stop polling for invalid responses - might be temporary
            }
        } catch (error) {
            console.error("Error fetching job status:", error);
        } finally {
            setLoading(false);
        }
    };

    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        if (jobId) {
            // Initial fetch
            fetchJobStatus();

            // Set up polling every 5 seconds for updates
            const interval = setInterval(fetchJobStatus, 5000);
            intervalRef.current = interval;

            return () => {
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                }
                // Cleanup audio on unmount
                if (audioElement) {
                    audioElement.pause();
                    audioElement.src = '';
                }
            };
        }
    }, [jobId]);

    // Cleanup audio when component unmounts
    useEffect(() => {
        return () => {
            if (audioElement) {
                audioElement.pause();
                audioElement.src = '';
            }
        };
    }, [audioElement]);

    const handleCancelJob = async () => {
        if (!jobId) return;

        if (confirm("Are you sure you want to cancel this job?")) {
            // In a real app, you would call a cancel endpoint
            // For now, just stop polling and show cancelled state
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
            setJobData({ ...jobData, status: "cancelled" });
        }
    };

    const handlePlayChunk = async (chunk: AvailableChunk) => {
        try {
            // Stop any currently playing audio
            if (audioElement) {
                audioElement.pause();
                audioElement.currentTime = 0;
            }

            // If clicking the same chunk, stop playback
            if (playingChunk === chunk.id) {
                setPlayingChunk(null);
                return;
            }

            // Download and play the chunk
            setPlayingChunk(chunk.id);
            const blob = await api.downloadChunk(jobId!, chunk.id);
            const audioUrl = URL.createObjectURL(blob);

            const audio = new Audio(audioUrl);
            setAudioElement(audio);

            audio.onended = () => {
                setPlayingChunk(null);
                URL.revokeObjectURL(audioUrl);
            };

            audio.onerror = () => {
                setPlayingChunk(null);
                URL.revokeObjectURL(audioUrl);
                alert("Failed to play chunk");
            };

            await audio.play();
        } catch (error) {
            console.error("Error playing chunk:", error);
            setPlayingChunk(null);
            alert("Failed to play chunk. It may not be ready yet.");
        }
    };

    const handleDownloadSample = async () => {
        if (!jobId) return;

        try {
            // Request a 10-30s sample chunk from the backend
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/jobs/${jobId}/sample`, {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to get sample: ${response.status}`);
            }

            const blob = await response.blob();
            if (blob.size === 0) {
                throw new Error("Sample not available yet");
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sample_10-30s_${jobId}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error("Sample download error:", error);
            alert("Unable to download sample. Processing might not be far enough along yet.");
        }
    };

    // Get authentication token
    const getToken = (): string | null => {
        if (typeof window === 'undefined') return null;
        const userStr = localStorage.getItem('octavia_user');
        if (userStr) {
            try {
                const user = JSON.parse(userStr);
                return user.token || null;
            } catch (error) {
                console.error('Failed to parse user token:', error);
                return null;
            }
        }
        return null;
    };

    const getStatusMessage = () => {
        if (!jobData) return "Loading...";

        switch (jobData.status) {
            case "processing":
                if (jobData.status_message) return jobData.status_message;
                return "Real-time AI orchestration in progress. You can monitor the status below.";
            case "completed":
                return "Translation completed! Redirecting to review...";
            case "failed":
                return `Job failed: ${jobData.error || "Unknown error"}`;
            case "cancelled":
                return "Job cancelled by user.";
            default:
                return "Job is being processed...";
        }
    };

    const renderPipelineStep = (step: PipelineStep, label: string) => {
        const status = getStepStatus(step);
        const isCompleted = status === "completed";
        const isActive = status === "active";

        return (
            <div className={`glass-card flex flex-col items-center gap-3 p-4 text-center ${isCompleted
                ? "border-accent-cyan/30 bg-accent-cyan/5"
                : isActive
                    ? "glass-panel-glow ring-1 ring-primary-purple/50 relative overflow-hidden"
                    : "opacity-50"
                }`}>
                {isActive && <div className="glass-shine" />}

                <div className="relative z-10 flex flex-col items-center gap-3">
                    <div className={`flex size-10 items-center justify-center rounded-full ${isCompleted
                        ? "bg-accent-cyan/20 text-accent-cyan"
                        : isActive
                            ? "bg-primary-purple/20 text-primary-purple-bright"
                            : "bg-white/5 text-slate-500"
                        } shadow-glow`}>
                        {isCompleted ? (
                            <CheckCircle className="w-5 h-5" />
                        ) : isActive ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Clock className="w-5 h-5" />
                        )}
                    </div>
                    <div>
                        <p className={`text-sm font-medium ${isActive ? "font-bold text-white text-glow-purple" : "text-white"
                            }`}>
                            {label}
                        </p>
                        <p className={`text-xs ${isCompleted
                            ? "text-accent-cyan"
                            : isActive
                                ? "text-primary-purple-bright"
                                : "text-slate-400"
                            }`}>
                            {status === "completed" ? "Completed" : status === "active" ? "In Progress" : "Queued"}
                        </p>
                    </div>
                </div>
            </div>
        );
    };

    if (loading && !jobData) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-center">
                    <Loader2 className="w-12 h-12 animate-spin text-primary-purple mx-auto mb-4" />
                    <p className="text-white">Loading job status...</p>
                </div>
            </div>
        );
    }

    if (!jobId) {
        return (
            <div className="text-center py-12">
                <AlertCircle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
                <h2 className="text-2xl font-bold text-white mb-4">No Job ID Provided</h2>
                <p className="text-gray-400 mb-6">Please start a translation job first.</p>
                <button
                    onClick={() => router.push('/dashboard/video')}
                    className="bg-primary-purple text-white px-6 py-3 rounded-lg hover:opacity-90 transition-opacity"
                >
                    Go to Video Upload
                </button>
            </div>
        );
    }

    const filename = jobData?.original_filename || "Video_File_Name.mp4";
    const status = jobData?.status || "processing";
    const isCompleted = status === "completed";
    const isFailed = status === "failed";
    const isCancelled = status === "cancelled";

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
                <div className="flex flex-col gap-1">
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple">
                        {isCompleted ? "✅ Translation Complete!" :
                            isFailed ? "❌ Translation Failed" :
                                isCancelled ? "⏹️ Translation Cancelled" :
                                    "Translating: " + filename}
                    </h1>
                    <p className="text-slate-400 text-sm">{getStatusMessage()}</p>
                </div>

                {!isCompleted && !isFailed && !isCancelled && (
                    <div className="flex items-center gap-3">
                        <button className="flex h-10 items-center justify-center rounded-lg bg-white/5 border border-white/10 px-4 text-sm font-bold text-white hover:bg-white/10 transition-colors">
                            <Pause className="w-4 h-4 mr-2" />
                            Pause
                        </button>
                        <button className="flex h-10 items-center justify-center rounded-lg bg-white/5 border border-white/10 px-4 text-sm font-bold text-white hover:bg-white/10 transition-colors">
                            <Play className="w-4 h-4 mr-2" />
                            Resume
                        </button>
                        <button
                            onClick={handleCancelJob}
                            className="flex h-10 items-center justify-center rounded-lg bg-red-500/10 border border-red-500/30 px-4 text-sm font-bold text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-colors"
                        >
                            <X className="w-4 h-4 mr-2" />
                            Cancel
                        </button>
                    </div>
                )}

                {isCompleted && jobData?.download_url && (
                    <button
                        onClick={handleDownloadSample}
                        className="flex h-10 items-center justify-center rounded-lg bg-accent-cyan/10 border border-accent-cyan/30 px-4 text-sm font-bold text-accent-cyan hover:bg-accent-cyan/20 hover:border-accent-cyan/50 transition-colors"
                    >
                        <Download className="w-4 h-4 mr-2" />
                        Download Result
                    </button>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 flex flex-col gap-8">
                    {/* Overall Progress */}
                    <div className="glass-panel p-6">
                        <div className="flex items-center justify-between gap-6 mb-3">
                            <p className="text-base font-medium text-white">Overall Progress</p>
                            <p className="text-2xl font-bold text-primary-purple-bright text-glow-purple">
                                {isCompleted ? "100%" : `${progress}%`}
                            </p>
                        </div>
                        <div className="w-full bg-white/5 rounded-full h-2.5 mb-3 overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${isCompleted ? 100 : progress}%` }}
                                transition={{ duration: 1, ease: "easeOut" }}
                                className={`h-2.5 rounded-full shadow-glow ${isCompleted
                                    ? "bg-accent-cyan"
                                    : isFailed || isCancelled
                                        ? "bg-red-500"
                                        : "bg-primary-purple"
                                    }`}
                            />
                        </div>
                        <p className="text-sm text-slate-400">
                            {isCompleted
                                ? "Translation completed successfully!"
                                : isFailed
                                    ? `Failed: ${jobData?.error || "Unknown error"}`
                                    : isCancelled
                                        ? "Job cancelled by user."
                                        : jobData?.status_message || `Processing... ${jobData?.processed_chunks || 0}/${jobData?.total_chunks || 0} chunks`}
                        </p>

                        {/* Overall Quality Metrics */}
                        {availableChunks.length > 0 && (
                            <div className="mt-4 p-3 bg-white/5 rounded-lg border border-white/10">
                                <div className="flex items-center gap-4 text-sm">
                                    <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse"></div>
                                        <span className="text-slate-400">Quality Score:</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        {(() => {
                                            const avgConfidence = availableChunks
                                                .filter(c => c.confidence_score !== undefined)
                                                .reduce((sum, c) => sum + (c.confidence_score || 0), 0) / Math.max(1, availableChunks.filter(c => c.confidence_score !== undefined).length);

                                            const qualityRating = avgConfidence >= 0.8 ? 'Excellent' :
                                                avgConfidence >= 0.6 ? 'Good' :
                                                    avgConfidence >= 0.4 ? 'Fair' : 'Needs Review';

                                            return (
                                                <>
                                                    <span className={`font-bold ${avgConfidence >= 0.8 ? 'text-accent-cyan' :
                                                        avgConfidence >= 0.6 ? 'text-yellow-400' :
                                                            'text-red-400'
                                                        }`}>
                                                        {(avgConfidence * 100).toFixed(0)}%
                                                    </span>
                                                    <span className={`px-2 py-0.5 rounded text-xs ${qualityRating === 'Excellent' ? 'bg-accent-cyan/20 text-accent-cyan' :
                                                        qualityRating === 'Good' ? 'bg-blue-500/20 text-blue-400' :
                                                            qualityRating === 'Fair' ? 'bg-yellow-500/20 text-yellow-400' :
                                                                'bg-red-500/20 text-red-400'
                                                        }`}>
                                                        {qualityRating}
                                                    </span>
                                                </>
                                            );
                                        })()}
                                    </div>
                                </div>
                                <p className="text-xs text-slate-500 mt-2">
                                    Based on {availableChunks.length} processed chunks.
                                    Higher scores indicate better transcription and translation quality.
                                </p>
                            </div>
                        )}

                        {jobData?.chunk_size && (
                            <div className="mt-4 p-3 bg-white/5 rounded-lg">
                                <p className="text-xs text-slate-400">
                                    Chunk Size: {jobData.chunk_size}s |
                                    Total Chunks: {jobData.total_chunks || "Calculating..."} |
                                    Processed: {jobData.processed_chunks || 0}
                                </p>
                            </div>
                        )}
                    </div>

                    {/* Translation Pipeline */}
                    <div>
                        <h2 className="text-xl font-bold text-white mb-4">Translation Pipeline</h2>
                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
                            {renderPipelineStep("splitting", "Splitting")}
                            {renderPipelineStep("transcribing", "Transcribing")}
                            {renderPipelineStep("translating", "Translating")}
                            {renderPipelineStep("dubbing", "Dubbing")}
                            {renderPipelineStep("merging", "Merging")}
                        </div>
                    </div>

                    {/* Chunk Preview */}
                    {availableChunks.length > 0 && (
                        <div>
                            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                <Volume2 className="w-5 h-5 text-accent-cyan" />
                                Preview Translated Chunks
                            </h2>
                            <p className="text-slate-400 text-sm mb-4">
                                Listen to completed translated chunks before the full video is ready
                            </p>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                                {availableChunks.map((chunk) => (
                                    <motion.div
                                        key={chunk.id}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="glass-card p-4 hover:bg-white/10 transition-colors"
                                    >
                                        <div className="flex items-center justify-between mb-3">
                                            <div>
                                                <p className="text-white font-medium">Chunk {chunk.id + 1}</p>
                                                <p className="text-slate-400 text-xs">
                                                    {Math.floor(chunk.start_time / 60)}:{String(Math.floor(chunk.start_time % 60)).padStart(2, '0')} •
                                                    {chunk.duration.toFixed(1)}s
                                                </p>
                                                {chunk.confidence_score !== undefined && (
                                                    <div className="flex items-center gap-2 mt-1">
                                                        <div className="text-xs text-slate-400">Quality:</div>
                                                        <div className="flex items-center gap-1">
                                                            <div className={`w-2 h-2 rounded-full ${chunk.confidence_score >= 0.8 ? 'bg-accent-cyan' :
                                                                chunk.confidence_score >= 0.6 ? 'bg-yellow-500' :
                                                                    'bg-red-500'
                                                                }`}></div>
                                                            <span className={`text-xs font-medium ${chunk.confidence_score >= 0.8 ? 'text-accent-cyan' :
                                                                chunk.confidence_score >= 0.6 ? 'text-yellow-400' :
                                                                    'text-red-400'
                                                                }`}>
                                                                {(chunk.confidence_score * 100).toFixed(0)}%
                                                            </span>
                                                            {chunk.quality_rating && (
                                                                <span className={`text-xs px-1 py-0.5 rounded ${chunk.quality_rating === 'excellent' ? 'bg-accent-cyan/20 text-accent-cyan' :
                                                                    chunk.quality_rating === 'good' ? 'bg-blue-500/20 text-blue-400' :
                                                                        chunk.quality_rating === 'fair' ? 'bg-yellow-500/20 text-yellow-400' :
                                                                            'bg-red-500/20 text-red-400'
                                                                    }`}>
                                                                    {chunk.quality_rating}
                                                                </span>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <div className="w-2 h-2 rounded-full bg-accent-cyan"></div>
                                                <span className="text-accent-cyan text-xs">Ready</span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handlePlayChunk(chunk)}
                                            className={`w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg font-medium transition-all ${playingChunk === chunk.id
                                                ? 'bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30'
                                                : 'bg-accent-cyan/10 border border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/20'
                                                }`}
                                        >
                                            {playingChunk === chunk.id ? (
                                                <>
                                                    <VolumeX className="w-4 h-4" />
                                                    Stop
                                                </>
                                            ) : (
                                                <>
                                                    <PlayCircle className="w-4 h-4" />
                                                    Play Preview
                                                </>
                                            )}
                                        </button>
                                    </motion.div>
                                ))}
                            </div>
                            <div className="mt-4 space-y-3">
                                <div className="p-3 bg-accent-cyan/10 border border-accent-cyan/30 rounded-lg">
                                    <p className="text-accent-cyan text-sm">
                                        💡 <strong>Pro tip:</strong> These chunks show you how your translated audio will sound.
                                        Each chunk represents a continuous segment of speech that was transcribed and translated.
                                    </p>
                                </div>

                                {/* Quality Legend */}
                                <div className="p-3 bg-white/5 border border-white/10 rounded-lg">
                                    <h4 className="text-white text-sm font-semibold mb-2">Quality Indicators:</h4>
                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full bg-accent-cyan"></div>
                                            <span className="text-slate-300">Excellent (80%+)</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                                            <span className="text-slate-300">Good (60-79%)</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                            <span className="text-slate-300">Needs Review (&lt;60%)</span>
                                        </div>
                                    </div>
                                    <p className="text-slate-500 text-xs mt-2">
                                        Confidence scores are based on Whisper AI's transcription accuracy.
                                        Lower scores may indicate background noise, unclear speech, or complex audio.
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="lg:col-span-1 flex flex-col gap-6">
                    {/* Status Overview */}
                    <div className="glass-panel p-6">
                        <h3 className="text-lg font-bold text-white mb-4">Status Overview</h3>
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Job ID</p>
                                <p className="text-sm font-mono text-white">{jobId}</p>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Estimated Time</p>
                                <p className="text-sm font-bold text-white">
                                    {isCompleted ? "Complete" : isFailed ? "Failed" : isCancelled ? "Cancelled" : estimatedTime}
                                </p>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                <p className="text-sm text-slate-400">Target Language</p>
                                <p className="text-sm font-bold text-white">{jobData?.target_language || "es"}</p>
                            </div>

                            {availableChunks.length > 0 && (
                                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                    <p className="text-sm text-slate-400">Translation Quality</p>
                                    <div className="flex items-center gap-2">
                                        {(() => {
                                            const avgConfidence = availableChunks
                                                .filter(c => c.confidence_score !== undefined)
                                                .reduce((sum, c) => sum + (c.confidence_score || 0), 0) / Math.max(1, availableChunks.filter(c => c.confidence_score !== undefined).length);

                                            return (
                                                <>
                                                    <div className={`w-3 h-3 rounded-full ${avgConfidence >= 0.8 ? 'bg-accent-cyan' :
                                                        avgConfidence >= 0.6 ? 'bg-yellow-500' :
                                                            'bg-red-500'
                                                        }`}></div>
                                                    <span className={`text-sm font-bold ${avgConfidence >= 0.8 ? 'text-accent-cyan' :
                                                        avgConfidence >= 0.6 ? 'text-yellow-400' :
                                                            'text-red-400'
                                                        }`}>
                                                        {(avgConfidence * 100).toFixed(0)}%
                                                    </span>
                                                </>
                                            );
                                        })()}
                                    </div>
                                </div>
                            )}

                            {!isCompleted && !isFailed && !isCancelled && (
                                <button
                                    onClick={handleDownloadSample}
                                    disabled={progress < 50}
                                    className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5">
                                        <PlayCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                                        <span>Play Sample Chunk</span>
                                    </div>
                                </button>
                            )}

                            {isFailed && (
                                <button
                                    onClick={() => router.push('/dashboard/video')}
                                    className="w-full py-2.5 rounded-lg bg-primary-purple/10 border border-primary-purple/30 text-primary-purple-bright font-bold hover:bg-primary-purple/20 hover:border-primary-purple/50 transition-all"
                                >
                                    Try Again
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Logs */}
                    <div className="glass-panel overflow-hidden">
                        <button
                            onClick={() => setIsLogsOpen(!isLogsOpen)}
                            className="w-full flex items-center justify-between p-4 font-medium text-white bg-white/5 hover:bg-white/10 transition-colors"
                        >
                            <span className="flex items-center gap-2">
                                <Terminal className="w-4 h-4 text-slate-400" />
                                View Technical Logs
                            </span>
                            <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isLogsOpen ? "rotate-180" : ""}`} />
                        </button>

                        {isLogsOpen && (
                            <div className="h-80 overflow-y-auto p-4 font-mono text-xs space-y-1 custom-scrollbar bg-black/20 border-t border-white/5">
                                {logs.length > 0 ? (
                                    logs.map((log, index) => {
                                        // Simple log coloring
                                        let textColor = "text-slate-300";
                                        if (log.includes("successful") || log.includes("completed")) textColor = "text-accent-cyan";
                                        if (log.includes("error") || log.includes("failed") || log.includes("Error")) textColor = "text-red-400";
                                        if (log.includes("warning") || log.includes("Warning")) textColor = "text-yellow-400";
                                        if (log.includes("processing") || log.includes("Processing") || log.includes("progress")) textColor = "text-primary-purple-bright";

                                        return (
                                            <p key={index} className="text-slate-500">
                                                {log.split(" - ")[0]} - <span className={textColor}>{log.split(" - ")[1]}</span>
                                            </p>
                                        );
                                    })
                                ) : (
                                    <>
                                        <p className="text-slate-500">[14:32:01] - <span className="text-slate-300">Starting translation job for {filename}...</span></p>
                                        <p className="text-slate-500">[14:32:02] - <span className="text-slate-300">Video uploaded successfully.</span></p>
                                        <p className="text-slate-500">[14:32:05] - <span className="text-primary-purple-bright">Extracting audio from video...</span></p>
                                        <p className="text-slate-500">[14:32:08] - <span className="text-primary-purple-bright">Transcribing audio with Whisper AI...</span></p>
                                        <p className="text-slate-500">[14:32:12] - <span className="text-primary-purple-bright">Translating text with Helsinki NLP...</span></p>
                                        <p className="text-slate-500">[...]</p>
                                        <p className="text-slate-500">[14:38:35] - <span className="text-primary-purple-bright animate-pulse">Processing in real-time...</span></p>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Real-time updates indicator */}
            {!isCompleted && !isFailed && !isCancelled && (
                <div className="fixed bottom-4 right-4 flex items-center gap-2 px-3 py-2 bg-primary-purple/10 border border-primary-purple/30 rounded-lg">
                    <div className="w-2 h-2 bg-primary-purple-bright rounded-full animate-pulse"></div>
                    <span className="text-xs text-primary-purple-bright">Live Updates</span>
                </div>
            )}
        </div>
    );
}
