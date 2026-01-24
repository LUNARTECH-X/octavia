"use client";

import { motion } from "framer-motion";
import { RefreshCw, Download, AlertCircle, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useToast } from "@/hooks/use-toast";


export default function VideoReviewPage() {
    const searchParams = useSearchParams();
    const jobId = searchParams.get('jobId');
    const { toast } = useToast();

    const [jobStatus, setJobStatus] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);


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

    const fetchJobStatus = async () => {
        if (!jobId) {
            setError("No job ID provided");
            setLoading(false);
            return;
        }

        const token = getToken();
        if (!token) {
            setError("Authentication required");
            setLoading(false);
            return;
        }

        // If we already have video or error, or are downloading, don't fetch
        if (videoUrl || error || isDownloading) {
            return;
        }

        try {
            // Try main status endpoint
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/jobs/${jobId}/status`, {
                headers: { 'Authorization': `Bearer ${token}` },
            });

            if (!response.ok) {
                // Try translation routes endpoint
                const altResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/translate/jobs/${jobId}/status`, {
                    headers: { 'Authorization': `Bearer ${token}` },
                });

                if (!altResponse.ok) {
                    throw new Error(`Failed to fetch job status: ${response.status}`);
                }

                const altData = await altResponse.json();
                setJobStatus(altData.data || altData);

                // Check if completed and try to download
                if ((altData.data?.status === 'completed' || altData.status === 'completed') && !videoUrl && !isDownloading) {
                    setIsDownloading(true);
                    await downloadVideo(jobId, token);
                }
                return;
            }

            const data = await response.json();
            console.log('Job status:', data);
            setJobStatus(data.data || data);

            // If job is completed, try to download video (only once)
            if ((data.data?.status === 'completed' || data.status === 'completed') && !videoUrl && !isDownloading) {
                setIsDownloading(true);
                await downloadVideo(jobId, token);
            }

        } catch (err: any) {
            console.error('Error:', err);
            setError(err.message || "Failed to load translation status");
        } finally {
            setLoading(false);
        }
    };

    const downloadVideo = async (jobId: string, token: string) => {
        console.log('Trying to download video for job:', jobId);

        // Try multiple endpoints
        const endpoints = [
            `/api/download/video/${jobId}`,
            `/api/download/${jobId}`,
        ];

        for (const endpoint of endpoints) {
            try {
                const url = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${endpoint}`;
                console.log('Trying:', url);

                const response = await fetch(url, {
                    headers: { 'Authorization': `Bearer ${token}` },
                });

                if (response.ok) {
                    const blob = await response.blob();
                    console.log('Success! Blob size:', blob.size);

                    if (blob.size > 0) {
                        const objectUrl = URL.createObjectURL(blob);
                        setVideoUrl(objectUrl);
                        console.log('Video loaded successfully');
                        return true;
                    }
                }
            } catch (err) {
                console.log(`Endpoint ${endpoint} failed:`, err);
            }
        }

        console.log('All download attempts failed');
        return false;
    };



    // Cleanup video URLs on unmount
    useEffect(() => {
        return () => {
            if (videoUrl && videoUrl.startsWith('blob:')) {
                URL.revokeObjectURL(videoUrl);
            }
        };
    }, [videoUrl]);

    // Poll for job status
    useEffect(() => {
        fetchJobStatus();

        // Only poll if job is not complete (no videoUrl, no error, and not downloading)
        if (!videoUrl && !error && !isDownloading) {
            // Poll every 5 seconds if not completed
            const interval = setInterval(() => {
                if (!videoUrl && !error && !isDownloading) {
                    fetchJobStatus();
                }
            }, 5000);

            return () => clearInterval(interval);
        }
    }, [jobId, videoUrl, error, isDownloading]);

    const handleDownload = () => {
        if (videoUrl) {
            const link = document.createElement('a');
            link.href = videoUrl;
            link.download = `translated_video_${jobId}.mp4`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            toast({
                title: "Download started",
                description: "Your video is being downloaded.",
            });
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[400px]">
                <div className="flex flex-col items-center gap-4">
                    <Loader2 className="w-8 h-8 animate-spin text-primary-purple" />
                    <p className="text-slate-400">Loading video translation...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="space-y-8">
                <div className="glass-panel border-red-500/30 bg-red-500/10 p-8 text-center">
                    <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                    <h2 className="text-xl font-bold text-white mb-2">Error</h2>
                    <p className="text-red-400 mb-4">{error}</p>
                    <button
                        onClick={() => window.history.back()}
                        className="px-4 py-2 bg-red-500/20 border border-red-500/30 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                    >
                        Go Back
                    </button>
                </div>
            </div>
        );
    }

    const isProcessing = jobStatus?.status === 'processing' || jobStatus?.status === 'pending';

    if (isProcessing) {
        return (
            <div className="space-y-8">
                <div className="glass-panel p-8 text-center">
                    <RefreshCw className="w-16 h-16 text-blue-400 mx-auto mb-4 animate-spin" />
                    <h2 className="text-xl font-bold text-white mb-2">Processing Video Translation</h2>
                    <p className="text-slate-400 mb-4">
                        {jobStatus?.progress
                            ? `Processing... ${jobStatus.progress}% complete`
                            : 'Your video is being translated...'
                        }
                    </p>
                    {jobStatus?.progress !== undefined && (
                        <div className="w-full max-w-md mx-auto">
                            <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright"
                                    initial={{ width: "0%" }}
                                    animate={{ width: `${jobStatus.progress}%` }}
                                    transition={{ duration: 0.5 }}
                                />
                            </div>
                            <p className="text-xs text-slate-500 mt-2">{jobStatus.progress}% complete</p>
                        </div>
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-8">
            <div className="flex flex-col lg:flex-row gap-8">
                {/* Video Player */}
                <div className="flex-1 flex flex-col gap-4">
                    {videoUrl ? (
                        <div className="glass-panel p-1">
                            <video
                                controls
                                autoPlay
                                className="w-full aspect-video rounded-lg bg-black"
                                src={videoUrl}
                            >
                                Your browser does not support the video tag.
                            </video>
                        </div>
                    ) : (
                        <div className="glass-panel p-1">
                            <div className="relative flex items-center justify-center bg-black bg-cover bg-center aspect-video rounded-lg overflow-hidden group">
                                <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 to-black" />
                                <div className="relative z-10 text-center p-8">
                                    <AlertCircle className="w-16 h-16 text-yellow-400 mx-auto mb-4" />
                                    <h2 className="text-white text-xl font-bold mb-2">Video Translation Complete</h2>
                                    <p className="text-slate-400 mb-4">Your video has been translated successfully.</p>
                                    <p className="text-slate-500 text-sm">Use the download button to get your video.</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Job Info */}
                    {jobStatus && (
                        <div className="glass-card p-4">
                            <h3 className="text-white font-bold mb-2">Translation Details</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-slate-400">Job ID:</span>
                                    <span className="text-white font-mono">{jobId}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-400">Status:</span>
                                    <span className="text-accent-cyan">Completed</span>
                                </div>
                                {jobStatus.target_language && (
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Language:</span>
                                        <span className="text-white">{jobStatus.target_language.toUpperCase()}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Column */}
                <div className="w-full lg:w-96 flex flex-col gap-6">
                    <div className="flex flex-col gap-2">
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">
                            {videoUrl ? 'Your Video is Ready!' : 'Translation Complete'}
                        </h1>
                        <p className="text-slate-400 text-sm">
                            {videoUrl
                                ? 'Watch your translated video below and download it.'
                                : 'Your video has been processed. Download it now.'}
                        </p>
                    </div>

                    {/* Download Section */}
                    <div className="glass-card p-6">
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center gap-3">
                                <div className="flex size-10 items-center justify-center rounded-full bg-accent-cyan/20 text-accent-cyan shadow-glow">
                                    <RefreshCw className="w-5 h-5" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Status</p>
                                    <p className="text-white text-lg font-bold">Completed</p>
                                </div>
                            </div>

                            <div className="pt-4 border-t border-white/10">
                                <button
                                    onClick={handleDownload}
                                    disabled={!videoUrl}
                                    className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3 text-base w-full">
                                        <Download className="w-5 h-5 text-primary-purple-bright group-hover:scale-110 transition-transform" />
                                        <span className="font-bold">Download MP4</span>
                                    </div>
                                </button>
                                {!videoUrl && (
                                    <p className="text-xs text-red-400 text-center mt-2">
                                        Video preview not available. Try downloading directly.
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Success Message */}
                    <div className="glass-card p-4 border-accent-cyan/30 bg-accent-cyan/10">
                        <div className="flex items-start gap-3">
                            <div className="mt-0.5">
                                <div className="w-5 h-5 rounded-full bg-accent-cyan/20 border border-accent-cyan/30 flex items-center justify-center">
                                    <span className="text-accent-cyan text-xs">✓</span>
                                </div>
                            </div>
                            <div className="flex-1">
                                <p className="text-accent-cyan text-sm font-medium">Translation Complete!</p>
                                <p className="text-slate-400 text-xs mt-1">
                                    Your video has been successfully translated.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}