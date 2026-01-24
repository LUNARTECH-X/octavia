"use client";

import { useEffect, useState, useCallback } from "react";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";

interface ProgressTrackerProps {
    jobId: string;
    onComplete?: (jobId: string) => void;
    onError?: (error: string) => void;
    pollInterval?: number; // milliseconds, default 2000
}

interface ProgressData {
    progress: number;
    status: string;
    message: string;
    eta_seconds?: number;
    job_id: string;
}

export function ProgressTracker({
    jobId,
    onComplete,
    onError,
    pollInterval = 2000,
}: ProgressTrackerProps) {
    const [progress, setProgress] = useState<ProgressData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isPolling, setIsPolling] = useState(true);

    const fetchProgress = useCallback(async () => {
        try {
            // Get token from localStorage (matching the app's pattern)
            let token = localStorage.getItem("token");
            if (!token) {
                // Fallback: try getting from octavia_user object
                const userStr = localStorage.getItem('octavia_user');
                if (userStr) {
                    try {
                        const user = JSON.parse(userStr);
                        token = user.token;
                    } catch (e) {
                        console.error('Failed to parse user token:', e);
                    }
                }
            }

            if (!token) {
                throw new Error("Not authenticated");
            }

            const response = await fetch(
                `${process.env.NEXT_PUBLIC_API_URL}/api/progress/${jobId}`,
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    },
                }
            );

            if (!response.ok) {
                throw new Error(`Failed to fetch progress: ${response.statusText}`);
            }

            const data: ProgressData = await response.json();
            setProgress(data);

            // Stop polling if job is complete or failed
            if (data.status === "completed") {
                setIsPolling(false);
                onComplete?.(jobId);
            } else if (data.status === "failed") {
                setIsPolling(false);
                const errorMsg = data.message || "Translation failed";
                setError(errorMsg);
                onError?.(errorMsg);
            }
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : "Unknown error";
            setError(errorMsg);
            setIsPolling(false);
            onError?.(errorMsg);
        }
    }, [jobId, onComplete, onError]);

    useEffect(() => {
        if (!isPolling) return;

        // Initial fetch
        fetchProgress();

        // Set up polling
        const interval = setInterval(fetchProgress, pollInterval);

        // Cleanup
        return () => clearInterval(interval);
    }, [isPolling, fetchProgress, pollInterval]);

    if (error) {
        return (
            <div className="glass-panel border-red-500/30 bg-red-500/10 p-4">
                <div className="flex items-start gap-3">
                    <XCircle className="h-5 w-5 text-red-400 mt-0.5" />
                    <div className="flex-1">
                        <p className="text-red-400 text-sm">{error}</p>
                    </div>
                </div>
            </div>
        );
    }

    if (!progress) {
        return (
            <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Initializing...</span>
            </div>
        );
    }

    const formatETA = (seconds?: number) => {
        if (!seconds) return null;
        const mins = Math.ceil(seconds / 60);
        return mins === 1 ? "~1 minute" : `~${mins} minutes`;
    };

    return (
        <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                    {progress.status === "completed" ? (
                        <CheckCircle2 className="h-4 w-4 text-accent-cyan" />
                    ) : (
                        <Loader2 className="h-4 w-4 animate-spin text-primary-purple-bright" />
                    )}
                    <span className="font-medium text-white">
                        {progress.message || "Processing..."}
                    </span>
                </div>
                <div className="flex items-center gap-3 text-slate-400">
                    <span className="font-bold text-primary-purple-bright">{progress.progress}%</span>
                    {progress.eta_seconds && (
                        <span className="text-xs">{formatETA(progress.eta_seconds)}</span>
                    )}
                </div>
            </div>
            <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright transition-all duration-500 ease-out"
                    style={{ width: `${progress.progress}%` }}
                />
            </div>
        </div>
    );
}
