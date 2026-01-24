"use client";

import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, Loader2, Clock, AlertCircle, XCircle, Download, Terminal, ChevronDown } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";

interface JobStatus {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  status_message?: string;
  message?: string;
  language?: string;
  segment_count?: number;
  download_url?: string;
  error?: string;
  estimated_time?: string;
  format?: string;
  original_filename?: string;
  created_at?: string;
}

type PipelineStep = "initializing" | "extraction" | "transcription" | "formatting";

export default function SubtitleGenerationProgressPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("jobId") || localStorage.getItem("current_subtitle_job");

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [isLogsOpen, setIsLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setError("No job ID found. Please start a subtitle generation first.");
      setLoading(false);
      return;
    }

    const pollJobStatus = async () => {
      try {
        const response = await api.getSubtitleJobStatus(jobId);
        const jobData = response.data || (response as any);

        if (response.success || jobData.job_id) {
          const status = jobData.status;

          setJobStatus({
            job_id: jobData.job_id,
            status: status,
            progress: jobData.progress,
            status_message: jobData.status_message || jobData.message,
            message: jobData.message,
            language: jobData.language,
            segment_count: jobData.segment_count,
            download_url: jobData.download_url,
            error: jobData.error,
            format: jobData.format,
            created_at: jobData.created_at
          });

          // Add message to logs if it's new
          const currentMessage = jobData.status_message || jobData.message;
          if (currentMessage) {
            setLogs(prev => {
              const formattedLog = `[${new Date().toLocaleTimeString()}] ${currentMessage}`;
              if (prev.length === 0 || !prev[prev.length - 1].includes(currentMessage)) {
                return [...prev, formattedLog].slice(-20);
              }
              return prev;
            });
          }

          if (status === "completed" || status === "failed") {
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }

            if (status === "completed" && !isRedirecting) {
              setIsRedirecting(true);
              setTimeout(() => {
                router.push(`/dashboard/subtitles/review?jobId=${jobId}`);
              }, 3000);
            }
          }
        } else {
          setError(response.error || "Failed to fetch job status");
        }
      } catch (err) {
        setError("Network error while checking job status");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    pollJobStatus();
    const interval = setInterval(pollJobStatus, 2000);
    intervalRef.current = interval;

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [jobId]);

  const getActiveStep = (): PipelineStep => {
    if (!jobStatus) return "initializing";
    const prog = jobStatus.progress;
    if (prog < 20) return "initializing";
    if (prog < 40) return "extraction";
    if (prog < 85) return "transcription";
    return "formatting";
  };

  const getStepStatus = (step: PipelineStep): "completed" | "active" | "queued" => {
    const activeStep = getActiveStep();
    const steps: PipelineStep[] = ["initializing", "extraction", "transcription", "formatting"];
    const currentIndex = steps.indexOf(activeStep);
    const stepIndex = steps.indexOf(step);

    if (jobStatus?.status === "completed") return "completed";
    if (stepIndex < currentIndex) return "completed";
    if (stepIndex === currentIndex) return "active";
    return "queued";
  };

  const getStatusMessage = () => {
    if (jobStatus?.status === "completed") return "Subtitles generated successfully! Redirecting to review...";
    if (jobStatus?.status === "failed") return jobStatus.error || "Generation failed.";
    return jobStatus?.status_message || jobStatus?.message || "Processing your file...";
  };

  const renderPipelineStep = (step: PipelineStep, label: string) => {
    const status = getStepStatus(step);
    const isCompleted = status === "completed";
    const isActive = status === "active";

    return (
      <div className={`glass-card flex flex-col items-center gap-3 p-4 text-center ${isCompleted ? "border-accent-cyan/30 bg-accent-cyan/5" :
        isActive ? "glass-panel-glow ring-1 ring-primary-purple/50 relative overflow-hidden" :
          "opacity-50"
        }`}>
        {isActive && <div className="glass-shine" />}
        <div className={`flex size-10 items-center justify-center rounded-full ${isCompleted ? "bg-accent-cyan/20 text-accent-cyan" :
          isActive ? "bg-primary-purple/20 text-primary-purple-bright" :
            "bg-white/5 text-slate-500"
          } shadow-glow`}>
          {isCompleted ? <CheckCircle2 className="w-5 h-5" /> :
            isActive ? <Loader2 className="w-5 h-5 animate-spin" /> :
              <Clock className="w-5 h-5" />}
        </div>
        <div>
          <p className={`text-sm font-medium ${isActive ? "text-white text-glow-purple" : "text-white"}`}>{label}</p>
          <p className={`text-xs ${isCompleted ? "text-accent-cyan" : isActive ? "text-primary-purple-bright" : "text-slate-400"}`}>
            {isCompleted ? "Completed" : isActive ? "In Progress" : "Queued"}
          </p>
        </div>
      </div>
    );
  };

  if (error) {
    return (
      <div className="space-y-8 max-w-4xl mx-auto">
        <div className="flex flex-col gap-1 border-b border-white/10 pb-6">
          <h1 className="font-display text-3xl font-black text-white text-glow-purple">Error</h1>
        </div>
        <div className="glass-panel p-8 text-center">
          <XCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-white mb-2">Processing Error</h3>
          <p className="text-slate-400 mb-6">{error}</p>
          <button onClick={() => router.push("/dashboard/subtitles")} className="bg-primary-purple text-white px-6 py-2 rounded-lg">
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (loading && !jobStatus) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary-purple mx-auto mb-4" />
          <p className="text-white">Loading job status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
        <div className="flex flex-col gap-1">
          <h1 className="font-display text-3xl font-black text-white text-glow-purple">
            {jobStatus?.status === "completed" ? "✅ Generation Complete!" :
              jobStatus?.status === "failed" ? "❌ Generation Failed" :
                "Generating Subtitles: " + (jobStatus?.original_filename || "Processing...")}
          </h1>
          <p className="text-slate-400 text-sm">{getStatusMessage()}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          {/* Progress Card */}
          <div className="glass-panel p-6">
            <div className="flex items-center justify-between gap-6 mb-3">
              <p className="text-base font-medium text-white">Overall Progress</p>
              <p className="text-2xl font-bold text-primary-purple-bright text-glow-purple">
                {jobStatus?.progress || 0}%
              </p>
            </div>
            <div className="w-full bg-white/5 rounded-full h-2.5 mb-3 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${jobStatus?.progress || 0}%` }}
                transition={{ duration: 1 }}
                className={`h-2.5 rounded-full shadow-glow ${jobStatus?.status === "completed" ? "bg-accent-cyan" :
                  jobStatus?.status === "failed" ? "bg-red-500" : "bg-primary-purple"
                  }`}
              />
            </div>
            <p className="text-sm text-slate-400">{getStatusMessage()}</p>
          </div>

          {/* Pipeline */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Processing Pipeline</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {renderPipelineStep("initializing", "Initializing")}
              {renderPipelineStep("extraction", "Audio Extraction")}
              {renderPipelineStep("transcription", "Transcription")}
              {renderPipelineStep("formatting", "Finalizing")}
            </div>
          </div>
        </div>

        <div className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4">Job Info</h3>
            <div className="space-y-4">
              <div className="flex justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                <span className="text-slate-400 text-sm">Language</span>
                <span className="text-white text-sm font-bold">{jobStatus?.language || "Detecting..."}</span>
              </div>
              <div className="flex justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                <span className="text-slate-400 text-sm">Format</span>
                <span className="text-white text-sm font-bold">{jobStatus?.format?.toUpperCase() || "SRT"}</span>
              </div>
              {jobStatus?.segment_count && (
                <div className="flex justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                  <span className="text-slate-400 text-sm">Segments</span>
                  <span className="text-white text-sm font-bold">{jobStatus.segment_count}</span>
                </div>
              )}
            </div>
          </div>

          {/* Logs */}
          <div className="glass-panel overflow-hidden">
            <button
              onClick={() => setIsLogsOpen(!isLogsOpen)}
              className="w-full flex items-center justify-between p-4 font-medium text-white bg-white/5 hover:bg-white/10"
            >
              <span className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-slate-400" />
                Technical Logs
              </span>
              <ChevronDown className={`w-4 h-4 transition-transform ${isLogsOpen ? "rotate-180" : ""}`} />
            </button>
            {isLogsOpen && (
              <div className="h-64 overflow-y-auto p-4 font-mono text-[10px] space-y-1 bg-black/20 border-t border-white/5">
                {logs.length > 0 ? logs.map((log, i) => (
                  <p key={i} className="text-slate-400">{log}</p>
                )) : <p className="text-slate-600 italic">No logs yet...</p>}
              </div>
            )}
          </div>
        </div>
      </div>

      {!isRedirecting && jobStatus?.status === "processing" && (
        <div className="fixed bottom-4 right-4 flex items-center gap-2 px-3 py-2 bg-primary-purple/10 border border-primary-purple/30 rounded-lg">
          <div className="w-2 h-2 bg-primary-purple-bright rounded-full animate-pulse"></div>
          <span className="text-xs text-primary-purple-bright">Live Updates</span>
        </div>
      )}
    </div>
  );
}