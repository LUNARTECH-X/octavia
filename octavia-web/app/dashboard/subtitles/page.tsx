"use client";

import { motion } from "framer-motion";
import { FileVideo, Captions, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { useState, useCallback, useEffect } from "react";
import { useUser } from "@/contexts/UserContext";
import { api, safeApiResponse, isSuccess } from "@/lib/api";
import { useRouter } from 'next/navigation';
import { useToast } from "@/hooks/use-toast";

export default function SubtitleGenerationPage() {
  const { toast } = useToast();
  const router = useRouter();
  const { user, refreshCredits } = useUser();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("en");
  const [format, setFormat] = useState("srt");
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<any>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [projectId, setProjectId] = useState<string | null>(null);

  // Check for project context on mount
  useEffect(() => {
    // Attempt to recover active job from localStorage
    const savedJobId = localStorage.getItem('current_subtitle_job');
    if (savedJobId) {
      setActiveJobId(savedJobId);
      setIsUploading(true);
    }

    const projectContext = localStorage.getItem('octavia_project_context');
    if (projectContext) {
      try {
        const context = JSON.parse(projectContext);
        if (context.fileUrl && context.projectType === 'Subtitle Generation') {
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
    if (!activeJobId || !isUploading) return;

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
            setIsUploading(false);
            localStorage.removeItem('current_subtitle_job');
            if (data.status === 'completed') {
              setSuccess('Subtitle generation completed! Redirecting...');
              setTimeout(() => {
                router.push(`/dashboard/subtitles/review?jobId=${activeJobId}`);
              }, 2000);
            } else {
              setError(data.error || 'Subtitle generation failed');
            }
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    const interval = setInterval(pollProgress, 2000);
    return () => clearInterval(interval);
  }, [activeJobId, isUploading, router]);

  const handleFileSelect = useCallback((file: File) => {
    // Check file type
    if (!file.type.startsWith('video/') && !file.type.startsWith('audio/')) {
      setError('Please select a valid video or audio file');
      return;
    }

    // Check file size (max 500MB for videos, 100MB for audio)
    const maxSize = file.type.startsWith('video/') ? 500 * 1024 * 1024 : 100 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(`File size must be less than ${file.type.startsWith('video/') ? '500MB' : '100MB'}`);
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

  const handleGenerateSubtitles = async () => {
    if (!selectedFile || !user) {
      setError('Please select a file and ensure you are logged in');
      return;
    }

    setIsUploading(true);
    setError(null);
    setSuccess(null);
    setUploadProgress(10);

    try {
      const credits = user?.credits || 0;

      if (credits < 1) {
        // Try to add test credits automatically
        const addCreditsResponse = await api.addTestCredits(10);
        if (!addCreditsResponse.success) {
          setError('Insufficient credits. Unable to add test credits automatically.');
          setIsUploading(false);
          return;
        }

        // Refresh credits after adding
        await refreshCredits();
      }

      const response = await api.generateSubtitles(selectedFile, format, language, projectId || undefined);
      const resData = response.data;

      if (response.success && resData && resData.job_id) {
        setUploadProgress(10); // Start at a small value, let polling take over
        setActiveJobId(resData.job_id);
        localStorage.setItem('current_subtitle_job', resData.job_id);

        // Refresh credits to show updated balance
        await refreshCredits();

        // Add job to project in localStorage if projectId exists
        if (projectId) {
          const storedJobs = localStorage.getItem(`octavia_project_jobs_${projectId}`);
          const projectJobs = storedJobs ? JSON.parse(storedJobs) : [];

          const newJob = {
            id: resData.job_id,
            type: 'subtitles',
            status: 'processing',
            progress: 0,
            created_at: new Date().toISOString(),
            language: language,
            format: format
          };

          if (!projectJobs.find((j: any) => j.id === resData.job_id)) {
            const updatedJobs = [newJob, ...projectJobs];
            localStorage.setItem(`octavia_project_jobs_${projectId}`, JSON.stringify(updatedJobs));

            window.dispatchEvent(new StorageEvent('storage', {
              key: `octavia_project_jobs_${projectId}`,
              newValue: JSON.stringify(updatedJobs)
            }));
          }
        }
      } else {
        setError(response.error || 'Failed to start subtitle generation');
        setIsUploading(false);
      }
    } catch (err) {
      console.error('Subtitle generation error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      setIsUploading(false);
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

  const formatOptions = [
    { value: 'srt', label: 'SRT (SubRip)' },
    { value: 'vtt', label: 'VTT (WebVTT)' },
    { value: 'ass', label: 'ASS (Advanced SubStation)' },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle Generation</h1>
        <p className="text-slate-400 text-sm">Generate accurate subtitles from your video or audio files using AI</p>
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
          onClick={() => !selectedFile && document.getElementById('media-upload')?.click()}
        >
          <div className="glass-shine" />
          <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

          <div className="relative z-20 py-12 px-6">
            {selectedFile ? (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-accent-cyan/10 border border-accent-cyan/30">
                  <FileVideo className="w-8 h-8 text-accent-cyan" />
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-green">{selectedFile.name}</h3>
                  <p className="text-slate-400 text-sm">
                    {(selectedFile.size / (1024 * 1024)).toFixed(1)} MB • {selectedFile.type.split('/')[0].toUpperCase()} file
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
                  {isUploading ? <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" /> : <FileVideo className="w-8 h-8 text-primary-purple-bright" />}
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                    {isDragging ? 'Drop your media file here' : 'Drop your media file here'}
                  </h3>
                  <p className="text-slate-400 text-sm">or click to browse files • Video or Audio supported • Max 500MB</p>
                </div>
              </div>
            )}
          </div>

          <input
            id="media-upload"
            type="file"
            accept="video/*,audio/*"
            onChange={handleFileInput}
            className="hidden"
          />
        </motion.div>
      )}

      {/* Configuration */}
      {selectedFile && !activeJobId && !isUploading && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="glass-card p-4">
              <label className="text-white text-sm font-semibold mb-2 block">Audio Language</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="glass-select w-full"
                disabled={isUploading}
              >
                {languageOptions.map((lang) => (
                  <option key={lang.value} value={lang.value}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="glass-card p-4">
              <label className="text-white text-sm font-semibold mb-2 block">Subtitle Format</label>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                className="glass-select w-full"
                disabled={isUploading}
              >
                {formatOptions.map((fmt) => (
                  <option key={fmt.value} value={fmt.value}>
                    {fmt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Start Button */}
          <button
            onClick={handleGenerateSubtitles}
            disabled={!selectedFile || isUploading || !user}
            className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
              {isUploading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>In Progress...</span>
                </>
              ) : (
                <>
                  <Captions className="w-5 h-5" />
                  <span>{selectedFile ? 'Generate Subtitles' : 'Select a file to continue'}</span>
                </>
              )}
            </div>
          </button>
        </motion.div>
      )}

      {/* Glass Loading Section */}
      {isUploading && (
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel p-12 flex flex-col items-center justify-center space-y-8 text-center"
        >
          <div className="relative">
            <div className="size-24 rounded-full border-4 border-primary-purple/20 border-t-primary-purple-bright animate-spin shadow-glow" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="size-16 rounded-full bg-primary-purple/10 flex items-center justify-center animate-pulse">
                <Captions className="size-8 text-primary-purple-bright" />
              </div>
            </div>
            <div className="glow-purple" style={{ width: "150px", height: "150px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", opacity: 0.2 }} />
          </div>

          <div className="space-y-3">
            <h2 className="text-2xl font-black text-white text-glow-purple">
              {jobProgress?.message || "Analyzing Media..."}
            </h2>
            <p className="text-slate-400 max-w-md mx-auto">
              Our AI is transcribing and formatting your subtitles. You will get the subtitles soon.
            </p>
          </div>

          <div className="flex items-center gap-2 p-3 px-6 rounded-full bg-white/5 border border-white/10 text-xs font-mono text-slate-400">
            <span className="w-2 h-2 rounded-full bg-primary-purple-bright animate-pulse" />
            {activeJobId ? `Job ID: ${activeJobId.substring(0, 12)}...` : "Preparing Upload..."}
          </div>
        </motion.div>
      )}

      {/* Info Panel */}
      {!isUploading && !selectedFile && (
        <div className="glass-card p-4">
          <h4 className="text-white font-semibold mb-2">How it works:</h4>
          <ul className="text-slate-400 text-sm space-y-1">
            <li>• Upload any video or audio file (MP4, MOV, MP3, WAV, etc.)</li>
            <li>• AI analyzes the audio and generates accurate timestamps</li>
            <li>• Choose your preferred language and subtitle format</li>
            <li>• Download your SRT, VTT, or ASS subtitle file instantly</li>
            <li>• Cost: 1 credit per generation</li>
          </ul>
        </div>
      )}
    </div>
  );
}
