"use client";

import { motion } from "framer-motion";
import { FileVideo, Captions, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { useState, useCallback } from "react";
import { useUser } from "@/contexts/UserContext";
import { api, safeApiResponse, isSuccess } from "@/lib/api";
import { useRouter } from 'next/navigation';

export default function SubtitleGenerationPage() {
  const router = useRouter();
  const { user, refreshCredits } = useUser();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("en");
  const [format, setFormat] = useState("srt");
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

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

      const response = await api.generateSubtitles(selectedFile, format, language);

      if (isSuccess(response) && response.data && response.data.job_id) {
        setSuccess('Subtitle generation started! Redirecting to progress page...');

        // Store job ID for progress tracking
        localStorage.setItem('current_subtitle_job', response.data.job_id);

        // Refresh credits to show updated balance
        await refreshCredits();

        // Redirect to progress page after a short delay
        setTimeout(() => {
          router.push(`/dashboard/subtitles/progress?jobId=${response.data?.job_id}`);
        }, 1500);
      } else {
        setError(response.error || 'Failed to start subtitle generation');
      }
    } catch (err) {
      console.error('Subtitle generation error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
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
        onClick={() => !selectedFile && document.getElementById('media-upload')?.click()}
      >
        <div className="glass-shine" />
        <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

        <div className="relative z-20 py-12 px-6">
          {selectedFile ? (
            <div className="flex flex-col items-center justify-center gap-3 text-center">
              <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-green-500/10 border border-green-500/30">
                <FileVideo className="w-8 h-8 text-green-400" />
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

      {/* Configuration */}
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
              <span>Generating Subtitles...</span>
            </>
          ) : (
            <>
              <Captions className="w-5 h-5" />
              <span>{selectedFile ? 'Generate Subtitles' : 'Select a file to continue'}</span>
            </>
          )}
        </div>
      </button>

      {/* Info Panel */}
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
    </div>
  );
}
