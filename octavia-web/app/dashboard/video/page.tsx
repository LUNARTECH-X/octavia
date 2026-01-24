"use client";

import { motion } from "framer-motion";
import { useState, useRef, useEffect, useMemo } from "react";
import { useUser } from "@/contexts/UserContext";
import { useRouter } from 'next/navigation';
import { useToast } from "@/hooks/use-toast";
import { Upload, Video, Rocket, Loader2, Sparkles, FileVideo, CheckCircle, AlertCircle, Play, Pause, Volume2, AudioLines } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function VideoTranslationPage() {
  const router = useRouter();
  const { user, refreshCredits } = useUser();
  const [file, setFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [thumbnail, setThumbnail] = useState<string | null>(null);
  const [separate, setSeparate] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState<any>(null);

  const [previewText, setPreviewText] = useState("Hello! This is a preview of how the translated audio will sound.");
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

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

  const isDemoUser = useMemo(() => {
    return user?.email === "demo@octavia.com" || false;
  }, [user]);

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
    'zh-cn': 'Chinese (Mandarin)',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'sv': 'Swedish'
  };

  useEffect(() => {
    // Attempt to recover active job from localStorage
    const savedJobId = localStorage.getItem('octavia_active_video_job');
    if (savedJobId && !activeJobId) {
      console.log("Recovering active job:", savedJobId);
      setActiveJobId(savedJobId);
      setLoading(true);
    }

    return () => {
      if (audioElement) {
        audioElement.pause();
      }
    };
  }, []);

  // Poll for job progress
  useEffect(() => {
    if (!activeJobId || !loading) return;

    const pollProgress = async () => {
      try {
        const token = getToken();
        if (!token) return;

        const response = await fetch(`${API_BASE_URL}/api/progress/${activeJobId}`, {
          headers: { 'Authorization': `Bearer ${token}` },
        });

        if (response.ok) {
          const data = await response.json();
          setJobProgress(data);

          if (data.progress !== undefined) {
            setUploadProgress(data.progress);
          }

          if (data.status === 'completed' || data.status === 'failed') {
            setLoading(false);
            localStorage.removeItem('octavia_active_video_job');
            if (data.status === 'completed') {
              toast({
                title: "Translation completed!",
                description: "Taking you to the review page...",
                variant: "default",
              });
              // Redirect to review page after a short delay for the toast
              setTimeout(() => {
                router.push(`/dashboard/video/review?jobId=${activeJobId}`);
              }, 1500);
            } else {
              setError(data.error || "Translation failed");
            }
          }
        }
      } catch (err) {
        console.error("Polling error:", err);
        // If we get persistent errors for a recovered job, maybe it's dead
        if (activeJobId && loading) {
          // We'll keep trying for now, but in a real app we might clear after N failures
        }
      }
    };

    const interval = setInterval(pollProgress, 2000);
    return () => clearInterval(interval);
  }, [activeJobId, loading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handlePreviewVoice = async () => {
    if (!targetLanguage) return;

    try {
      const token = getToken();
      if (!token && !isDemoUser) {
        alert("Please log in to preview voices");
        return;
      }

      setIsPlayingPreview(true);

      const formData = new FormData();
      formData.append('voice_id', targetLanguage);
      formData.append('text', previewText);
      formData.append('language', targetLanguage);

      const response = await fetch(`${API_BASE_URL}/api/voices/preview`, {
        method: 'POST',
        headers: token ? { 'Authorization': `Bearer ${token}` } : {},
        body: formData,
      });

      if (!response.ok) throw new Error("Preview generation failed");

      const data = await response.json();
      if (data.success && data.preview_url) {
        const audio = new Audio(`${API_BASE_URL}${data.preview_url}`);
        audio.onended = () => setIsPlayingPreview(false);
        setAudioElement(audio);
        audio.play();
        refreshCredits();
      }
    } catch (err) {
      setIsPlayingPreview(false);
      alert("Failed to generate voice preview");
    }
  };

  const handleStartTranslation = async () => {
    if (!file) return;
    const token = getToken();
    if (!token) {
      setError("Please log in to start translation");
      return;
    }

    setLoading(true);
    setError(null);
    setUploadProgress(10);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);
      formData.append('separate', separate.toString());

      const response = await fetch(`${API_BASE_URL}/api/translate/video`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Upload failed");

      if (data.success && data.job_id) {
        setUploadProgress(100);
        setActiveJobId(data.job_id);
        localStorage.setItem('octavia_active_video_job', data.job_id);
        toast({ title: "Translation started!", description: "Watch progress below." });
      }
    } catch (err: any) {
      setError(err.message || "Failed to start translation");
      setLoading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setUploadProgress(0);
    setError(null);
    setActiveJobId(null);
    setJobProgress(null);
    localStorage.removeItem('octavia_active_video_job');
  };

  return (
    <div className="space-y-8 min-h-screen">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="font-display text-3xl font-black text-white text-glow-purple">Video Translation</h1>
            <p className="text-slate-400 text-sm">Transform videos across languages with AI-powered audio alignment</p>
          </div>
          <div className="glass-card px-4 py-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-primary-purple-bright animate-pulse" />
              <span className="text-white text-sm font-medium">Credits: <span className="font-bold text-lg">{user?.credits || 0}</span></span>
            </div>
            <p className="text-slate-400 text-[10px] mt-1 uppercase tracking-wider">Available Balance</p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="glass-panel border-red-500/30 bg-red-500/10 p-4">
          <div className="flex items-center gap-3 text-red-400">
            <AlertCircle className="w-5 h-5" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        </motion.div>
      )}

      {/* Upload Zone */}
      <div onClick={!file && !loading ? handleUploadClick : undefined} className="relative">
        <input type="file" ref={fileInputRef} accept=".mp4,.avi,.mov,.mkv,.webm" onChange={handleFileChange} className="hidden" disabled={loading} />

        <motion.div
          whileHover={!file && !loading ? { scale: 1.01 } : {}}
          className={`glass-panel glass-panel-high relative border-2 border-dashed transition-all min-h-[300px] flex items-center justify-center
            ${file ? 'border-green-500/50 cursor-default' : loading ? 'border-primary-purple/30 cursor-wait' : 'border-primary-purple/30 hover:border-primary-purple/50 cursor-pointer'}`}
        >
          <div className="glass-shine" />
          <div className="relative z-20 py-12 px-6 text-center">
            {file ? (
              <div className="flex flex-col items-center gap-4">
                <div className="w-20 h-20 rounded-2xl bg-green-500/10 border border-green-500/30 flex items-center justify-center shadow-glow-green">
                  <FileVideo className="w-10 h-10 text-green-500" />
                </div>
                <div>
                  <h3 className="text-white text-xl font-bold mb-1 text-glow-green">{file.name}</h3>
                  <p className="text-slate-400 text-sm">{(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to translate</p>
                </div>
                {!loading && (
                  <button onClick={(e) => { e.stopPropagation(); clearFile(); }} className="mt-4 px-4 py-2 text-sm bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    Remove File
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4 group">
                <div className={`w-16 h-16 rounded-2xl ${loading ? 'bg-primary-purple/20' : 'bg-primary-purple/10'} border border-primary-purple/30 flex items-center justify-center shadow-glow group-hover:scale-110 transition-transform`}>
                  {loading ? <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" /> : <Upload className="w-8 h-8 text-primary-purple-bright" />}
                </div>
                <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">Drop video or click to browse</h3>
                <p className="text-slate-400 text-sm">MP4, MOV, MKV supported • Max 500MB</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Configuration */}
      {file && !activeJobId && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Target Language */}
            <div className="glass-card p-4">
              <label className="text-white text-sm font-bold mb-3 block">Target Language</label>
              <select className="glass-select w-full" value={targetLanguage} onChange={(e) => setTargetLanguage(e.target.value)}>
                {Object.entries(languageNames).map(([code, name]) => (
                  <option key={code} value={code}>{name}</option>
                ))}
              </select>
            </div>

            {/* Voice Preview */}
            <div className="glass-card p-4">
              <label className="text-white text-sm font-bold mb-3 block">Voice Preview</label>
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <p className="text-slate-300 text-sm">{languageNames[targetLanguage]} Voice</p>
                </div>
                <button onClick={handlePreviewVoice} className="flex items-center gap-2 px-4 py-2 bg-primary-purple/20 hover:bg-primary-purple/30 text-primary-purple-bright rounded-lg text-sm transition-colors">
                  {isPlayingPreview ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  Preview
                </button>
              </div>
            </div>
          </div>

          {/* Magic Mode Toggle */}
          <div className="glass-panel p-4">
            <label className="flex items-center justify-between cursor-pointer">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${separate ? 'bg-indigo-500/20 border-indigo-500/50' : 'bg-white/5 border-white/10'}`}>
                  <Sparkles className={`w-5 h-5 ${separate ? 'text-indigo-400' : 'text-slate-500'}`} />
                </div>
                <div>
                  <h4 className="text-white text-sm font-bold flex items-center gap-2">
                    Magic Mode (Vocal Separation)
                  </h4>
                  <p className="text-slate-400 text-xs text-glow-indigo">Separate background music from vocals for cleaner dubbing</p>
                </div>
              </div>
              <div className="relative">
                <input type="checkbox" className="sr-only" checked={separate} onChange={() => setSeparate(!separate)} />
                <div className={`block w-14 h-8 rounded-full transition-all ${separate ? 'bg-indigo-600 shadow-glow' : 'bg-slate-700'}`} />
                <div className={`absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-all ${separate ? 'transform translate-x-6' : ''}`} />
              </div>
            </label>
          </div>

          {/* Start Button */}
          <button onClick={handleStartTranslation} className="btn-border-beam w-full group">
            <div className="btn-border-beam-inner py-4 text-base font-bold flex items-center justify-center gap-2">
              <Rocket className="w-5 h-5 group-hover:scale-110 transition-transform" />
              Start Translation
            </div>
          </button>
        </motion.div>
      )}

      {/* Progress & Breakdown Section */}
      {loading && (
        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="glass-panel p-4 space-y-4">
          <div className="flex justify-between items-center text-sm">
            <span className="text-slate-300 font-medium">{jobProgress?.message || "Processing..."}</span>
            <span className="text-primary-purple-bright font-bold">{Math.round(uploadProgress)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright transition-all duration-500" style={{ width: `${uploadProgress}%` }} />
          </div>

          {/* Stage breakdown */}
          <div className="grid grid-cols-5 gap-2 pt-2">
            {[
              { label: "Upload", range: [0, 30] },
              { label: "Transcribe", range: [30, 55] },
              { label: "Translate", range: [55, 75] },
              { label: "Synthesize", range: [75, 95] },
              { label: "Aligning", range: [95, 101] }
            ].map((stage, i) => (
              <div key={i} className={`text-center p-2 rounded transition-all duration-500 
                ${uploadProgress >= stage.range[0] && uploadProgress < stage.range[1] ? 'bg-primary-purple/20 border border-primary-purple/30' : uploadProgress >= stage.range[1] ? 'bg-green-500/10 border border-green-500/30' : 'bg-gray-500/10 opacity-40'}`}>
                <div className="text-[10px] text-slate-400 uppercase">{stage.label}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Instructions & Technical Details */}
      {!loading && !activeJobId && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-8">
          <div className="glass-card p-4 lg:col-span-2">
            <h3 className="text-white font-bold text-lg mb-3 flex items-center gap-2">
              <AudioLines className="w-5 h-5 text-primary-purple-bright" />
              Neural Translation Workflow
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { title: "Speech Recognition", desc: "Faster Whisper decodes audio." },
                { title: "Smart Translation", desc: "Context-aware LLM flow." },
                { title: "Synthesized Audio", desc: "gTTS high-fidelity generation." },
                { title: "Temporal Alignment", desc: "Precision audio-visual synchronization." }
              ].map((item, i) => (
                <div key={i} className="bg-white/5 p-3 rounded-lg border border-white/5">
                  <h4 className="text-white font-bold text-sm mb-1">{item.title}</h4>
                  <p className="text-slate-500 text-xs leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="glass-card p-4 border-indigo-500/10 h-min">
            <h3 className="text-white font-bold text-lg mb-3">Magic Mode</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />
                <p>Isolates vocal tracks from background scores.</p>
              </li>
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />
                <p>Removes echoes for studio quality.</p>
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
