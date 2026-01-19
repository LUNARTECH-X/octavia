"use client";

import { motion } from "framer-motion";
import { useState, useRef, useEffect, useMemo } from "react";
import { useUser } from "@/contexts/UserContext";
import { api, safeApiResponse, isSuccess } from "@/lib/api";
import { useRouter } from 'next/navigation';
import { useToast } from "@/hooks/use-toast";
import { Upload, Video, Rocket, Loader2, Sparkles, FileVideo, CheckCircle, AlertCircle, Play, Pause, Volume2 } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function VideoTranslationPage() {
  const { user, refreshCredits } = useUser();
  const [file, setFile] = useState<File | null>(null);
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [thumbnail, setThumbnail] = useState<string | null>(null);
  const [separate, setSeparate] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [previewText, setPreviewText] = useState("Hello! This is a preview of how the translated audio will sound.");
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
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
    return () => {
      if (audioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
      }
    };
  }, [audioElement]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      const validTypes = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'];
      const fileExtension = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();

      if (!validTypes.includes(fileExtension)) {
        setError(`Please upload a video file (${validTypes.join(', ')})`);
        return;
      }

      if (selectedFile.size > 500 * 1024 * 1024) {
        setError("File size too large. Maximum size is 500MB");
        return;
      }

      setFile(selectedFile);
      setError(null);
      generateThumbnail(selectedFile);
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

      const headers: Record<string, string> = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch(`${API_BASE_URL}/api/voices/preview`, {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Preview generation failed");
      }

      const data = await response.json();

      if (data.success && data.preview_url) {
        const audioUrl = `${API_BASE_URL}${data.preview_url}`;
        const audio = new Audio(audioUrl);

        audio.onended = () => {
          setIsPlayingPreview(false);
        };

        audio.onerror = () => {
          setIsPlayingPreview(false);
          alert("Failed to play preview audio");
        };

        setAudioElement(audio);
        setPreviewAudioUrl(audioUrl);
        audio.play().catch((err) => {
          setIsPlayingPreview(false);
          alert(`Failed to play audio: ${err.message}`);
        });

        refreshCredits();
      }
    } catch (err) {
      console.error("Preview error:", err);
      setIsPlayingPreview(false);
      alert("Failed to generate voice preview");
    }
  };

  const stopPreview = () => {
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
      setIsPlayingPreview(false);
    }
  };

  const handleStartTranslation = async () => {
    if (!file) {
      setError("Please select a video file first");
      return;
    }

    if (!file.name) {
      setError("Invalid file selected");
      return;
    }

    const token = getToken();
    if (!token) {
      setError("Please log in to start translation");
      return;
    }

    setLoading(true);
    setError(null);
    setUploadProgress(10);

    try {
      console.log("Starting video translation for file:", file.name);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('target_language', targetLanguage);
      formData.append('separate', separate.toString());

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/translate/video`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      const responseText = await response.text();
      console.log('Response status:', response.status);
      console.log('Response text:', responseText);

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('Failed to parse response:', parseError);
        throw new Error(`Server returned invalid JSON: ${responseText.substring(0, 100)}...`);
      }

      if (!response.ok) {
        const errorMessage = data.error || data.detail || data.message || `Upload failed: ${response.statusText}`;
        throw new Error(typeof errorMessage === 'string' ? errorMessage : JSON.stringify(errorMessage));
      }

      if (data.success && data.job_id) {
        setUploadProgress(100);

        toast({
          title: "Translation started!",
          description: "Your video translation has been queued. Redirecting to progress page...",
          variant: "default",
        });

        router.push(`/dashboard/video/progress?jobId=${data.job_id}`);

      } else {
        throw new Error(data.error || data.message || "Failed to start translation");
      }
    } catch (err: unknown) {
      console.error('Translation error:', err);
      const errorMessage = err instanceof Error ? err.message : "Failed to start video translation";
      setError(errorMessage);

      toast({
        title: "Translation failed",
        description: errorMessage || "Failed to start translation. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const generateThumbnail = (videoFile: File) => {
    const videoUrl = URL.createObjectURL(videoFile);
    setThumbnail(videoUrl);
  };

  const clearFile = () => {
    setFile(null);
    setThumbnail(null);
    setUploadProgress(0);
    setError(null);
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
    setPreviewAudioUrl(null);
    setIsPlayingPreview(false);
  };

  return (
    <div className="min-h-screen bg-[#030014]">
      <main className="relative overflow-hidden rounded-2xl">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]"></div>
        </div>

        <div className="relative max-w-6xl mx-auto px-6 py-12">
          {/* Header */}
          <div className="text-center mb-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h1 className="font-display text-5xl font-black text-white mb-4 text-glow-purple">
                Video Translation
              </h1>
              <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                Upload a video and translate it into any language with AI-powered lip-sync and voice cloning.
              </p>
            </motion.div>
          </div>

          {/* File Upload */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel glass-panel-glow p-8 mb-8 relative overflow-hidden rounded-2xl"
          >
            <div className="glass-shine" />
            <div className="relative z-10">
              {!file ? (
                <div
                  onClick={handleUploadClick}
                  className="border-2 border-dashed border-white/20 rounded-2xl p-12 text-center cursor-pointer hover:border-primary-purple/50 hover:bg-white/5 transition-all group"
                >
                  <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-primary-purple/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                    <Upload className="w-10 h-10 text-primary-purple-bright" />
                  </div>
                  <h3 className="text-white text-xl font-bold mb-2">Upload your video</h3>
                  <p className="text-slate-400 mb-4">Drag and drop or click to browse</p>
                  <p className="text-slate-500 text-sm">MP4, AVI, MOV, MKV, WebM, WMV, FLV (Max 500MB)</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".mp4,.avi,.mov,.mkv,.webm,.wmv,.flv"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </div>
              ) : (
                <div>
                  <div className="flex items-start gap-4">
                    {thumbnail && (
                      <div className="w-48 h-32 rounded-lg overflow-hidden flex-shrink-0">
                        <video src={thumbnail} className="w-full h-full object-cover" />
                      </div>
                    )}
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-white font-bold text-lg">{file.name}</h3>
                        <button
                          onClick={clearFile}
                          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                        >
                          <AlertCircle className="w-5 h-5 text-slate-400" />
                        </button>
                      </div>
                      <p className="text-slate-400 text-sm mb-4">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </p>

                      {loading && (
                        <div className="mb-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                              <span className="text-gray-400">
                                {uploadProgress < 100 ? "Uploading video file..." : "Video uploaded successfully!"}
                              </span>
                            </div>
                            <span className="text-white font-bold">{Math.round(uploadProgress)}%</span>
                          </div>
                          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                            <motion.div
                              className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright"
                              initial={{ width: "0%" }}
                              animate={{ width: `${uploadProgress}%` }}
                              transition={{ duration: 0.5 }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Target Language & Voice Preview */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6 pt-6 border-t border-white/10">
                    {/* Target Language */}
                    <div className="glass-card p-4">
                      <label className="text-white text-sm font-semibold mb-3 block">Translate To</label>
                      <select
                        className="glass-select w-full"
                        value={targetLanguage}
                        onChange={(e) => {
                          setTargetLanguage(e.target.value);
                          if (audioElement) {
                            audioElement.pause();
                            audioElement.currentTime = 0;
                          }
                          setIsPlayingPreview(false);
                        }}
                        disabled={loading}
                      >
                        <optgroup label="Popular Languages">
                          <option value="es">Spanish</option>
                          <option value="en">English</option>
                          <option value="fr">French</option>
                          <option value="de">German</option>
                          <option value="it">Italian</option>
                          <option value="pt">Portuguese</option>
                          <option value="ru">Russian</option>
                          <option value="ja">Japanese</option>
                          <option value="ko">Korean</option>
                          <option value="zh-cn">Chinese (Mandarin)</option>
                        </optgroup>
                        <optgroup label="European Languages">
                          <option value="nl">Dutch</option>
                          <option value="pl">Polish</option>
                          <option value="tr">Turkish</option>
                          <option value="sv">Swedish</option>
                        </optgroup>
                        <optgroup label="Other Languages">
                          <option value="ar">Arabic</option>
                        </optgroup>
                      </select>
                      <p className="text-slate-500 text-xs mt-2">
                        {languageNames[targetLanguage]} voice will be used for translation
                      </p>
                    </div>

                    {/* Voice Preview */}
                    <div className="glass-card p-4">
                      <label className="text-white text-sm font-semibold mb-3 block">Voice Preview</label>
                      <div className="flex items-center gap-2 mb-3">
                        <Volume2 className="w-4 h-4 text-primary-purple-bright" />
                        <span className="text-slate-300 text-sm">{languageNames[targetLanguage]}</span>
                      </div>
                      <textarea
                        value={previewText}
                        onChange={(e) => setPreviewText(e.target.value)}
                        className="glass-input w-full h-16 p-2 text-sm mb-3"
                        placeholder="Enter preview text..."
                      />
                      <div className="flex items-center gap-3">
                        {isPlayingPreview ? (
                          <button
                            onClick={stopPreview}
                            className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg text-sm transition-colors"
                          >
                            <Pause className="w-4 h-4" />
                            Stop
                          </button>
                        ) : (
                          <button
                            onClick={handlePreviewVoice}
                            className="flex items-center gap-2 px-4 py-2 bg-primary-purple/20 hover:bg-primary-purple/30 text-primary-purple-bright rounded-lg text-sm transition-colors"
                          >
                            <Play className="w-4 h-4" />
                            Preview Voice
                          </button>
                        )}
                        {isDemoUser && (
                          <span className="text-green-400 text-xs">Demo: Free</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* File Info */}
                  {file && (
                    <div className="glass-panel border-white/10 bg-white/5 mt-6 p-4 rounded-xl">
                      <div className="flex items-center gap-3">
                        <FileVideo className="w-5 h-5 text-slate-400" />
                        <div>
                          <p className="text-slate-300 text-sm">
                            Video ready for translation to {languageNames[targetLanguage]}
                          </p>
                          <p className="text-slate-500 text-xs mt-1">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB - Processing will begin when you start
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Magic Mode Toggle */}
                  <div className="glass-panel mt-6 p-4 rounded-xl">
                    <label className="flex items-center justify-between cursor-pointer group/toggle">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${separate ? 'bg-indigo-500/20 border-indigo-500/50' : 'bg-white/5 border-white/10'}`}>
                          <Sparkles className={`w-5 h-5 ${separate ? 'text-indigo-400' : 'text-slate-500'}`} />
                        </div>
                        <div>
                          <h4 className="text-white text-sm font-bold flex items-center gap-2">
                            Magic Mode (Vocal Separation)
                            {separate && <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500 text-white uppercase tracking-wider animate-pulse">Active</span>}
                          </h4>
                          <p className="text-slate-400 text-xs">Separate background music from vocals for cleaner dubbing</p>
                        </div>
                      </div>
                      <div className="relative">
                        <input
                          type="checkbox"
                          className="sr-only"
                          checked={separate}
                          onChange={() => setSeparate(!separate)}
                          disabled={loading}
                        />
                        <div className={`block w-14 h-8 rounded-full transition-all ${separate ? 'bg-indigo-600 shadow-glow' : 'bg-slate-700'}`}></div>
                        <div className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition-all ${separate ? 'transform translate-x-6 shadow-lg' : ''}`}></div>
                      </div>
                    </label>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex flex-col gap-4 mt-8">
                    <button
                      onClick={handleStartTranslation}
                      disabled={!file || loading}
                      className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                        {loading ? (
                          <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Processing...</span>
                          </>
                        ) : (
                          <>
                            <Rocket className="w-5 h-5" />
                            <span>Start Translation</span>
                          </>
                        )}
                      </div>
                    </button>

                    <div className="flex items-center justify-center gap-6 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        Secure processing
                      </span>
                      <span className="flex items-center gap-1">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        No upload limits
                      </span>
                      <span className="flex items-center gap-1">
                        <CheckCircle className="w-3 h-3 text-green-500" />
                        Fast delivery
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* Features */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-panel p-8"
          >
            <h2 className="text-white text-2xl font-bold mb-6 text-center">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div>
                <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center mb-4">
                  <FileVideo className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-white font-bold mb-2">1. Upload Video</h3>
                <p className="text-slate-400 text-sm">
                  Simply upload your video file. Our AI will analyze the content and prepare it for translation.
                </p>
              </div>
              <div>
                <div className="w-12 h-12 rounded-xl bg-primary-purple/20 flex items-center justify-center mb-4">
                  <Sparkles className="w-6 h-6 text-primary-purple-bright" />
                </div>
                <h3 className="text-white font-bold mb-2">2. AI Processing</h3>
                <p className="text-slate-400 text-sm">
                  Our AI transcribes, translates, and generates new speech with perfect timing and lip-sync.
                </p>
              </div>
              <div>
                <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center mb-4">
                  <Video className="w-6 h-6 text-green-400" />
                </div>
                <h3 className="text-white font-bold mb-2">3. Download Result</h3>
                <p className="text-slate-400 text-sm">
                  Receive your translated video with synchronized dubbed audio and optional subtitles.
                </p>
              </div>
            </div>

            <div className="mt-8 pt-8 border-t border-white/10">
              <h3 className="text-white font-bold mb-4">Technical Specifications</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="p-3 rounded-lg bg-white/5">
                  <p className="text-slate-500 mb-1">Input Formats</p>
                  <p className="text-white">MP4, AVI, MOV, MKV</p>
                </div>
                <div className="p-3 rounded-lg bg-white/5">
                  <p className="text-slate-500 mb-1">Max File Size</p>
                  <p className="text-white">500 MB</p>
                </div>
                <div className="p-3 rounded-lg bg-white/5">
                  <p className="text-slate-500 mb-1">Languages</p>
                  <p className="text-white">50+ Languages</p>
                </div>
                <div className="p-3 rounded-lg bg-white/5">
                  <p className="text-slate-500 mb-1">Output</p>
                  <p className="text-white">MP4 with Audio</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
