"use client";

import { motion } from "framer-motion";
import { useState, useRef, useEffect, useMemo } from "react";
import { useUser } from "@/contexts/UserContext";
import { api, safeApiResponse, isSuccess } from "@/lib/api";
import { useRouter } from 'next/navigation';
import { useToast } from "@/hooks/use-toast";
import { Upload, Video, Rocket, Loader2, Sparkles, FileVideo, CheckCircle, AlertCircle, Play, Pause, Volume2, AudioLines } from "lucide-react";

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

  // Check for project context on mount
  useEffect(() => {
    const projectContext = localStorage.getItem('octavia_project_context');
    if (projectContext) {
      try {
        const context = JSON.parse(projectContext);
        if (context.fileUrl && context.projectType === 'Video Translation') {
          // Fetch the blob and create a File object
          fetch(context.fileUrl)
            .then(response => response.blob())
            .then(blob => {
              const projectFile = new File([blob], context.fileName, { type: context.fileType });
              setFile(projectFile);
              setError(null);
              generateThumbnail(projectFile);
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
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="font-display text-3xl font-black text-white text-glow-purple">Video Translation</h1>
            <p className="text-slate-400 text-sm">Transform videos across languages with AI-powered lip-sync</p>
          </div>
          <div className="glass-card px-4 py-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-primary-purple-bright animate-pulse"></div>
              <span className="text-white text-sm">Credits: <span className="font-bold">{user?.credits || 0}</span></span>
            </div>
            <p className="text-slate-400 text-xs mt-1">Varies by video length</p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel border-red-500/30 bg-red-500/10 p-4"
        >
          <div className="flex items-start gap-3">
            <div className="mt-0.5">
              <div className="w-5 h-5 rounded-full bg-red-500/20 border border-red-500/30 flex items-center justify-center">
                <span className="text-red-400 text-xs">!</span>
              </div>
            </div>
            <div className="flex-1">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-300 transition-colors"
            >
              <AlertCircle className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Upload Zone */}
      <div
        onClick={!file && !loading ? handleUploadClick : undefined}
        className="relative"
      >
        <input
          type="file"
          ref={fileInputRef}
          accept=".mp4,.avi,.mov,.mkv,.webm,.wmv,.flv"
          onChange={handleFileChange}
          className="hidden"
          disabled={loading}
        />
        
        <motion.div
          whileHover={!file && !loading ? { scale: 1.01 } : {}}
          className={`glass-panel glass-panel-high relative border-2 border-dashed transition-all mb-6 overflow-hidden
            ${file ? 'border-green-500/50 cursor-default' : 
              loading ? 'border-primary-purple/30 cursor-wait' : 
              'border-primary-purple/30 hover:border-primary-purple/50 cursor-pointer'}`}
        >
          <div className="glass-shine" />
          <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

          <div className="relative z-20 py-12 px-6">
            {file ? (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-green-500/10 border border-green-500/30 shadow-glow">
                  <FileVideo className="w-8 h-8 text-green-500" />
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-green">{file.name}</h3>
                  <p className="text-slate-400 text-sm">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready to translate
                  </p>
                </div>
                {!loading && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      clearFile();
                    }}
                    className="mt-4 px-4 py-2 text-sm bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors flex items-center gap-2"
                  >
                    <AlertCircle className="w-4 h-4" />
                    Remove File
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className={`flex items-center justify-center w-16 h-16 rounded-2xl ${loading ? 'bg-primary-purple/20' : 'bg-primary-purple/10'} border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform`}>
                  {loading ? (
                    <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
                  ) : (
                    <Upload className="w-8 h-8 text-primary-purple-bright" />
                  )}
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                    {loading ? 'Processing...' : 'Drop your video file here'}
                  </h3>
                  <p className="text-slate-400 text-sm">
                    {loading ? 'Video translation in progress...' : 'or click to browse files • MP4, AVI, MOV, MKV supported'}
                  </p>
                  <p className="text-slate-500 text-xs mt-2">Max file size: 500MB</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Configuration */}
      {file && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Target Language */}
          <div className="glass-card p-4">
            <label className="text-white text-sm font-semibold mb-2 block">Target Language</label>
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
            <label className="text-white text-sm font-semibold mb-2 block">Voice Preview</label>
            <div className="flex items-center gap-2 mb-3">
              <Volume2 className="w-4 h-4 text-primary-purple-bright" />
              <span className="text-slate-300 text-sm">{languageNames[targetLanguage]}</span>
            </div>
            <textarea
              value={previewText}
              onChange={(e) => setPreviewText(e.target.value)}
              className="glass-input w-full h-16 p-2 text-sm mb-3"
              placeholder="Enter preview text..."
              disabled={loading}
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
                  disabled={loading}
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
      )}

      {/* Progress Bar */}
      {loading && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-4 mb-6"
        >
          <div className="flex justify-between text-sm mb-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${uploadProgress < 100 ? 'bg-blue-500 animate-pulse' : 'bg-green-500'}`}></div>
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
          
          {/* Stage breakdown */}
          <div className="grid grid-cols-5 gap-2 mt-4">
            <div className={`text-center p-2 rounded ${uploadProgress < 30 ? 'bg-blue-500/20 border border-blue-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">1. Upload</div>
              <div className={`text-xs ${uploadProgress < 30 ? 'text-blue-400' : 'text-gray-500'}`}>Video File</div>
            </div>
            <div className={`text-center p-2 rounded ${uploadProgress >= 30 && uploadProgress < 55 ? 'bg-yellow-500/20 border border-yellow-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">2. Transcribe</div>
              <div className={`text-xs ${uploadProgress >= 30 && uploadProgress < 55 ? 'text-yellow-400' : 'text-gray-500'}`}>Whisper</div>
            </div>
            <div className={`text-center p-2 rounded ${uploadProgress >= 55 && uploadProgress < 75 ? 'bg-purple-500/20 border border-purple-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">3. Translate</div>
              <div className={`text-xs ${uploadProgress >= 55 && uploadProgress < 75 ? 'text-purple-400' : 'text-gray-500'}`}>LLM</div>
            </div>
            <div className={`text-center p-2 rounded ${uploadProgress >= 75 && uploadProgress < 95 ? 'bg-pink-500/20 border border-pink-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">4. Synthesize</div>
              <div className={`text-xs ${uploadProgress >= 75 && uploadProgress < 95 ? 'text-pink-400' : 'text-gray-500'}`}>edge-tts</div>
            </div>
            <div className={`text-center p-2 rounded ${uploadProgress >= 95 ? 'bg-green-500/20 border border-green-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">5. Lip-Sync</div>
              <div className={`text-xs ${uploadProgress >= 95 ? 'text-green-400' : 'text-gray-500'}`}>Wav2Lip</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Magic Mode Toggle */}
      {file && (
        <div className="glass-panel p-4 mb-6">
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
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={handleStartTranslation}
          disabled={!file || loading}
          className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed bg-primary-purple/10 border-primary-purple/30 hover:bg-primary-purple/20 transition-all duration-300"
        >
          <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Rocket className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Start Translation</span>
              </>
            )}
          </div>
        </button>
      </div>

      {/* Status Message */}
      {loading && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel border-blue-500/30 bg-blue-500/10 p-4"
        >
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
              </div>
            </div>
            <div>
              <h3 className="text-white font-semibold">Translation in Progress</h3>
              <p className="text-blue-400 text-sm">Your video is being processed. This may take several minutes.</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      <div className="glass-card p-4 mt-8">
        <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
          <Video className="w-5 h-5 text-primary-purple-bright" />
          How Video Translation Works:
        </h3>
        <ol className="text-slate-400 text-sm space-y-3 pl-2">
          <li className="flex items-start gap-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-purple/20 border border-primary-purple/30 flex items-center justify-center text-primary-purple-bright text-xs font-bold">
              1
            </div>
            <div>
              <span className="font-medium text-slate-300">Upload Video</span>
              <p className="text-slate-500">Select your video file (MP4, AVI, MOV, MKV). Maximum size: 500MB.</p>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-purple/20 border border-primary-purple/30 flex items-center justify-center text-primary-purple-bright text-xs font-bold">
              2
            </div>
            <div>
              <span className="font-medium text-slate-300">Configure Settings</span>
              <p className="text-slate-500">Choose target language, preview voice, and enable Magic Mode for vocal separation.</p>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-purple/20 border border-primary-purple/30 flex items-center justify-center text-primary-purple-bright text-xs font-bold">
              3
            </div>
            <div>
              <span className="font-medium text-slate-300">AI Processing</span>
              <p className="text-slate-500">Whisper transcribes, LLM translates, edge-tts generates audio, Wav2Lip syncs lips.</p>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-purple/20 border border-primary-purple/30 flex items-center justify-center text-primary-purple-bright text-xs font-bold">
              4
            </div>
            <div>
              <span className="font-medium text-slate-300">Download Result</span>
              <p className="text-slate-500">Download your translated video with synchronized lip-sync and dubbed audio.</p>
            </div>
          </li>
        </ol>
        
        <div className="mt-6 p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-lg">
          <div className="flex items-start gap-3">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
                <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <div>
              <h4 className="text-blue-400 font-semibold mb-1">Important Notes</h4>
              <ul className="text-blue-300/80 text-sm space-y-1">
                <li>• Credit cost varies based on video length (1-2 min processing time per minute)</li>
                <li>• OpenAI Whisper provides high-accuracy transcription with timestamps</li>
                <li>• LLM-powered translation (TranslateGemma, NLLB) for accurate translations</li>
                <li>• edge-tts generates natural voice synthesis with proper intonation</li>
                <li>• Wav2Lip provides lip-sync synchronization (±7ms precision)</li>
                <li>• Your files are processed securely and deleted after 24 hours</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Process */}
      <div className="glass-card p-4">
        <h3 className="text-white font-semibold mb-3">Technical Process</h3>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="bg-blue-500/10 border border-blue-500/20 p-3 rounded">
            <div className="text-blue-400 text-xs font-semibold mb-1">1. OpenAI Whisper</div>
            <p className="text-slate-400 text-xs">Speech-to-text transcription with timestamps</p>
          </div>
          <div className="bg-purple-500/10 border border-purple-500/20 p-3 rounded">
            <div className="text-purple-400 text-xs font-semibold mb-1">2. LLM Translation</div>
            <p className="text-slate-400 text-xs">TranslateGemma/NLLB models</p>
          </div>
          <div className="bg-pink-500/10 border border-pink-500/20 p-3 rounded">
            <div className="text-pink-400 text-xs font-semibold mb-1">3. edge-tts</div>
            <p className="text-slate-400 text-xs">Text-to-speech synthesis</p>
          </div>
          <div className="bg-green-500/10 border border-green-500/20 p-3 rounded">
            <div className="text-green-400 text-xs font-semibold mb-1">4. Wav2Lip</div>
            <p className="text-slate-400 text-xs">Lip-sync synchronization</p>
          </div>
          <div className="bg-yellow-500/10 border border-yellow-500/20 p-3 rounded">
            <div className="text-yellow-400 text-xs font-semibold mb-1">5. Output</div>
            <p className="text-slate-400 text-xs">MP4 with synced audio</p>
          </div>
        </div>
      </div>

      {/* File Info */}
      {file && (
        <div className="glass-panel border-white/10 bg-white/5 p-4 rounded-xl">
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

      {/* Trust Indicators */}
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
  );
}
