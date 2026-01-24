"use client";

import { motion } from "framer-motion";
import { FileText, AudioWaveform, Sparkles, Upload, Download, X, Loader2, Info, Play, Pause, Volume2, CheckCircle } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { useUser } from "@/contexts/UserContext";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface LanguageOption {
  value: string;
  label: string;
  code: string;
}

interface VoiceOption {
  value: string;
  label: string;
}

export default function SubtitleToAudioPage() {
  const { toast } = useToast();
  const { user, refreshCredits } = useUser();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>("idle");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [voice, setVoice] = useState("Aria (Female)");
  const [outputFormat, setOutputFormat] = useState("mp3");
  const [availableVoices, setAvailableVoices] = useState<VoiceOption[]>([]);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [previewText, setPreviewText] = useState("Hello! This is a preview of how the translated audio will sound.");
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const [previewAudioElement, setPreviewAudioElement] = useState<HTMLAudioElement | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const languageOptions: LanguageOption[] = [
    { value: "en", label: "English", code: "en" },
    { value: "es", label: "Spanish", code: "es" },
    { value: "fr", label: "French", code: "fr" },
    { value: "de", label: "German", code: "de" },
    { value: "it", label: "Italian", code: "it" },
    { value: "pt", label: "Portuguese", code: "pt" },
    { value: "ru", label: "Russian", code: "ru" },
    { value: "ja", label: "Japanese", code: "ja" },
    { value: "ko", label: "Korean", code: "ko" },
    { value: "zh", label: "Chinese", code: "zh" },
    { value: "ar", label: "Arabic", code: "ar" },
    { value: "hi", label: "Hindi", code: "hi" },
  ];

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

  // Refresh credits when component mounts
  useEffect(() => {
    refreshCredits();
  }, []);

  // Check for project context on mount
  useEffect(() => {
    const projectContext = localStorage.getItem('octavia_project_context');
    if (projectContext) {
      try {
        const context = JSON.parse(projectContext);
        if (context.fileUrl && context.projectType === 'Subtitle to Audio') {
          // Fetch the blob and create a File object
          fetch(context.fileUrl)
            .then(response => response.blob())
            .then(blob => {
              const projectFile = new File([blob], context.fileName, { type: context.fileType });
              setSelectedFile(projectFile);
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

  // Load available voices when target language changes
  useEffect(() => {
    const loadVoices = async () => {
      try {
        const token = getToken();
        if (!token) return;

        const response = await fetch(`${API_BASE_URL}/api/voices/${targetLanguage}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.voices) {
            const voices = data.voices.map((v: any) => ({
              value: v.voice_id || v.name || v,
              label: v.name || v
            }));
            setAvailableVoices(voices);
            if (voices.length > 0) {
              setVoice(voices[0].value);
            }
          }
        }
      } catch (error) {
        console.error("Failed to load voices:", error);
        // Fallback voices based on language
        const fallbackVoices: Record<string, VoiceOption[]> = {
          en: [
            { value: "Aria (Female)", label: "Aria (Female)" },
            { value: "David (Male)", label: "David (Male)" },
          ],
          es: [
            { value: "Elena (Female)", label: "Elena (Female)" },
            { value: "Alvaro (Male)", label: "Alvaro (Male)" },
          ],
          fr: [
            { value: "Denise (Female)", label: "Denise (Female)" },
            { value: "Henri (Male)", label: "Henri (Male)" },
          ],
          de: [
            { value: "Katja (Female)", label: "Katja (Female)" },
            { value: "Conrad (Male)", label: "Conrad (Male)" },
          ],
        };
        setAvailableVoices(fallbackVoices[targetLanguage] || fallbackVoices.en);
      }
    };

    loadVoices();
  }, [targetLanguage]);

  // Load audio for playback when job completes
  useEffect(() => {
    if (jobStatus === "completed" && downloadUrl && !audioUrl) {
      loadAudioForPlayback();
    }
  }, [jobStatus, downloadUrl, audioUrl]);

  // Handle voice preview
  const handlePreviewVoice = async () => {
    if (!voice) return;

    try {
      const token = getToken();
      if (!token) {
        alert("Please log in to preview voices");
        return;
      }

      setIsPlayingPreview(true);

      const formData = new FormData();
      formData.append('voice_id', voice);
      formData.append('text', previewText);
      formData.append('language', targetLanguage);

      const response = await fetch(`${API_BASE_URL}/api/voices/preview`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
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

        setPreviewAudioElement(audio);
        audio.play().catch((err) => {
          setIsPlayingPreview(false);
          alert(`Failed to play audio: ${err.message}`);
        });
      }
    } catch (err) {
      console.error("Preview error:", err);
      setIsPlayingPreview(false);
      alert("Failed to generate voice preview");
    }
  };

  const stopPreview = () => {
    if (previewAudioElement) {
      previewAudioElement.pause();
      previewAudioElement.currentTime = 0;
    }
    setIsPlayingPreview(false);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validExtensions = ['.srt', '.vtt', '.ass', '.ssa', '.txt'];
    const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!validExtensions.includes(fileExt)) {
      setError("Please upload SRT, VTT, ASS, SSA, or TXT files only.");
      return;
    }

    // Check file size (max 10MB for subtitle files)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setError("File size too large. Maximum size is 10MB.");
      return;
    }

    setSelectedFile(file);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];

    if (file) {
      const validExtensions = ['.srt', '.vtt', '.ass', '.ssa', '.txt'];
      const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (validExtensions.includes(fileExt)) {
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
          setError("File size too large. Maximum size is 10MB.");
          return;
        }
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Please upload SRT, VTT, ASS, SSA, or TXT files only.");
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleGenerateAudio = async () => {
    if (!selectedFile) {
      setError("Please select a subtitle file first.");
      return;
    }

    const token = getToken();
    if (!token) {
      setError("Please log in to generate audio.");
      return;
    }

    // Check if user has enough credits (5 credits for subtitle to audio)
    if ((user?.credits || 0) < 5) {
      setError(`Insufficient credits. You need 5 credits but only have ${user?.credits || 0}. Please purchase more credits.`);
      return;
    }

    setIsGenerating(true);
    setJobStatus("uploading");
    setUploadProgress(10);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('source_language', sourceLanguage);
      formData.append('target_language', targetLanguage);
      formData.append('voice', voice);
      formData.append('output_format', outputFormat);

      console.log('Generating audio from subtitle:', selectedFile.name);
      console.log('Source language:', sourceLanguage);
      console.log('Target language:', targetLanguage);
      console.log('Voice:', voice);
      console.log('Format:', outputFormat);

      const response = await fetch(`${API_BASE_URL}/api/generate/subtitle-audio`, {
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
        throw new Error(data.error || data.detail || data.message || `Upload failed: ${response.statusText}`);
      }

      if (data.success && data.job_id) {
        setJobId(data.job_id);
        setJobStatus("processing");
        setUploadProgress(30);

        // Start polling for job status
        pollJobStatus(data.job_id);

        // Refresh user credits to update the balance
        refreshCredits();
        pollJobStatus(data.job_id);
      } else {
        throw new Error(data.error || data.message || "Failed to start audio generation");
      }
    } catch (err: any) {
      console.error('Generation error:', err);
      setError(err.message || "Failed to generate audio");
      setIsGenerating(false);
      setJobStatus("idle");
      setUploadProgress(0);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const token = getToken();
    let attempts = 0;
    const maxAttempts = 60; // 5 minutes max
    let pollInterval: NodeJS.Timeout;

    const poll = async () => {
      attempts++;
      console.log(`Polling job ${jobId}, attempt ${attempts}`);

      try {
        const response = await fetch(`${API_BASE_URL}/api/generate/subtitle-audio/status/${jobId}`, {
          headers: token ? {
            'Authorization': `Bearer ${token}`
          } : {},
        });

        if (!response.ok) {
          throw new Error(`Status check failed: ${response.statusText}`);
        }

        const data = await response.json();
        console.log('Poll response:', data);

        if (data.success || data.status) {
          const jobProgress = data.progress || 0;
          const calculatedProgress = 30 + (jobProgress * 0.7);
          setUploadProgress(Math.min(calculatedProgress, 99));

          if (data.status === "completed") {
            setJobStatus("completed");
            setUploadProgress(100);

            // Set download URL from the response
            const downloadUrl = data.download_url ||
              `${API_BASE_URL}/api/download/subtitle-audio/${jobId}`;
            setDownloadUrl(downloadUrl);
            setIsGenerating(false);
            clearInterval(pollInterval);

            console.log('Job completed! Download URL:', downloadUrl);

            toast({
              title: "Audio generated successfully!",
              description: "Your audio file is ready to download.",
              variant: "default",
            });
          } else if (data.status === "failed") {
            setJobStatus("failed");
            setIsGenerating(false);
            clearInterval(pollInterval);

            toast({
              title: "Generation failed",
              description: data.error || "Failed to generate audio.",
              variant: "destructive",
            });
          } else {
            // Continue polling
            if (attempts >= maxAttempts) {
              setJobStatus("failed");
              setIsGenerating(false);
              clearInterval(pollInterval);

              toast({
                title: "Generation timeout",
                description: "Audio generation took too long. Please try again.",
                variant: "destructive",
              });
            }
          }
        } else {
          setJobStatus("failed");
          setIsGenerating(false);
          clearInterval(pollInterval);

          toast({
            title: "Generation failed",
            description: data.error || "Job status check failed",
            variant: "destructive",
          });
        }
      } catch (err: any) {
        console.error("Polling error:", err);
        if (attempts >= maxAttempts) {
          setJobStatus("failed");
          setIsGenerating(false);
          clearInterval(pollInterval);

          toast({
            title: "Generation timeout",
            description: "Failed to check generation status",
            variant: "destructive",
          });
        }
      }
    };

    // Start polling
    pollInterval = setInterval(poll, 2000);

    // Cleanup on component unmount
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  };

  const handleDownload = async () => {
    if (!jobId) {
      setError("No job ID found. Please start a generation first.");
      return;
    }

    try {
      const token = getToken();
      if (!token) {
        setError("Please log in to download files.");
        return;
      }

      console.log('Downloading audio for job:', jobId);
      console.log('Download URL from job status:', downloadUrl);
      console.log('Job ID:', jobId);
      console.log('Output format:', outputFormat);
      console.log('API Base URL:', API_BASE_URL);

      // Try multiple possible download endpoints, but prioritize the one from job status
      const downloadUrls = [
        downloadUrl, // Use the URL from job status FIRST (most likely to work)
        `${API_BASE_URL}/api/download/subtitle-audio/${jobId}`,
        `${API_BASE_URL}/api/generate/subtitle-audio/download/${jobId}`,
        `${API_BASE_URL}/api/download/${jobId}`
      ].filter(url => url !== null && url !== undefined && url !== '');

      console.log('Download URLs to try:', downloadUrls);

      let success = false;
      let lastError = null;

      for (const url of downloadUrls) {
        if (!url) continue;
        try {
          console.log('Trying download URL:', url);

          // Ensure URL is absolute
          let absoluteUrl = url;
          if (url.startsWith('/')) {
            absoluteUrl = `${API_BASE_URL}${url}`;
          }

          console.log('Using absolute URL:', absoluteUrl);

          const response = await fetch(absoluteUrl, {
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          });

          console.log('Response status:', response.status);
          console.log('Response headers:', Object.fromEntries(response.headers.entries()));

          if (response.ok) {
            const blob = await response.blob();

            // Extract filename from response headers or create one
            const contentDisposition = response.headers.get('content-disposition');
            let fileName = `subtitle_audio_${jobId}.${outputFormat}`;

            if (contentDisposition) {
              const match = contentDisposition.match(/filename="?(.+?)"?$/);
              if (match) {
                fileName = match[1];
              }
            }

            // Create download link
            const downloadUrlObj = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = downloadUrlObj;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(downloadUrlObj);

            console.log('Audio file downloaded:', fileName);
            success = true;

            toast({
              title: "Download started",
              description: "Your audio file is being downloaded.",
              variant: "default",
            });

            break;
          } else {
            console.log(`Download failed for ${absoluteUrl}: ${response.status} ${response.statusText}`);
            lastError = `Server returned ${response.status}: ${response.statusText}`;
          }
        } catch (err: any) {
          console.error(`Error downloading from ${url}:`, err);
          lastError = err.message;
        }
      }

      if (!success) {
        throw new Error(lastError || "Failed to download from all possible endpoints");
      }

    } catch (err: any) {
      console.error('Download error:', err);
      setError(`Failed to download audio file: ${err.message}. Please try again or contact support.`);

      toast({
        title: "Download failed",
        description: `Failed to download audio file: ${err.message}`,
        variant: "destructive",
      });
    }
  };

  // Audio playback functions
  const loadAudioForPlayback = async () => {
    if (!downloadUrl || !jobId) return;

    try {
      const token = getToken();
      if (!token) return;

      // Try to get the audio URL for playback
      const audioUrls = [
        downloadUrl,
        `${API_BASE_URL}/api/download/subtitle-audio/${jobId}`,
        `${API_BASE_URL}/api/generate/subtitle-audio/download/${jobId}`,
        `${API_BASE_URL}/api/download/${jobId}`
      ].filter(url => url !== null && url !== undefined && url !== '');

      let audioBlob: Blob | null = null;

      for (const url of audioUrls) {
        try {
          let absoluteUrl = url;
          if (url.startsWith('/')) {
            absoluteUrl = `${API_BASE_URL}${url}`;
          }

          const response = await fetch(absoluteUrl, {
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          });

          if (response.ok) {
            audioBlob = await response.blob();
            break;
          }
        } catch (err) {
          console.error(`Error loading audio from ${url}:`, err);
        }
      }

      if (audioBlob) {
        const audioObjectUrl = URL.createObjectURL(audioBlob);
        setAudioUrl(audioObjectUrl);

        // Set up audio element
        if (audioRef.current) {
          audioRef.current.src = audioObjectUrl;
          audioRef.current.addEventListener('loadedmetadata', () => {
            setAudioDuration(audioRef.current?.duration || 0);
          });
          audioRef.current.addEventListener('timeupdate', () => {
            setAudioProgress((audioRef.current?.currentTime || 0) / (audioRef.current?.duration || 1) * 100);
          });
          audioRef.current.addEventListener('ended', () => {
            setIsPlaying(false);
            setAudioProgress(0);
          });
        }
      }
    } catch (err) {
      console.error('Error loading audio for playback:', err);
    }
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleProgressChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;

    const newTime = (parseFloat(e.target.value) / 100) * audioDuration;
    audioRef.current.currentTime = newTime;
    setAudioProgress(parseFloat(e.target.value));
  };

  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const resetForm = () => {
    setSelectedFile(null);
    setJobId(null);
    setJobStatus("idle");
    setDownloadUrl(null);
    setAudioUrl(null);
    setUploadProgress(0);
    setAudioProgress(0);
    setAudioDuration(0);
    setIsPlaying(false);
    setError(null);
    setIsGenerating(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
  };

  return (
    <div className="space-y-8 min-h-screen">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle to Audio</h1>
            <p className="text-slate-400 text-sm">Generate natural-sounding audio from your subtitle files</p>
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
              {error.includes("credits") && (
                <button
                  onClick={() => window.location.href = '/dashboard/credits'}
                  className="mt-2 px-3 py-1 text-xs bg-red-500/20 border border-red-500/30 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                >
                  Buy Credits
                </button>
              )}
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-300 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </motion.div>
      )}

      {/* Upload Zone */}
      <div
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !selectedFile && !isGenerating && fileInputRef.current?.click()}
        className="relative"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept=".srt,.vtt,.ass,.ssa,.txt"
          className="hidden"
          disabled={isGenerating}
        />

        <motion.div
          whileHover={!selectedFile && !isGenerating ? { scale: 1.01 } : {}}
          className={`glass-panel glass-panel-high relative border-2 border-dashed transition-all mb-6 overflow-hidden
            ${selectedFile ? 'border-green-500/50 cursor-default' :
              isGenerating ? 'border-primary-purple/30 cursor-wait' :
                'border-primary-purple/30 hover:border-primary-purple/50 cursor-pointer'}`}
        >
          <div className="glass-shine" />
          <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

          <div className="relative z-20 py-12 px-6">
            {selectedFile ? (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-green-500/10 border border-green-500/30 shadow-glow">
                  <FileText className="w-8 h-8 text-green-500" />
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-green">{selectedFile.name}</h3>
                  <p className="text-slate-400 text-sm">
                    {(selectedFile.size / 1024).toFixed(2)} KB • Ready to process
                  </p>
                </div>
                {!isGenerating && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemoveFile();
                    }}
                    className="mt-4 px-4 py-2 text-sm bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors flex items-center gap-2"
                  >
                    <X className="w-4 h-4" />
                    Remove File
                  </button>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className={`flex items-center justify-center w-16 h-16 rounded-2xl ${isGenerating ? 'bg-primary-purple/20' : 'bg-primary-purple/10'} border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform`}>
                  {isGenerating ? (
                    <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
                  ) : (
                    <Upload className="w-8 h-8 text-primary-purple-bright" />
                  )}
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                    {isGenerating ? 'Processing...' : 'Drop your subtitle file here'}
                  </h3>
                  <p className="text-slate-400 text-sm">
                    {isGenerating ? 'Audio generation in progress...' : 'or click to browse files • SRT, VTT, ASS, SSA, TXT supported'}
                  </p>
                  <p className="text-slate-500 text-xs mt-2">Max file size: 10MB</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Progress Bar */}
      {isGenerating && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-4 mb-6"
        >
          <div className="flex justify-between text-sm mb-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${jobStatus === "uploading" ? 'bg-blue-500 animate-pulse' :
                  jobStatus === "processing" ? 'bg-yellow-500 animate-pulse' :
                    'bg-green-500'
                }`}></div>
              <span className="text-gray-400">
                {jobStatus === "uploading" && "Uploading subtitle file..."}
                {jobStatus === "processing" && "Generating audio..."}
                {jobStatus === "completed" && "Completed!"}
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
          {jobId && (
            <p className="text-slate-500 text-xs mt-2">Job ID: {jobId}</p>
          )}
        </motion.div>
      )}

      {/* Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block flex items-center gap-2">
            <span>Original Language</span>
            <Info className="w-3 h-3 text-slate-500" />
          </label>
          <select
            className="glass-select w-full"
            value={sourceLanguage}
            onChange={(e) => setSourceLanguage(e.target.value)}
            disabled={isGenerating}
          >
            {languageOptions.map((lang) => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
          <p className="text-slate-500 text-xs mt-2">Language of the subtitle file</p>
        </div>

        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block flex items-center gap-2">
            <span>Target Language</span>
            <Info className="w-3 h-3 text-slate-500" />
          </label>
          <select
            className="glass-select w-full"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            disabled={isGenerating}
          >
            {languageOptions.map((lang) => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
          <p className="text-slate-500 text-xs mt-2">Language for generated audio</p>
        </div>

        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block flex items-center gap-2">
            <span>Select Voice</span>
            <Info className="w-3 h-3 text-slate-500" />
          </label>
          <div className="flex items-center gap-2 mb-3">
            <Volume2 className="w-4 h-4 text-primary-purple-bright" />
            <span className="text-slate-300 text-sm">{voice}</span>
          </div>
          <select
            className="glass-select w-full mb-3"
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            disabled={isGenerating || availableVoices.length === 0}
          >
            {availableVoices.map((voiceOption) => (
              <option key={voiceOption.value} value={voiceOption.value}>{voiceOption.label}</option>
            ))}
            {availableVoices.length === 0 && (
              <option>Loading voices...</option>
            )}
          </select>
          <textarea
            value={previewText}
            onChange={(e) => setPreviewText(e.target.value)}
            className="glass-input w-full h-16 p-2 text-sm mb-3"
            placeholder="Enter preview text..."
            disabled={isGenerating}
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
                disabled={isGenerating}
                className="flex items-center gap-2 px-4 py-2 bg-primary-purple/20 hover:bg-primary-purple/30 text-primary-purple-bright rounded-lg text-sm transition-colors"
              >
                <Play className="w-4 h-4" />
                Preview Voice
              </button>
            )}
          </div>
        </div>

        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block">Output Format</label>
          <select
            className="glass-select w-full"
            value={outputFormat}
            onChange={(e) => setOutputFormat(e.target.value)}
            disabled={isGenerating}
          >
            <option value="mp3">MP3 (Recommended)</option>
            <option value="wav">WAV (High Quality)</option>
            <option value="ogg">OGG (Compressed)</option>
          </select>
          <p className="text-slate-500 text-xs mt-2">Audio file format</p>
        </div>
      </div>

      {/* AI Orchestration Info */}
      <div className="glass-panel glass-panel-glow mb-6 p-5 relative overflow-hidden">
        <div className="glass-shine" />
        <div className="relative z-10 flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-primary-purple-bright" />
          <div>
            <h3 className="text-white text-sm font-bold mb-1">AI Orchestration Process</h3>
            <p className="text-slate-400 text-xs">
              {targetLanguage !== sourceLanguage
                ? `1. Parse subtitle file → 2. LLM translation (TranslateGemma/NLLB) → 3. Generate audio with ${voice} voice (edge-tts) → 4. Sync timing to match subtitle cues`
                : `1. Parse subtitle file → 2. Generate audio with "${voice}" voice (edge-tts) → 3. Sync timing to match subtitle cues`}
            </p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        {jobStatus === "completed" ? (
          <>
            <button
              onClick={handleDownload}
              className="btn-border-beam w-full sm:w-auto group bg-green-500/10 border-green-500/30 hover:bg-green-500/20 transition-all duration-300"
              disabled={!downloadUrl}
            >
              <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                <Download className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Download Audio</span>
              </div>
            </button>
            <button
              onClick={resetForm}
              className="btn-border-beam w-full sm:w-auto group bg-primary-purple/10 border-primary-purple/30 hover:bg-primary-purple/20 transition-all duration-300"
            >
              <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                <FileText className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Process Another File</span>
              </div>
            </button>
          </>
        ) : (
          <button
            onClick={handleGenerateAudio}
            disabled={!selectedFile || isGenerating || (user?.credits || 0) < 5}
            className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed bg-primary-purple/10 border-primary-purple/30 hover:bg-primary-purple/20 transition-all duration-300"
          >
            <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <AudioWaveform className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                  <span>
                    {(user?.credits || 0) < 5 ? 'Insufficient Credits' : 'Generate Audio'}
                    {(user?.credits || 0) >= 5 && ` (5 credits)`}
                  </span>
                </>
              )}
            </div>
          </button>
        )}
      </div>

      {/* Audio Player */}
      {jobStatus === "completed" && audioUrl && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-6 mb-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-white font-semibold flex items-center gap-2">
              <Volume2 className="w-5 h-5 text-primary-purple-bright" />
              Preview Audio
            </h3>
            <div className="text-slate-400 text-sm">
              {formatTime(audioRef.current?.currentTime || 0)} / {formatTime(audioDuration)}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={togglePlayback}
              className="flex items-center justify-center w-12 h-12 rounded-full bg-primary-purple/20 border border-primary-purple/30 hover:bg-primary-purple/30 transition-colors"
            >
              {isPlaying ? (
                <Pause className="w-6 h-6 text-primary-purple-bright" />
              ) : (
                <Play className="w-6 h-6 text-primary-purple-bright ml-1" />
              )}
            </button>

            <div className="flex-1">
              <input
                type="range"
                min="0"
                max="100"
                value={audioProgress}
                onChange={handleProgressChange}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
                style={{
                  background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${audioProgress}%, #374151 ${audioProgress}%, #374151 100%)`
                }}
              />
            </div>
          </div>

          <p className="text-slate-500 text-xs mt-2">
            Click play to preview your generated audio before downloading
          </p>
        </motion.div>
      )}

      {/* Status Message */}
      {jobStatus === "completed" && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel border-green-500/30 bg-green-500/10 p-4"
        >
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-green-500/20 border border-green-500/30 flex items-center justify-center">
                <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
            <div>
              <h3 className="text-white font-semibold">Audio Generation Complete!</h3>
              <p className="text-green-400 text-sm">Your audio has been successfully generated. Click the download button to get your file.</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions & Technical Details */}
      {!isGenerating && jobStatus !== "completed" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-8">
          <div className="glass-card p-4 lg:col-span-2">
            <h3 className="text-white font-bold text-lg mb-3 flex items-center gap-2">
              <AudioWaveform className="w-5 h-5 text-primary-purple-bright" />
              Neural Sync Workflow
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { title: "Temporal Parsing", desc: "Native parsing of timestamps for precise audio placement." },
                { title: "Smart Translation", desc: "LLM-powered translation with context preservation." },
                { title: "Voice Synthesis", desc: "High-fidelity generation via edge-tts with natural intonation." },
                { title: "Audio Mastering", desc: "Final mixing and synchronized audio-visual alignment." }
              ].map((item, i) => (
                <div key={i} className="bg-white/5 p-3 rounded-lg border border-white/5">
                  <h4 className="text-white font-bold text-sm mb-1">{item.title}</h4>
                  <p className="text-slate-500 text-xs leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="glass-card p-4 border-indigo-500/10 h-min">
            <h3 className="text-white font-bold text-lg mb-3">Sync Specs</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />
                <p>Perfectly synced with subtitle timestamps.</p>
              </li>
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />
                <p>Costs 5 credits per generation.</p>
              </li>
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-green-500 shrink-0" />
                <p>Secure 24-hour file deletion.</p>
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* Hidden Audio Element */}
      <audio ref={audioRef} preload="metadata" />
    </div>
  );
}
