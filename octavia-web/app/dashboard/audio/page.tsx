"use client";

import { motion } from "framer-motion";
import { AudioLines, Play, Loader2, Upload, FileAudio, X, Download, CheckCircle } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { api, ApiResponse } from "@/lib/api"; // Import your API service
import { useUser } from "@/contexts/UserContext";
import { useToast } from "@/hooks/use-toast";

export default function AudioTranslationPage() {
  const { toast } = useToast();
  const { user, refreshCredits } = useUser();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [jobStatus, setJobStatus] = useState<string>("idle");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingStage, setProcessingStage] = useState<string>("idle");
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [audioPlaybackSupported, setAudioPlaybackSupported] = useState(true);

  // Polling interval reference
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Refresh credits when component mounts
  useEffect(() => {
    refreshCredits();

    // Check if browser supports audio playback
    const tempAudio = document.createElement('audio');
    const supportsWav = tempAudio.canPlayType('audio/wav') !== '' ||
      tempAudio.canPlayType('audio/wave') !== '';
    setAudioPlaybackSupported(supportsWav);
    console.log(`Browser WAV support: ${supportsWav}`);
  }, []);

  // Check for project context on mount
  useEffect(() => {
    const projectContext = localStorage.getItem('octavia_project_context');
    if (projectContext) {
      try {
        const context = JSON.parse(projectContext);
        if (context.fileUrl && context.projectType === 'Audio Translation') {
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

  // Add test credits for development
  const handleAddTestCredits = async () => {
    try {
      const response = await api.addTestCredits(100);
      if (response.success && response.data) {
        refreshCredits();
        setError(null);
      } else {
        setError("Failed to add test credits");
      }
    } catch (error) {
      console.error('Add test credits error:', error);
      setError("Failed to add test credits");
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/x-flac', 'audio/ogg', 'audio/x-m4a', 'audio/x-aac'];
    const fileType = file.type;

    // Also check file extension
    const validExtensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.aac'];
    const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!validTypes.includes(fileType) && !validExtensions.includes(fileExt)) {
      setError("Please upload MP3, WAV, FLAC, OGG, or M4A files only.");
      return;
    }

    // Check file size (max 100MB)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      setError("File size too large. Maximum size is 100MB.");
      return;
    }

    setSelectedFile(file);
    setError(null);
  };

  // Handle drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    const file = e.dataTransfer.files[0];

    if (file) {
      const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/x-flac', 'audio/ogg', 'audio/x-m4a'];
      const fileType = file.type;

      const validExtensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.mp4'];
      const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

      if (!validTypes.includes(fileType) && !validExtensions.includes(fileExt)) {
        setError("Please upload MP3, WAV, FLAC, OGG, or M4A files only.");
        return;
      }

      const maxSize = 100 * 1024 * 1024;
      if (file.size > maxSize) {
        setError("File size too large. Maximum size is 100MB.");
        return;
      }

      setSelectedFile(file);
      setError(null);
    }
  };

  // Start audio translation using the API service
  const handleUpload = async () => {
    if (!selectedFile) {
      setError("No audio file selected. Please choose an audio file to translate.");
      return;
    }

    // Check if user has enough credits (10 credits for audio translation)
    if ((user?.credits || 0) < 10) {
      setError(`Insufficient credits. You need 10 credits but only have ${user?.credits || 0}. Please purchase more credits.`);
      return;
    }

    setIsProcessing(true);
    setProcessingStage("uploading");
    setProgress(10);
    setJobStatus("uploading");
    setError(null);

    try {
      console.log('Starting audio translation:', selectedFile.name);
      console.log('Target language:', targetLanguage);

      // Use the API service to translate audio
      const response = await api.translateAudio({
        file: selectedFile,
        sourceLanguage,
        targetLanguage,
        projectId: projectId || undefined
      });

      console.log('API Response:', response);

      if (!response.success) {
        throw new Error(response.error || response.message || "Failed to start translation job");
      }

      if (response.job_id) {
        setJobId(response.job_id);
        setProgress(30);
        setProcessingStage("transcribing");
        setJobStatus("processing");

        // Start polling for job status
        startPollingJobStatus(response.job_id);

        // Add job to project in localStorage if projectId exists
        if (projectId) {
          const storedJobs = localStorage.getItem(`octavia_project_jobs_${projectId}`);
          const projectJobs = storedJobs ? JSON.parse(storedJobs) : [];

          const newJob = {
            id: response.job_id,
            type: 'audio',
            status: 'processing',
            progress: 0,
            created_at: new Date().toISOString(),
            target_language: targetLanguage,
            source_language: sourceLanguage
          };

          if (!projectJobs.find((j: any) => j.id === response.job_id)) {
            const updatedJobs = [newJob, ...projectJobs];
            localStorage.setItem(`octavia_project_jobs_${projectId}`, JSON.stringify(updatedJobs));

            window.dispatchEvent(new StorageEvent('storage', {
              key: `octavia_project_jobs_${projectId}`,
              newValue: JSON.stringify(updatedJobs)
            }));
          }
        }

        // Refresh user credits to update the balance
        refreshCredits();
      } else {
        throw new Error(response.error || response.message || "Failed to get job ID");
      }
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.message || "Failed to upload audio file");
      setIsProcessing(false);
      setProgress(0);
      setJobStatus("idle");
      setProcessingStage("error");
    }
  };

  // Start polling for job status
  const startPollingJobStatus = (jobId: string) => {
    // Clear any existing polling
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }

    let pollCount = 0;
    const maxPolls = 300; // Maximum 10 minutes (300 * 2 seconds)

    const poll = async () => {
      pollCount++;

      try {
        console.log(`Polling job ${jobId} (attempt ${pollCount}/${maxPolls})`);

        const response = await api.getJobStatus(jobId);

        if (response.success) {
          // The job data is directly in the response, not in response.data
          const jobData = response.data || response; // Fallback to response if data is not present
          const job = {
            job_id: jobData.job_id,
            status: jobData.status,
            progress: jobData.progress || 0,
            download_url: jobData.download_url,
            error: jobData.error,
            message: (jobData as any).status_message || (jobData as { message?: string }).message
          };

          console.log(`Job ${jobId} status: ${job.status}, progress: ${job.progress}`);

          const jobProgress = job.progress || 0;

          // Update progress based on stage
          let calculatedProgress = 30;
          if (job.status === "processing") {
            // Simulate stages: 30-60% transcription, 60-80% translation, 80-95% synthesis
            if (jobProgress < 33) {
              setProcessingStage("transcribing");
              calculatedProgress = 30 + (jobProgress * 0.3);
            } else if (jobProgress < 66) {
              setProcessingStage("translating");
              calculatedProgress = 40 + ((jobProgress - 33) * 0.3);
            } else {
              setProcessingStage("synthesizing");
              calculatedProgress = 50 + ((jobProgress - 66) * 0.5);
            }
          }

          setProgress(Math.min(calculatedProgress, 99));

          if (job.status === "completed") {
            console.log(`Job ${jobId} completed successfully!`);

            setJobStatus("completed");
            setProgress(100);
            setProcessingStage("delivery");

            // Set download URL
            const downloadUrl = job.download_url || `/api/download/audio/${jobId}`;
            setDownloadUrl(downloadUrl);
            setIsProcessing(false);

            // Clear polling
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }

            console.log('Job completed! Download URL:', downloadUrl);
            return; // Stop polling
          } else if (job.status === "failed") {
            console.log(`Job ${jobId} failed: ${job.error}`);

            setJobStatus("failed");
            setProcessingStage("error");
            setError(job.error || "Translation failed");
            setIsProcessing(false);

            // Clear polling
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }

            return; // Stop polling
          }
        } else {
          console.warn(`Job ${jobId} polling failed:`, response);
        }

        // Check if we've exceeded max polls
        if (pollCount >= maxPolls) {
          console.error(`Job ${jobId} polling timed out after ${maxPolls} attempts`);

          setJobStatus("failed");
          setProcessingStage("error");
          setError("Translation timed out. Please try again.");
          setIsProcessing(false);

          // Clear polling
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          return; // Stop polling
        }

      } catch (err: any) {
        console.error(`Polling error for job ${jobId}:`, err);

        // If we get 404 (job not found), stop polling
        if (err.message && err.message.includes('404')) {
          console.log(`Job ${jobId} not found, stopping polling`);

          setJobStatus("failed");
          setProcessingStage("error");
          setError("Job not found. Please try starting a new translation.");
          setIsProcessing(false);

          // Clear polling
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          return; // Stop polling
        }
      }
    };

    // Start polling every 2 seconds
    pollingIntervalRef.current = setInterval(poll, 2000);

    // Initial poll
    poll();
  };

  // Download translated audio
  const handleDownload = async () => {
    if (!jobId) {
      setError("No job ID found. Please start a translation first.");
      return;
    }

    try {
      console.log('Downloading audio for job:', jobId);

      // For audio jobs, try audio-specific endpoints first
      const token = localStorage.getItem('octavia_user') ? JSON.parse(localStorage.getItem('octavia_user')!).token : null;
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      // Try audio-specific endpoints first
      const audioEndpoints = [
        `/api/download/audio/${jobId}`,
        `/api/download/${jobId}`,
        `/api/download/video/${jobId}` // fallback to video endpoint
      ];

      let blob: Blob | null = null;
      let lastError: Error | null = null;

      for (const endpoint of audioEndpoints) {
        const url = `${API_BASE_URL}${endpoint}`;

        try {
          console.log(`Trying audio download endpoint: ${endpoint}`);
          const response = await fetch(url, {
            headers: token ? {
              'Authorization': `Bearer ${token}`
            } : {},
          });

          if (response.ok) {
            blob = await response.blob();
            console.log(`Audio download successful from: ${endpoint}`);
            break;
          }

          // If 404, try next endpoint
          if (response.status === 404) {
            console.log(`Endpoint ${endpoint} returned 404, trying next...`);
            continue;
          }

          // For other errors, capture but continue trying other endpoints
          const errorText = await response.text();
          const error = new Error(`Download failed (${response.status}): ${errorText}`);
          console.warn(`Endpoint ${endpoint} failed:`, error);
          lastError = error;

        } catch (error) {
          console.warn(`Endpoint ${endpoint} network error:`, error);
          lastError = error as Error;
        }
      }

      if (!blob) {
        throw lastError || new Error('Download failed: All endpoints failed');
      }

      // Extract filename
      const fileName = `octavia_translated_audio_${jobId}.mp3`;

      // Save the file
      api.saveFile(blob, fileName);

      console.log('Audio file downloaded:', fileName);

    } catch (err: any) {
      console.error('Download error:', err);
      setError("Failed to download audio file. Please try again or contact support.");
    }
  };

  // Remove selected file
  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Handle audio play/pause
  const handlePlayPause = async () => {
    if (!jobId) return;

    // Don't attempt playback if not supported
    if (!audioPlaybackSupported) {
      setError("Audio playback is not supported in your browser. Please download the file instead.");
      return;
    }

    if (isPlaying && audioElement) {
      audioElement.pause();
      setIsPlaying(false);
    } else {
      try {
        let audioToPlay = audioElement;

        if (!audioToPlay) {
          // Download the audio file as a blob first
          console.log('Downloading audio file for playback...');

          // For audio jobs, try audio-specific endpoints first
          const token = localStorage.getItem('octavia_user') ? JSON.parse(localStorage.getItem('octavia_user')!).token : null;
          const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

          // Try audio-specific endpoints first
          const audioEndpoints = [
            `/api/download/audio/${jobId}`,
            `/api/download/${jobId}`,
            `/api/download/video/${jobId}` // fallback to video endpoint
          ];

          let audioBlob: Blob | null = null;
          let lastError: Error | null = null;

          for (const endpoint of audioEndpoints) {
            const url = `${API_BASE_URL}${endpoint}`;

            try {
              console.log(`Trying audio download endpoint: ${endpoint}`);
              const response = await fetch(url, {
                headers: token ? {
                  'Authorization': `Bearer ${token}`
                } : {},
              });

              if (response.ok) {
                audioBlob = await response.blob();
                console.log(`Audio download successful from: ${endpoint}`);
                break;
              }

              // If 404, try next endpoint
              if (response.status === 404) {
                console.log(`Endpoint ${endpoint} returned 404, trying next...`);
                continue;
              }

              // For other errors, capture but continue trying other endpoints
              const errorText = await response.text();
              const error = new Error(`Download failed (${response.status}): ${errorText}`);
              console.warn(`Endpoint ${endpoint} failed:`, error);
              lastError = error;

            } catch (error) {
              console.warn(`Endpoint ${endpoint} network error:`, error);
              lastError = error as Error;
            }
          }

          if (!audioBlob) {
            throw lastError || new Error('Download failed: All endpoints failed');
          }

          // Validate blob
          if (!audioBlob || audioBlob.size === 0) {
            throw new Error('Downloaded audio file is empty or invalid');
          }

          console.log(`Downloaded audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`);

          // Ensure correct MIME type for audio playback
          // Backend generates WAV files, so set appropriate MIME type
          let correctedBlob = audioBlob;
          if (!audioBlob.type || !audioBlob.type.startsWith('audio/')) {
            // Check if browser supports WAV playback
            const tempAudio = document.createElement('audio');
            const canPlayWav = tempAudio.canPlayType('audio/wav') !== '' ||
              tempAudio.canPlayType('audio/wave') !== '';

            if (canPlayWav) {
              correctedBlob = new Blob([audioBlob], { type: 'audio/wav' });
              console.log(`Set blob type to: audio/wav`);
            } else {
              console.warn('Browser does not support WAV playback, audio may not play');
              // Still try with wav type
              correctedBlob = new Blob([audioBlob], { type: 'audio/wav' });
            }
          }

          // Create an object URL from the corrected blob
          const audioUrl = URL.createObjectURL(correctedBlob);

          const audio = new Audio(audioUrl);

          // Set up event listeners
          audio.addEventListener('ended', () => {
            setIsPlaying(false);
            console.log('Audio playback ended');
          });

          audio.addEventListener('error', (e) => {
            console.error('Audio playback error:', e);
            console.error('Audio error details:', {
              error: (e.target as HTMLAudioElement).error,
              src: (e.target as HTMLAudioElement).src,
              networkState: (e.target as HTMLAudioElement).networkState,
              readyState: (e.target as HTMLAudioElement).readyState
            });
            const audioError = e.target as HTMLAudioElement;
            let errorMessage = "Failed to play audio. ";

            // Provide more specific error messages
            if (audioError.error) {
              switch (audioError.error.code) {
                case MediaError.MEDIA_ERR_ABORTED:
                  errorMessage += "Playback was aborted.";
                  break;
                case MediaError.MEDIA_ERR_NETWORK:
                  errorMessage += "Network error occurred.";
                  break;
                case MediaError.MEDIA_ERR_DECODE:
                  errorMessage += "Audio format not supported or corrupted.";
                  break;
                case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
                  errorMessage += "Audio format not supported by your browser.";
                  break;
                default:
                  errorMessage += `Error code: ${audioError.error.code}.`;
              }
            } else {
              errorMessage += "Browser could not load the audio file.";
            }

            setError(errorMessage + " Please try downloading the file instead.");
            setAudioPlaybackSupported(false); // Disable playback attempts
            setIsPlaying(false);
            setAudioPlaybackSupported(false); // Disable playback for this job

            // Clean up the object URL
            URL.revokeObjectURL(audioUrl);
          });

          audio.addEventListener('canplay', () => {
            console.log('Audio can play successfully');
          });

          audio.addEventListener('loadstart', () => {
            console.log('Audio load started');
          });

          setAudioElement(audio);
          audioToPlay = audio;
        }

        // Validate audio element before playing
        if (!audioToPlay.src) {
          throw new Error('Audio source not set');
        }

        console.log('Attempting to play audio...');
        await audioToPlay.play();
        setIsPlaying(true);
        console.log('Audio playback started successfully');

      } catch (err: any) {
        console.error('Audio play error:', err);
        console.error('Error details:', {
          name: err.name,
          message: err.message,
          stack: err.stack
        });
        let errorMessage = "Failed to play audio. ";

        if (err.name === 'NotSupportedError' || err.message?.includes('NotSupportedError')) {
          errorMessage += "Audio format not supported by your browser. This may be due to an incompatible WAV format.";
        } else if (err.name === 'NotAllowedError') {
          errorMessage += "Playback blocked by browser. Please interact with the page first.";
        } else if (err.message?.includes('no supported source')) {
          errorMessage += "Audio format not recognized by your browser.";
        } else if (err.message) {
          errorMessage += err.message;
        } else {
          errorMessage += "Unknown error occurred.";
        }

        setError(errorMessage + " Please try downloading the file and playing it with a media player.");
      }
    }
  };

  // Reset form for new translation
  const handleReset = () => {
    // Stop and cleanup audio
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
      // Clean up the object URL if it exists
      if (audioElement.src && audioElement.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElement.src);
      }
      setAudioElement(null);
    }
    setIsPlaying(false);

    setSelectedFile(null);
    setJobId(null);
    setProgress(0);
    setJobStatus("idle");
    setProcessingStage("idle");
    setDownloadUrl(null);
    setError(null);
    setIsProcessing(false);
    setAudioPlaybackSupported(true); // Reset for new job
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // Clear any polling interval
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  // Get stage description
  const getStageDescription = () => {
    switch (processingStage) {
      case "uploading":
        return "Uploading audio file...";
      case "transcribing":
        return "Transcribing audio with OpenAI Whisper...";
      case "translating":
        return "Translating text with Helsinki NLP...";
      case "synthesizing":
        return "Generating audio with edge-tts...";
      case "delivery":
        return "Preparing download...";
      case "error":
        return "Error occurred";
      default:
        return "Ready to translate";
    }
  };

  const languageOptions = [
    { value: "en", label: "English" },
    { value: "es", label: "Spanish" },
    { value: "fr", label: "French" },
    { value: "de", label: "German" },
    { value: "it", label: "Italian" },
    { value: "pt", label: "Portuguese" },
    { value: "ru", label: "Russian" },
    { value: "ja", label: "Japanese" },
    { value: "ko", label: "Korean" },
    { value: "zh", label: "Chinese" },
    { value: "ar", label: "Arabic" },
    { value: "hi", label: "Hindi" },
  ];

  return (
    <div className="space-y-8 min-h-screen">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="font-display text-3xl font-black text-white text-glow-purple">Audio Translation</h1>
            <p className="text-slate-400 text-sm">Transform audio files across languages with AI-powered voice synthesis</p>
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
        onClick={() => !selectedFile && !isProcessing && fileInputRef.current?.click()}
        className="relative"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.mp4,.aac"
          className="hidden"
          disabled={isProcessing}
        />

        <motion.div
          whileHover={!selectedFile && !isProcessing ? { scale: 1.01 } : {}}
          className={`glass-panel glass-panel-high relative border-2 border-dashed transition-all mb-6 overflow-hidden
            ${selectedFile ? 'border-accent-cyan/50 cursor-default' :
              isProcessing ? 'border-primary-purple/30 cursor-wait' :
                'border-primary-purple/30 hover:border-primary-purple/50 cursor-pointer'}`}
        >
          <div className="glass-shine" />
          <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

          <div className="relative z-20 py-12 px-6">
            {selectedFile ? (
              <div className="flex flex-col items-center justify-center gap-3 text-center">
                <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-accent-cyan/10 border border-accent-cyan/30 shadow-glow">
                  <FileAudio className="w-8 h-8 text-accent-cyan" />
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-accent-cyan">{selectedFile.name}</h3>
                  <p className="text-slate-400 text-sm">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to translate
                  </p>
                </div>
                {!isProcessing && (
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
                <div className={`flex items-center justify-center w-16 h-16 rounded-2xl ${isProcessing ? 'bg-primary-purple/20' : 'bg-primary-purple/10'} border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform`}>
                  {isProcessing ? (
                    <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
                  ) : (
                    <Upload className="w-8 h-8 text-primary-purple-bright" />
                  )}
                </div>
                <div>
                  <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                    {isProcessing ? 'Processing...' : 'Drop your audio file here'}
                  </h3>
                  <p className="text-slate-400 text-sm">
                    {isProcessing ? 'Translation in progress...' : 'or click to browse files • MP3, WAV, FLAC, OGG, M4A supported'}
                  </p>
                  <p className="text-slate-500 text-xs mt-2">Max file size: 100MB</p>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block">Source Language</label>
          <select
            value={sourceLanguage}
            onChange={(e) => setSourceLanguage(e.target.value)}
            className="glass-select w-full"
            disabled={isProcessing}
          >
            <option value="auto">Auto-detect</option>
            {languageOptions.map((lang) => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
          <p className="text-slate-500 text-xs mt-2">Language of the original audio</p>
        </div>

        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block">Target Language</label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            className="glass-select w-full"
            disabled={isProcessing}
          >
            {languageOptions.map((lang) => (
              <option key={lang.value} value={lang.value}>{lang.label}</option>
            ))}
          </select>
          <p className="text-slate-500 text-xs mt-2">Language to translate to</p>
        </div>
      </div>

      {/* Progress Tracking */}
      {isProcessing && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-4 mb-6"
        >
          <div className="flex justify-between text-sm mb-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${processingStage === "uploading" ? 'bg-blue-500 animate-pulse' :
                processingStage === "transcribing" ? 'bg-yellow-500 animate-pulse' :
                  processingStage === "translating" ? 'bg-purple-500 animate-pulse' :
                    processingStage === "synthesizing" ? 'bg-pink-500 animate-pulse' :
                      processingStage === "delivery" ? 'bg-accent-cyan animate-pulse' :
                        'bg-gray-500'
                }`}></div>
              <span className="text-gray-400">
                {getStageDescription()}
              </span>
            </div>
            <span className="text-white font-bold">{Math.round(progress)}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright"
              initial={{ width: "0%" }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>

          {/* Stage breakdown */}
          <div className="grid grid-cols-4 gap-2 mt-4">
            <div className={`text-center p-2 rounded ${processingStage === "uploading" ? 'bg-blue-500/20 border border-blue-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">1. Upload</div>
              <div className={`text-xs ${processingStage === "uploading" ? 'text-blue-400' : 'text-gray-500'}`}>Audio File</div>
            </div>
            <div className={`text-center p-2 rounded ${processingStage === "transcribing" ? 'bg-yellow-500/20 border border-yellow-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">2. Transcribe</div>
              <div className={`text-xs ${processingStage === "transcribing" ? 'text-yellow-400' : 'text-gray-500'}`}>OpenAI Whisper</div>
            </div>
            <div className={`text-center p-2 rounded ${processingStage === "translating" ? 'bg-purple-500/20 border border-purple-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">3. Translate</div>
              <div className={`text-xs ${processingStage === "translating" ? 'text-purple-400' : 'text-gray-500'}`}>LLM Models</div>
            </div>
            <div className={`text-center p-2 rounded ${processingStage === "synthesizing" || processingStage === "delivery" ? 'bg-pink-500/20 border border-pink-500/30' : 'bg-gray-500/10'}`}>
              <div className="text-xs text-gray-400">4. Synthesize</div>
              <div className={`text-xs ${processingStage === "synthesizing" || processingStage === "delivery" ? 'text-pink-400' : 'text-gray-500'}`}>edge-tts / gTTS</div>
            </div>
          </div>

          {jobId && (
            <p className="text-slate-500 text-xs mt-2">Job ID: {jobId}</p>
          )}
        </motion.div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4">
        {jobStatus === "completed" ? (
          <>
            {audioPlaybackSupported ? (
              <button
                onClick={handlePlayPause}
                className="btn-border-beam w-full sm:w-auto group bg-blue-500/10 border-blue-500/30 hover:bg-blue-500/20 transition-all duration-300"
                disabled={!downloadUrl}
              >
                <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                  {isPlaying ? (
                    <>
                      <div className="w-1 h-4 bg-blue-400 mr-1"></div>
                      <div className="w-1 h-4 bg-blue-400"></div>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                      <span>Play Translated Audio</span>
                    </>
                  )}
                </div>
              </button>
            ) : (
              <div className="w-full sm:w-auto px-4 py-2 bg-gray-500/10 border border-gray-500/30 rounded-lg text-center">
                <div className="flex items-center justify-center gap-2 text-sm text-gray-400">
                  <div className="w-4 h-4 rounded-full bg-gray-500/20 flex items-center justify-center">
                    <span className="text-xs">!</span>
                  </div>
                  <span>Audio playback not supported in browser</span>
                </div>
              </div>
            )}
            <button
              onClick={handleDownload}
              className="btn-border-beam w-full sm:w-auto group bg-accent-cyan/10 border-accent-cyan/30 hover:bg-accent-cyan/20 transition-all duration-300"
              disabled={!downloadUrl}
            >
              <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                <Download className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Download Translated Audio</span>
              </div>
            </button>
            <button
              onClick={handleReset}
              className="btn-border-beam w-full sm:w-auto group bg-primary-purple/10 border-primary-purple/30 hover:bg-primary-purple/20 transition-all duration-300"
            >
              <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
                <AudioLines className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Translate Another File</span>
              </div>
            </button>
          </>
        ) : (
          <button
            onClick={handleUpload}
            disabled={!selectedFile || isProcessing || (user?.credits || 0) < 10}
            className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed bg-primary-purple/10 border-primary-purple/30 hover:bg-primary-purple/20 transition-all duration-300"
          >
            <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 group-hover:rotate-12 transition-transform duration-300" />
                  <span>
                    {(user?.credits || 0) < 10 ? 'Insufficient Credits' : 'Start Audio Translation'}
                    {(user?.credits || 0) >= 10 && ` (10 credits)`}
                  </span>
                </>
              )}
            </div>
          </button>
        )}
      </div>

      {/* Status Message */}
      {jobStatus === "completed" && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-panel border-accent-cyan/30 bg-accent-cyan/10 p-4"
        >
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-accent-cyan/20 border border-accent-cyan/30 flex items-center justify-center">
                <svg className="w-5 h-5 text-accent-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
            <div>
              <h3 className="text-white font-semibold">Translation Complete!</h3>
              <p className="text-accent-cyan text-sm">Your audio has been successfully translated. Play it here or download the file.</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Instructions & Technical Details */}
      {!isProcessing && jobStatus !== "completed" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-8">
          <div className="glass-card p-4 lg:col-span-2">
            <h3 className="text-white font-bold text-lg mb-3 flex items-center gap-2">
              <AudioLines className="w-5 h-5 text-primary-purple-bright" />
              Neural Audio Workflow
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                { title: "Speech Recognition", desc: "Faster Whisper decodes audio to text." },
                { title: "Smart Translation", desc: "Context-aware LLM refinement." },
                { title: "Synthesized Audio", desc: "gTTS & Edge-TTS high-fidelity audio." },
                { title: "Audio Mastering", desc: "Final audio mixing and normalization." }
              ].map((item, i) => (
                <div key={i} className="bg-white/5 p-3 rounded-lg border border-white/5">
                  <h4 className="text-white font-bold text-sm mb-1">{item.title}</h4>
                  <p className="text-slate-500 text-xs leading-relaxed">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="glass-card p-4 border-indigo-500/10 h-min">
            <h3 className="text-white font-bold text-lg mb-3">Audio Specs</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-accent-cyan shrink-0" />
                <p>High-fidelity output (MP3/WAV).</p>
              </li>
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-accent-cyan shrink-0" />
                <p>Costs 10 credits per generation.</p>
              </li>
              <li className="flex items-start gap-2 text-xs text-slate-400">
                <CheckCircle className="w-4 h-4 text-accent-cyan shrink-0" />
                <p>Secure 24-hour file deletion.</p>
              </li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}


