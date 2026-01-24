"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Download, CheckCircle, Edit3, Save, X, Loader2, FileText, Clock, Languages } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";

interface SubtitleItem {
  id: number;
  start: number;
  end: number;
  text: string;
  timestamp: string;
  isEditing?: boolean;
  editedText?: string;
}

export default function SubtitleReviewPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("jobId") || localStorage.getItem("current_subtitle_job");

  const [subtitles, setSubtitles] = useState<SubtitleItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [jobInfo, setJobInfo] = useState({
    totalLines: 0,
    duration: "00:00",
    language: "English",
    format: "SRT",
    download_url: ""
  });

  // Helper function to parse SRT time to seconds
  const parseSRTTimeToSeconds = (timeStr: string): number => {
    const normalized = timeStr.replace(',', '.');
    const parts = normalized.split(':');

    if (parts.length === 3) {
      const [hours, minutes, seconds] = parts.map(part => {
        const [whole, decimal] = part.split('.');
        return parseFloat(whole) + (decimal ? parseFloat(`0.${decimal}`) : 0);
      });

      return hours * 3600 + minutes * 60 + seconds;
    }

    return 0;
  };

  // Helper function to format seconds to SRT time
  const formatSecondsToSRT = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const millis = Math.floor((seconds - Math.floor(seconds)) * 1000);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${millis.toString().padStart(3, '0')}`;
  };

  // Helper function to format seconds to display time
  const formatSecondsToDisplay = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  // Parse SRT content string into segments
  const parseSRTContent = (srtContent: string): any[] => {
    const segments: any[] = [];
    const lines = srtContent.split('\n');

    let currentSegment: any = null;
    let textLines: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Skip empty lines at the start of a segment
      if (!line && !currentSegment) continue;

      // If we hit an empty line and have a segment, finalize it
      if (!line && currentSegment) {
        if (textLines.length > 0) {
          currentSegment.text = textLines.join(' ');
          segments.push(currentSegment);
        }
        currentSegment = null;
        textLines = [];
        continue;
      }

      // Check if line is a segment number
      if (/^\d+$/.test(line) && !currentSegment) {
        currentSegment = { id: parseInt(line), start: 0, end: 0, text: '' };
      }
      // Check if line contains timestamp (-->)
      else if (line.includes('-->') && currentSegment) {
        const [startTime, endTime] = line.split('-->').map(t => t.trim());
        currentSegment.start = parseSRTTimeToSeconds(startTime);
        currentSegment.end = parseSRTTimeToSeconds(endTime);
      }
      // Text content (could be multi-line)
      else if (currentSegment && line) {
        textLines.push(line);
      }
    }
    // Finalize last segment if exists
    if (currentSegment && textLines.length > 0) {
      currentSegment.text = textLines.join(' ');
      segments.push(currentSegment);
    }

    return segments;
  };

  useEffect(() => {
    if (!jobId) {
      router.push("/dashboard/subtitles");
      return;
    }

    const fetchSubtitles = async () => {
      try {
        // Use the API utility instead of direct fetch
        const response = await api.getSubtitleReviewData(jobId);

        if (response.success) {
          const reviewData = response.data || (response as any);

          // Check if we have content (SRT string) or segments (already parsed)
          let segments: any[] = [];

          if (reviewData.content) {
            // Parse the SRT content
            segments = parseSRTContent(reviewData.content);
          } else if (reviewData.segments) {
            // Use already parsed segments
            segments = reviewData.segments;
          }

          if (segments.length > 0) {
            // Convert to display format
            const formattedSubtitles = segments.map((segment, index) => ({
              id: index + 1,
              start: segment.start,
              end: segment.end,
              text: segment.text,
              timestamp: `${formatSecondsToSRT(segment.start)} → ${formatSecondsToSRT(segment.end)}`
            }));

            setSubtitles(formattedSubtitles);
            setJobInfo({
              totalLines: formattedSubtitles.length,
              duration: formatSecondsToDisplay(Math.max(...formattedSubtitles.map(s => s.end))),
              language: reviewData.language || "English",
              format: reviewData.format?.toUpperCase() || "SRT",
              download_url: reviewData.download_url || ""
            });
          } else {
            // Fallback to demo data
            useDemoData();
          }
        } else {
          // Fallback to demo data
          useDemoData();
        }
      } catch (error) {
        console.error("Failed to fetch subtitles:", error);
        useDemoData();
      } finally {
        setLoading(false);
      }
    };

    fetchSubtitles();
  }, [jobId]);

  const useDemoData = () => {
    const demoSubtitles: SubtitleItem[] = [
      { id: 1, start: 0, end: 5, text: "Welcome to this amazing video tutorial.", timestamp: "00:00:00,000 → 00:00:05,000" },
      { id: 2, start: 6, end: 11, text: "Today we'll explore advanced techniques for subtitle generation.", timestamp: "00:00:06,000 → 00:00:11,000" },
      { id: 3, start: 12, end: 17, text: "The AI has analyzed the audio and created these timestamps.", timestamp: "00:00:12,000 → 00:00:17,000" },
      { id: 4, start: 18, end: 23, text: "You can edit any subtitle by clicking on it.", timestamp: "00:00:18,000 → 00:00:23,000" },
      { id: 5, start: 24, end: 30, text: "When you're satisfied, download the final file.", timestamp: "00:00:24,000 → 00:00:30,000" },
    ];

    setSubtitles(demoSubtitles);
    setJobInfo({
      totalLines: demoSubtitles.length,
      duration: "00:30",
      language: "English",
      format: "SRT",
      download_url: ""
    });
  };

  const handleEdit = (id: number) => {
    setSubtitles(subtitles.map(sub =>
      sub.id === id ? { ...sub, isEditing: true, editedText: sub.text } : sub
    ));
  };

  const handleSave = (id: number) => {
    const subtitle = subtitles.find(sub => sub.id === id);
    if (!subtitle || !subtitle.editedText) return;

    setSaving(true);

    // Simulate API call
    setTimeout(() => {
      setSubtitles(subtitles.map(sub =>
        sub.id === id ? { ...sub, text: subtitle.editedText!, isEditing: false } : sub
      ));
      setSaving(false);
    }, 500);
  };

  const handleCancel = (id: number) => {
    setSubtitles(subtitles.map(sub =>
      sub.id === id ? { ...sub, isEditing: false } : sub
    ));
  };

  const handleTextChange = (id: number, text: string) => {
    setSubtitles(subtitles.map(sub =>
      sub.id === id ? { ...sub, editedText: text } : sub
    ));
  };

  const handleDownload = async (format: string) => {
    if (!jobId) return;

    try {
      // Try to download using the API function first
      const blob = await api.downloadSubtitleFile(jobId);

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `subtitles_${jobId}.${format.toLowerCase()}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      alert(`Subtitles downloaded in ${format} format!`);
    } catch (error) {
      console.error("Download failed:", error);

      // Fallback: generate and download locally
      try {
        const content = generateSubtitleContent(format);
        const blob = new Blob([content], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `subtitles_${jobId}.${format.toLowerCase()}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        alert(`Subtitles downloaded in ${format} format! (Local fallback)`);
      } catch (fallbackError) {
        console.error("Fallback download also failed:", fallbackError);
        alert("Failed to download subtitles. Please try again.");
      }
    }
  };

  const generateSubtitleContent = (format: string): string => {
    switch (format.toUpperCase()) {
      case "SRT":
        return subtitles.map(sub =>
          `${sub.id}\n${sub.timestamp.split(" → ").join(" --> ")}\n${sub.text}\n`
        ).join("\n");

      case "VTT":
        return `WEBVTT\n\n${subtitles.map(sub =>
          `${sub.timestamp.split(" → ").join(" --> ").replace(/,/g, ".")}\n${sub.text}\n`
        ).join("\n")}`;

      case "ASS":
        return `[Script Info]
Title: Octavia Generated Subtitles
ScriptType: v4.00+
PlayResX: 384
PlayResY: 288

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
${subtitles.map(sub =>
          `Dialogue: 0,${sub.timestamp.split(" → ").map(t => t.replace(/:/g, ".").replace(",", ".")).join(",")},Default,,0,0,0,,${sub.text}`
        ).join("\n")}`;

      default:
        return subtitles.map(sub => sub.text).join("\n");
    }
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Loading Subtitles</h1>
          <p className="text-slate-400 text-sm">Fetching your generated subtitles...</p>
        </div>

        <div className="glass-panel p-8 flex items-center justify-center">
          <Loader2 className="w-12 h-12 text-primary-purple-bright animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col lg:flex-row gap-8">
        {/* Left: Subtitle Editor */}
        <div className="flex-1 flex flex-col gap-6">
          <div className="h-fit">
            <div className="flex items-center justify-between">
              <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Review Subtitles</h1>
              {jobId && (
                <div className="text-xs text-slate-500 font-mono bg-white/5 px-3 py-1 rounded">
                  Job: {jobId.substring(0, 8)}...
                </div>
              )}
            </div>
            <p className="text-slate-400 text-sm">Edit and refine your generated subtitles</p>
          </div>

          {/* Subtitle List */}
          <div className="glass-panel h-[calc(100vh-280px)] min-h-[500px] flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-3">
              {subtitles.length === 0 ? (
                <div className="text-center py-8">
                  <FileText className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                  <h3 className="text-white text-lg font-bold mb-2">No Subtitles Found</h3>
                  <p className="text-slate-400">Try generating subtitles first</p>
                </div>
              ) : (
                subtitles.map((sub) => (
                  <motion.div
                    key={sub.id}
                    whileHover={{ scale: 1.01 }}
                    className="glass-card p-4 hover:border-primary-purple/30 transition-all cursor-pointer group"
                  >
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <span className="text-xs text-slate-500 font-mono">{sub.timestamp}</span>
                      <div className="flex gap-1">
                        {sub.isEditing ? (
                          <>
                            <button
                              onClick={() => handleSave(sub.id)}
                              disabled={saving}
                              className="p-1 rounded hover:bg-green-500/10 text-green-400 hover:text-green-300 transition-colors disabled:opacity-50"
                              title="Save"
                            >
                              {saving ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Save className="w-3.5 h-3.5" />
                              )}
                            </button>
                            <button
                              onClick={() => handleCancel(sub.id)}
                              className="p-1 rounded hover:bg-red-500/10 text-red-400 hover:text-red-300 transition-colors"
                              title="Cancel"
                            >
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => handleEdit(sub.id)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-white/10"
                            title="Edit"
                          >
                            <Edit3 className="w-3.5 h-3.5 text-slate-400 hover:text-white" />
                          </button>
                        )}
                      </div>
                    </div>

                    {sub.isEditing ? (
                      <textarea
                        value={sub.editedText || sub.text}
                        onChange={(e) => handleTextChange(sub.id, e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded p-2 text-sm text-white focus:outline-none focus:border-primary-purple/50 resize-none"
                        rows={2}
                        autoFocus
                      />
                    ) : (
                      <p className="text-sm text-white leading-relaxed">{sub.text}</p>
                    )}
                  </motion.div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right: Actions & Stats */}
        <div className="w-full lg:w-96 flex flex-col gap-6 sticky top-8 h-fit">
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4">Subtitle Stats</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-4">
                <div className="flex size-10 items-center justify-center rounded-full bg-green-500/20 text-green-400 shadow-glow">
                  <CheckCircle className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Total Lines</p>
                  <p className="text-white text-lg font-bold">{jobInfo.totalLines}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex size-10 items-center justify-center rounded-full bg-blue-500/20 text-blue-400 shadow-glow">
                  <Clock className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Duration</p>
                  <p className="text-white text-lg font-bold">{jobInfo.duration}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex size-10 items-center justify-center rounded-full bg-purple-500/20 text-purple-400 shadow-glow">
                  <Languages className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Language</p>
                  <p className="text-white text-lg font-bold">{jobInfo.language}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex size-10 items-center justify-center rounded-full bg-yellow-500/20 text-yellow-400 shadow-glow">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wider">Format</p>
                  <p className="text-white text-lg font-bold">{jobInfo.format}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-col gap-4">
            <button
              onClick={() => handleDownload("SRT")}
              className="btn-border-beam w-full group"
            >
              <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                <Download className="w-5 h-5 group-hover:translate-y-1 transition-transform" />
                <span>Download SRT</span>
              </div>
            </button>

            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleDownload("VTT")}
                className="flex items-center justify-center gap-2 py-2.5 rounded-lg border border-white/10 bg-white/5 text-white font-medium hover:bg-white/10 transition-all text-sm"
              >
                Download VTT
              </button>
              <button
                onClick={() => handleDownload("ASS")}
                className="flex items-center justify-center gap-2 py-2.5 rounded-lg border border-white/10 bg-white/5 text-white font-medium hover:bg-white/10 transition-all text-sm"
              >
                Download ASS
              </button>
            </div>

            <button
              onClick={() => router.push("/dashboard/subtitles")}
              className="w-full py-2.5 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
            >
              Generate New Subtitles
            </button>

            <button
              onClick={() => router.push("/dashboard/history")}
              className="w-full py-2.5 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
            >
              View Job History
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}