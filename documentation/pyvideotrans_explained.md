# PyVideoTrans Overview & Workflow

**Website:** [https://pyvideotrans.com/en/](https://pyvideotrans.com/en/)  
**GitHub:** [https://github.com/jianchang512/pyvideotrans](https://github.com/jianchang512/pyvideotrans)

## Overview

PyVideoTrans is an open-source Python-based tool for translating videos from one language to another. It handles the entire pipeline: automatic transcription, subtitle generation, text translation, dubbing (via text-to-speech), and final video assembly.

**Key Features:**
*   **Local Processing:** Processes videos locally on your machine (privacy-focused).
*   **Language Support:** Supports a wide range of languages (English, Chinese, Japanese, Korean, Spanish, French, etc.) with automatic detection.
*   **Core Technologies:** FFmpeg (media handling), various Speech Recognition models (transcription), multiple Translation APIs, and diverse TTS engines.
*   **High Configurability:** GUI interface for selecting engines, models, speeds, volumes, pitches, and advanced options.
*   **Magic Mode Features:** Background music preservation (UVR5), GPU acceleration (CUDA), context awareness, and subtitle recognition accuracy controls.

**Latest Updates (v3.80+):**
*   New TTS channels (Qwen-TTS, Gemini TTS variants).
*   Translation channels (Zhipu AI, Siliconflow, OpenRouter.ai).
*   Speaker recognition support.
*   Multi-role dubbing enhancements.
*   Optimizations for LLM sentence breaking and cold start speed.

---

## Detailed Workflow

This workflow describes the process from video input to final translated output.

### Step 1: Video Input and Audio Extraction

*   **Process:** The tool takes your input video (MP4, AVI, etc.) and uses FFmpeg to extract the audio track into a separate file (typically WAV). It also generates a "silent" version of the video to preserve visuals and timing.
*   **Vocal Separation:** If enabled, UVR5 isolates voices from background noise/music, creating dual audio streams to preserve the original background.
*   **Handling Long Videos:** Audio is segmented based on settings (e.g., max speech duration 5s, buffer 400ms) to manage processing load.
*   **Key Settings:**
    *   **Preserve Background Sound:** Checkbox (controlled via BGM volume slider).
    *   **GPU Acceleration:** Enable CUDA decode/transcode for performance.
    *   **Output Quality:** Configurable compression rates (H.264/H.265).

### Step 2: Speech Recognition (Transcription)

*   **Process:** Extracted audio is transcribed into timed text segments using the selected STT (Speech-to-Text) engine, generating an SRT file with timestamps.
*   **Engines:** faster-whisper, openai-whisper, OpenAI API, Deepgram, Gemini, Parakeet, GoogleSpeech, Ali FunASR, and more.
*   **Key Settings:**
    *   **Language:** Auto-detect or manual selection.
    *   **Prompts:** Custom prompts per language to improve accuracy (e.g., specific prompts for Japanese, Korean, etc.).
    *   **VAD (Voice Activity Detection):** Threshold (e.g., 0.45) to precisely detect speech boundaries.
    *   **Segmentation:** Options for shortest/max speech duration and silence buffering.

### Step 3: Text Translation

*   **Process:** The source SRT file is translated line-by-line or in batches into the target language, strictly preserving timestamps.
*   **Engines:** Microsoft, Google, Baidu, Tencent, ChatGPT, AzureAI, Gemini, DeepSeek, Claude, DeepL, and many others including offline options.
*   **Key Settings:**
    *   **Batch Processing:** Simultaneous subtitle translation (e.g., 20 lines at a time).
    *   **Context:** Options to send full content or use LLMs for context-aware translation.
    *   **Formatting:** Controls for line length (e.g., 20/80 chars) to ensure subtitles fit the screen.

### Step 4: Text-to-Speech (Dubbing Generation)

*   **Process:** Translated text is converted into audio segments. Each subtitle line becomes a specific audio clip, timed to match the original speech.
*   **Engines:** Edge TTS, Google TTS, Azure AI TTS, OpenAI TTS, ElevenLabs, GPT-SoVITS, Clone-Voice, and more.
*   **Key Settings:**
    *   **Voice Customization:** Select accents, adjust speed, volume, and pitch.
    *   **Multi-Role Dubbing:** Assign different voices to different detected speakers.
    *   **Adjustments:** Options to speed up dubbing or slow down video to match lengths.

### Step 5: Audio Stitching and Synchronization

*   **Process:** The generated audio segments are stitched together. The tool adjusts for length differences between the original and translated speech (e.g., by slightly speeding up audio or slowing down video).
*   **Subtitles:** Hard subtitles are burned into the video if configured (font, size, color, and position are customizable).
*   **Key Settings:**
    *   **Sync:** Force sync options and removal of end silence.
    *   **Subtitle Style:** Font (e.g., 微软雅黑), size (16), color (#ffffff), and position.

### Step 6: Final Merging

*   **Process:** The tool combines the synchronized dubbed audio, the optional background music track, subtitles, and the silent video into the final output file.
*   **Output:** A fully translated video file (e.g., `output.mp4`).
*   **Key Settings:**
    *   **Encoding:** Lossless or lossy compression options.
    *   **Format:** MP4 is standard, with options for others.

---

## Additional Notes

*   **Performance:** A 10-minute video typically takes 30-60 minutes on a CPU. GPU acceleration (CUDA) significantly speeds this up.
*   **Batch Processing:** The tool supports processing multiple videos in a queue.
*   **Limitations:** Requires Python 3.10+. Local modes work without internet, but cloud APIs require connectivity.
