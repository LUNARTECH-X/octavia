# Skill: Octavia Local Video Pipeline

Mastery of the local video translation and dubbing pipeline implemented in `VideoTranslationPipeline`.

## Pipeline Overview

The Octavia pipeline orchestrates high-performance video processing using local AI models and FFmpeg. It is designed as a monolithic FastAPI backend using multi-threading for concurrency.

### The 10-Step Workflow (Implemented)

1.  **Audio Extraction**: Uses FFmpeg (`extract_audio_fast`) to extract `pcm_s16le` audio.
2.  **Semantic Chunking**: Intelligent splitting in `chunk_audio_semantic` using VAD and sentence boundaries.
3.  **Transcription**: STT powered by `faster-whisper` (base model) via `transcribe_with_segments`.
4.  **Translation Selection**: Logical routing between Helsinki-NLP, NLLB-200 (for CJK/Russian), and Ollama.
5.  **Primary Translation**: LLM translation using `translategemma:4b` (via Ollama).
6.  **Duration Check**: Validates translated text against original duration with 15% tolerance.
7.  **Fit-to-Time Refinement**: Recursive LLM refinement (up to 3 retries) if duration mismatch is detected.
8.  **TTS Generation**: Synthesis using `edge-tts` (neural) or `gtts` fallback.
9.  **Speed Adjustment**: Frame-accurate lip-sync using FFmpeg `atempo` correction (0.85x - 1.1x).
10. **Final Merge**: Multiplexing dubbed audio with original video while preserving background music (if Magic Mode enabled).

## Core Implementation
- [`pipeline.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/pipeline.py): The main orchestrator class `VideoTranslationPipeline`.
- [`audio_translator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/audio_translator.py): Implements STT, Translation, and TTS synthesis.
- [`vocal_separator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/vocal_separator.py): Optional clean-up using `DemucsModel`.

## Key Logic Patterns
- **Memory Management**: The `MemoryMonitor` class throttles workers based on CPU/RAM/GPU percent.
- **Adaptive Chunking**: Adjusts `CHUNK_SIZE` based on speech complexity and punctuation density.
- **Sync Strategy**: Uses `atempo` filter. If `speed > 1.15`, the pipeline triggers a text reflow (translation rewrite).

## Debugging Local Faults
- **FFmpeg Error**: If `extract_audio_fast` fails, check if the video has an audio stream (`has_audio`).
- **OOM**: If `faster-whisper` crashes, the pipeline defaults to the `tiny` model.
- **TTS Drift**: Check `LIP_SYNC_TOLERANCE_MS` (default 200ms) and verify if the silence padding is active.
