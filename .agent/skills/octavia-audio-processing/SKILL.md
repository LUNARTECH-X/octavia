# Skill: Octavia Audio Translation

Technical mastery of the `AudioTranslator` module, focusing on transcription, multilingual translation, and local TTS synthesis.

## Core Component: AudioTranslator

The [`audio_translator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/audio_translator.py) manages the internal translation logic for local processing.

### Transcription (Whisper)
- **Primary**: `faster-whisper` (base model) with `float16` compute type on GPU.
- **Fallback**: Standard `whisper` base model.
- **Feature**: Uses `detect_language` for auto-detection with confidence scores.

### Translation Strategy
Octavia routes translation based on language pairs:
1. **Helsinki-NLP**: Used for most European language pairs (MarianMT).
2. **NLLB-200**: Used specifically for high-quality **CJK** (Chinese, Japanese, Korean) and **Russian** translation.
3. **Ollama**: Primary high-quality translation using `translategemma:4b` or `qwen2.5:7b`.

### TTS Synthesis
- **Edge-TTS**: Primary engine for Microsoft Neural voices (e.g., `en-US-JennyNeural`).
- **gTTS**: Secondary fallback using Google Translate engine.
- **Rate/Pitch**: Managed via `VOICE_RATES` (e.g., English: 12 chars/s) to estimate duration.

## Synchronization & Quality
- **Fit-to-Time**: If the estimated TTS duration exceeds the original, it triggers a specialized prompt: "Translate and fit within X seconds".
- **Audio Quality**: Uses `pydub` for normalization (-16 LUFS) and silence padding.
- **Semantic Cuts**: Cuts audio at `gap_to_next > 0.5s` or end-of-sentence punctuation (.!?).

## Common Debugging Scenarios
- **"Model not found"**: Helsinki/NLLB models are downloaded to `~/.cache/octavia`. Ensure disk space.
- **TTS Timeout**: Edge-TTS relies on an internet connection. If it fails, verify the system can reach Microsoft's TTS servers.
- **Language Mismatch**: Verify the `source_lang` being passed to `transcribe_with_segments`.
