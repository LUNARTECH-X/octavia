# Skill: Octavia Systematic Debugging

A structured methodology for identifying and resolving failures in the Octavia translation pipeline.

## Log Analysis Workflow

Always start by checking the dual log files:
1. `backend_debug.log`: API level issues (Auth, Routing, Job Lifecycle).
2. `octavia_pipeline.log`: Processing level issues (FFmpeg, Whisper, NMT, TTS).

## Common Failure Modes

### 1. FFmpeg "atempo" Errors
- **Symptom**: "Error while filtering... filter 'atempo' not found".
- **Cause**: Requesting speed outside the 0.5x - 2.0x range or malformed filter chain.
- **Fix**: Verify `speed_adjustment` value in `TranslationResult`.

### 2. Ollama Connectivity
- **Symptom**: "Connection refused" or 404 when calling LLM.
- **Cause**: Ollama service is not running or the model hasn't been pulled.
- **Fix**: Check `net start ollama` and verify model availability via `ollama list`.

### 3. Whisper OOM (Out of Memory)
- **Symptom**: Process crashes or returns `EXIT_CODE_139`.
- **Cause**: Input video is too long for the available GPU VRAM.
- **Fix**: Enable `memory_throttle_threshold` or reduce `CHUNK_SIZE` in `.env`.

### 4. Edge-TTS Empty Output
- **Symptom**: Transcription exists but synthesized audio is 0 bytes.
- **Cause**: Invalid voice ID for the selected language.
- **Fix**: Check `VOICE_MAPPING` in `audio_translator.py` for consistency.

## Testing Matrix
- `test-integration`: Full end-to-end check.
- `test_audio_translator.py`: Unit test for STT/Translation/TTS.
- `test_subtitle_functionality.py`: Checks for SRT/VTT formatting.

---

## 📝 Living Documentation
**New Failure Patterns**: If we encounter a new, recurring issue during development, **always add it to this file**. This ensures that Antigravity and future agents "remember" the fix, preventing the need for repeat prompting.
