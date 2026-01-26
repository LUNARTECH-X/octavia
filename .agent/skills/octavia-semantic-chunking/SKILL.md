# Skill: Octavia Semantic Fragmentation

Knowledge of how Octavia splits long-form audio into semantically meaningful chunks for translation.

## Fragmentation Logic

Instead of fixed-time splitting, Octavia uses a combination of VAD (Voice Activity Detection) and Sentence Boundary Detection.

### The Chunking Algorithm
1. **Raw Transcription**: Get segments with word-level timestamps via `faster-whisper`.
2. **Gap Analysis**: Calculate `gap_to_next` between segments.
3. **Punctuation Check**: Identify sentence ends (`. ! ? 。 ！ ？`).
4. **Decision Logic**:
   - If `potential_duration > CHUNK_SIZE` (default 30s) -> Force Cut.
   - If `duration >= MIN_CHUNK_SIZE` AND (`gap > 0.5s` OR `sentence_end` is True) -> Cut.
5. **Silence Filling**: Gaps between `start` and `last_end` are filled to ensure audio continuity.

## Core Files
- [`add_sentence_detection.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/add_sentence_detection.py): Logic for CJK and Western sentence splitting.
- [`pipeline.py` (chunk_audio_semantic)](file:///c:/Users/onyan/octavia/octavia/backend/modules/pipeline.py#L682): The orchestrator method for the fragmentation loop.

## Key Constants
- `min_pause_duration_ms`: 500ms.
- `vad_threshold`: 0.5.
- `max_chunk_duration_s`: 30s.

## Optimization & Tuning
- **CJK Challenges**: Semantic chunking is particularly important for Mandarin/Japanese to avoid cutting mid-thought, as word-to-word timing is less reliable.
- **Diarization Gap**: While full diarization is planned (XTTS v2), currently fragmentation relies on pauses to separate different speakers implicitly.
