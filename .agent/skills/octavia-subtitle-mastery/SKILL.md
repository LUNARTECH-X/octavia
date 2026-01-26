# Skill: Octavia Subtitle Processing

Expertise in generating, translating, and formatting subtitles within the Octavia platform.

## Subtitle Engine

The engine is built around `SubtitleGenerator` and `SubtitleTranslator`, utilizing local Whisper models and simple translation backends.

### Generation Workflow
- **Module**: [`subtitle_generator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/subtitle_generator.py)
- **Engine**: Whisper (`base` or `tiny` model).
- **Formats**: Native support for **SRT** and **VTT**.
- **Bilingual**: Logic in `format_to_bilingual_srt` to interleave source and target text.

### Translation Workflow
- **Module**: [`subtitle_translator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/subtitle_translator.py)
- **Backends**: Google (via `deep_translator`) or local LLM.
- **Parsing**: Advanced parsing in `_manual_parse_srt` to handle malformed files.

### Subtitle-to-Audio
- A specialized pipeline that takes an SRT file, parses timestamps, and synthesizes speech segments for each entry.
- **Critical**: Must preserve the original timestamps to ensure the resulting audio can be re-muxed correctly.

## Styling & Constraints
- **Line Length**: Target is under 42 characters.
- **Time Aligment**: Uses `_format_timestamp` (SRT: `HH:MM:SS,mmm`) to ensure sub-millisecond precision.
- **Styling**: Partial support for ASS/SSA styling headers.

## Maintenance & Testing
- Use `test_subtitle_functionality.py` to verify formatting.
- Check `backend/outputs/subtitles` for artifacts after a job completes.
