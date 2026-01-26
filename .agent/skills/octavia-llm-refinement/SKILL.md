# Skill: Octavia Prompt Engineering

Knowledge of the specialized prompts and refinement loops used for high-fidelity translation in Octavia.

## Specialized Prompt Sets

### 1. Fit-to-Time Refinement
Used to condense or expand translations to match original timing.
- **Pattern**: "Translate to [TARGET] but fit within [X.X] seconds. Original: [Y.Y] seconds. Summarize if needed, preserve core meaning."
- **Logic**: Implemented as a recursive loop in `audio_translator.py` with up to 3 attempts.

### 2. Sentence Boundary Detection (CJK/Legacy)
Used for CJK languages where punctuation is sparse.
- **Pattern**: "Identify natural sentence boundaries for the following text. Mark them with unique separators."
- **Model**: Typically uses `qwen2.5:1.5b` for speed.

### 3. Translation Post-Processing
Used to fix name transliterations or grammar when using NMT fallbacks (NLLB/Helsinki).
- **Core Prompt**: "You are a professional editor. Improve the following translation for natural fluency and name consistency while maintaining the exact meaning."

## Configuration
- `LLM_TRANSLATION_PROMPT`: The base system message for initial translation.
- `LLM_TEMPERATURE`: Default 0.3. Low temperature is critical to prevent hallucinations and maintain timestamp alignment.

## Expert Techniques
- **Context Injection**: The `translate_text_with_context` method provides the LLM with previous/next sentences to ensure narrative continuity.
- **Verification**: Cross-matching LLM output word count against `VOICE_RATES` to predict TTS duration BEFORE calling the TTS engine.
