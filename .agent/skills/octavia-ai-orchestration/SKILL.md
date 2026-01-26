# Skill: Octavia Local AI Orchestration

Expertise in managing the `AIOchestrator` and the decision-making logic that optimizes the translation pipeline.

## Orchestration Logic

Octavia uses a tiered AI system to manage processing parameters dynamically.

### AI Modes
1. **Ollama AI Mode**: Uses local models like `translategemma:4b` or `qwen2.5:7b` for high-level reasoning.
2. **Llama Mode**: Fallback via `llama.cpp` if Ollama is unavailable.
3. **Rule Mode**: Deterministic heuristics used when LLM providers are offline.

### Key Responsibilities
- **Dynamic Parameter Tuning**: Adjusting `CHUNK_SIZE` and `max_workers` based on system load metrics from `MemoryMonitor`.
- **Model Selection**: Deciding which Whisper or NMT model to use based on input language complexity.
- **Self-Healing**: Triggering OOM retries or model downgrades (e.g., swapping `base` whisper for `tiny`).

## Core Implementation
- [`ai_orchestrator.py`](file:///c:/Users/onyan/octavia/octavia/backend/modules/ai_orchestrator.py): The main orchestrator logic.
- [`pipeline.py#L398`](file:///c:/Users/onyan/octavia/octavia/backend/modules/pipeline.py#L398): How the pipeline integrates the orchestrator decisions.

## Metrics & Feedback
Octavia collects `ProcessingMetrics` to refine decisions:
- `cpu_percent`, `memory_percent`, `gpu_percent`.
- `stt_speed_ratio` (seconds of audio processed per second of real time).
- `translation_duration_ms`.

## Common Tasks
- **Updating Model IDs**: When you pull a new model (e.g., `ollama pull deepseek-r1`), update the `llm_model` in `TranslationConfig`.
- **Threshold Adjustment**: Tuning `memory_throttle_threshold` (default 85%) to prevent system-wide lag on low-RAM machines.
