# Skill: Octavia Local DevOps

Knowledge of managing Octavia's local infrastructure, configuration, and storage.

## Configuration Mastery

### Environment Variables (.env)
- **`DEMO_MODE`**: Set to `true` to skip Supabase authentication and persistence. This is the primary dev mode.
- **`OLLAMA_HOST`**: Typically `http://localhost:11434`.
- **`LOG_LEVEL`**: Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

### Directory Structure
- **`/tmp/octavia`**: Primary temporary workspace (should be cleared periodically).
- **`backend/outputs`**: Final artifacts (videos, audio, subtitles).
- **`~/.cache/octavia`**: Local storage for downloaded AI models.

## Local Infrastructure

### Local Persistence
- Octavia uses `job_storage.py` and `db_utils.py` for local persistence.
- **Note**: In local mode, jobs are often stored in JSON or local SQLite if not using the full Supabase fleet.

### Docker & Deployment
- **`docker-compose.yml`**: Full stack (Next.js + FastAPI).
- **`docker-compose.dev.yml`**: Mounts local volumes for rapid development.
- **Dependencies**: FFmpeg is a hard requirement; ensure it's in the system PATH.

## Maintenance Tasks
- **Cleanup**: Running `cleanup_utils.py` to remove temporary files older than 24 hours.
- **Logs**: Monitoring `backend_debug.log` and `octavia_pipeline.log`.
- **Model Management**: Using `ollama list` and `ollama pull` to manage local LLM state.
