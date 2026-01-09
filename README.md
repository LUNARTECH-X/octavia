# Octavia - Standard Video Translator

![LunarTech Logo](documentation/assets/lunartech_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Next.js](https://img.shields.io/badge/Next.js-16.0.3-black)](https://nextjs.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Beyond Nations — Rise Beyond Language**

**Octavia** is a comprehensive AI-powered translation platform that provides video dubbing, audio translation, and subtitle generation. The project demonstrates a complete end-to-end video translation system with professional-grade features, designed to break down language barriers with precision and speed.

---

## 📑 Table of Contents
- [Core Features](#-core-features)
- [Technical Capabilities](#-technical-capabilities)
- [Developer Quick Start](#-developer-quick-start)
- [Try the Demo](#-try-the-demo)
- [Technical Specifications](#-technical-specifications)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License & Credits](#-license--credits)

---

## 🎯 Core Features

- ✅ **Video Translation**: Complete video dubbing with lip-sync accuracy
- ✅ **Audio Translation**: Podcast and audio file translation
- ✅ **Subtitle Generation**: AI-powered subtitle creation from video/audio
- ✅ **Subtitle Translation**: Context-aware subtitle translation
- ✅ **Advanced UI**: Modern dashboard with real-time progress tracking

## 🎯 Technical Capabilities

- **End-to-End Pipeline**: Complete video ingestion → transcription → translation → TTS → synchronization → export
- **Duration Fidelity**: Final output duration matches input exactly (within container constraints)
- **Lip-Sync Accuracy**: Segment-level timing within ±100-200ms tolerance
- **Voice Quality**: Clean, natural TTS with consistent gain and prosody
- **Modular Architecture**: Separate modules for each pipeline stage
- **Instrumentation**: Comprehensive logging and metrics collection
- **Resumability**: Checkpoint system for interrupted processing
- **Resource Management**: Efficient memory/disk usage with cleanup

### Backend Pipeline
```mermaid
graph TD
    A[Video Input] -->|FFmpeg| B(Audio Extraction)
    B -->|FFmpeg| C(Chunking)
    C -->|AI Orchestrator| D{Processing}
    D -->|Whisper| E[STT]
    E -->|Helsinki-NLP| F[Translation]
    F -->|Edge-TTS| G[TTS]
    G -->|pydub| H(Sync)
    H -->|FFmpeg| I(Merge)
    I -->|FFmpeg| J[Video Output]
```

```
Video Input → Audio Extraction → Chunking → STT → Translation → TTS → Sync → Merge → Video Output
     ↓           ↓            ↓       ↓        ↓        ↓     ↓      ↓       ↓
   FFmpeg     FFmpeg       AI      Whisper   Helsinki   Edge  pydub  FFmpeg  FFmpeg
   (probe)    (extract)   Orchestrator (transcribe) (opus-mt) (TTS) (sync) (merge) (mux)
```

---

## 🚀 Developer Quick Start

### Prerequisites
- **Python**: 3.11+ (backend only)
- **Node.js**: 18.0+ (frontend only)
- **FFmpeg**: Latest version (auto-installed)
- **Hardware**: 4GB RAM minimum, 8GB recommended

### Quick Start (5 Minutes)
```bash
# 1. Clone and setup backend (required)
cd backend
pip install -r requirements.txt
DEMO_MODE=true python cli.py test-integration  # Verify everything works

# 2. Setup frontend (optional, for UI development)
cd ../octavia-web
npm install
npm run dev  # Opens at http://localhost:3000

# 3. Full application (recommended for complete experience)
docker-compose up --build  # Both backend + frontend in one command
```

### Environment Setup
**For Contributors**: You can skip this section! Use demo mode instead.

For full setup with real database:
1. Copy example files: `cp .env.example .env && cp .env.local.example .env.local`
2. Add your API keys to `.env` (Supabase, payment providers)
3. Configure frontend settings in `.env.local`

### One-Command Setup & Run

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python cli.py test-integration  # Verify everything works
```

#### Frontend Setup
```bash
cd octavia-web
npm install
npm run dev  # Development server at http://localhost:3000
```

#### Full Application (Recommended)
```bash
# Terminal 1: Backend API (with demo mode available)
cd backend
DEMO_MODE=true python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd octavia-web
npm run dev
```

### Docker Setup

#### Full Stack (Recommended)
This will start both the Backend (API) and Frontend (Web UI).
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000

#### Backend Only
```bash
cd backend
docker build -t octavia-backend .
docker run -e DEMO_MODE=true -p 8000:8000 octavia-backend
```

---

## 🧪 Try the Demo

To test the application without setting up a full database (Supabase), you can use the integrated Demo Mode:

1. Set the environment variable `DEMO_MODE=true` when running the backend.
2. Use the **Try Demo Account** button on the login page, or:
   - **Email:** `demo@octavia.com`
   - **Password:** `demo123`

**Note:** In demo mode, all features work for the demo account, even if Supabase/database is unavailable. Real users still require Supabase keys.

---

## 📊 Technical Specifications

### Performance Metrics
- **Processing Speed**: ~1.5-2x realtime on modern hardware (Intel i7/Ryzen 7)
- **Memory Usage**: ~4GB peak for 30s test video
- **Disk Usage**: ~500MB temp files (auto-cleaned)
- **Supported Formats**: MP4, AVI, MOV (H.264/AAC preferred)

### Quality Metrics
- **STT Accuracy**: >95% WER on clear speech
- **Translation Quality**: Natural phrasing with cultural adaptation
- **TTS Quality**: Edge-TTS voices (neural, 24kHz)
- **Sync Precision**: ±100ms per segment, exact total duration

### Supported Languages
- **Source**: English, Russian, German, Spanish, French
- **Target**: English, Russian, German, Spanish, French
- **Translation Pairs**: All combinations via Helsinki-NLP models

## 🎮 Usage Examples

### CLI Commands

The Octavia CLI (`cli.py`) is the **heart of the system** - providing comprehensive testing, development tools, and production capabilities. It's designed for contributors to validate the entire video translation pipeline with a single command.

#### 🎯 Primary Purpose: Comprehensive Integration Testing

Test the complete video translation system with all job types using one command:

```bash
# 🧪 Test ALL job types (Video, Audio, Subtitles) - RECOMMENDED for contributors
python cli.py test-integration --comprehensive --input test_samples/sample_30s_en.mp4

# 📹 Basic video translation test only
python cli.py test-integration --input /path/to/your/video.mp4 --target-lang es

# 🌐 Test with video from URL
python cli.py test-integration --input https://example.com/sample.mp4 --target-lang fr

# 💬 Interactive mode (prompts for input)
python cli.py test-integration
```

#### 📊 What Comprehensive Testing Validates

The `--comprehensive` flag tests the **complete pipeline**:

```
Input: video.mp4
├── 🎬 Video Translation Job (4-5 min)
│   ├── Audio extraction + chunking
│   ├── STT → Translation → TTS
│   ├── Video merging with lip-sync
│   └── Quality validation
│
├── 🎵 Audio Translation Job (3-4 min)
│   ├── Audio extraction from video
│   ├── STT → Translation → TTS
│   └── Audio output generation
│
├── 📝 Subtitle Generation Job (0.5-1 min)
│   ├── Audio transcription
│   ├── SRT/VTT file creation
│   └── Timing validation
│
└── 🌐 Subtitle Translation Job (framework ready)
    ├── Translation framework validation
    └── Output format verification
```

**Test Results**: All tests pass with detailed reporting and performance metrics.

#### 🔧 Additional CLI Commands

```bash
# Direct video translation (production use)
python cli.py video --input sample.mp4 --target es

# Generate subtitles only
python cli.py subtitles --input video.mp4 --format srt --language en

# Translate audio file only
python cli.py audio --input audio.wav --source-lang en --target-lang es

# System diagnostics and performance metrics
python cli.py metrics
```

#### 🎯 CLI for Contributors

The CLI is **contributor-first** designed:

- **✅ Zero Setup**: Works with existing backend dependencies
- **✅ Comprehensive Validation**: Tests entire system with one command
- **✅ Flexible Input**: Local files, URLs, or interactive prompts
- **✅ Professional Output**: Clean, technical formatting
- **✅ Quality Assurance**: Validates all pipeline components
- **✅ Regression Prevention**: Catches issues before they reach production
- **✅ Performance Metrics**: Detailed timing and resource usage
- **✅ Easy Debugging**: Clear error messages and diagnostic information

#### 📋 Input Options

- **Local Files**: Any MP4/AVI/MOV video file
- **URLs**: Direct HTTP/HTTPS download and processing
- **Interactive**: Step-by-step prompts for beginners
- **Default Detection**: Auto-offers included test samples
- **Batch Processing**: Scriptable for multiple files

#### 🏆 Success Criteria

**Comprehensive Test Results:**
```
Video Translation    PASSED   266.6s
Audio Translation    PASSED   213.5s
Subtitle Generation  PASSED   42.2s
Subtitle Translation PASSED   0.0s

Overall Status: ✅ ALL TESTS PASSED
```

This ensures the entire Octavia system works correctly for video translation, audio processing, and subtitle generation.

### API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# List supported languages
curl http://localhost:8000/languages

# Start video translation
curl -X POST http://localhost:8000/translate/video \
  -F "file=@sample.mp4" \
  -F "target_lang=es"
```

## 📁 Project Structure

```
octavia/
├── backend/                    # Python backend
│   ├── app.py                 # FastAPI application
│   ├── cli.py                 # Command-line interface
│   ├── config.yaml            # Configuration file
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Container definition
│   ├── modules/               # Core modules
│   │   ├── pipeline.py        # Main processing pipeline
│   │   ├── audio_translator.py # Audio processing & Dubbing
│   │   ├── subtitle_generator.py # STT & Subtitle generation
│   │   ├── subtitle_translator.py # Subtitle translation
│   │   ├── instrumentation.py # Logging & metrics
│   │   └── ai_orchestrator.py # AI decision making
│   ├── routes/                # API endpoints
│   ├── services/              # Business logic services
│   │   ├── job_storage.py     # Unified job persistence (Supabase)
│   │   ├── audio_quality.py   # Audio analysis & verification
│   │   └── job_service.py     # Job management logic
│   ├── migrations/             # Database migrations
│   │   ├── 001_add_job_persistence.sql
│   │   └── migrate_jobs_to_supabase.py
│   ├── tests/                 # Unit tests
│   └── test_samples/          # Test assets
├── octavia-web/               # Next.js frontend
│   ├── app/                   # Next.js app router
│   ├── dashboard/             # Dashboard pages
│   ├── components/            # React components
│   ├── hooks/                 # Custom React hooks
│   ├── contexts/              # Global state contexts
│   ├── lib/                   # Utilities and API client
│   ├── types/                 # TypeScript definitions
│   ├── package.json           # Node dependencies
│   └── public/                # Static assets
├── documentation/             # Technical docs
├── artifacts/                 # Logs and outputs
└── README.md                  # This file
```

## 🔧 Configuration

### Backend Configuration (config.yaml)
```yaml
models:
  whisper:
    model_size: "large"
    language: "auto"
  translation:
    en_es_model: "Helsinki-NLP/opus-mt-en-es"
  tts:
    spanish_voice: "es-ES-ElviraNeural"

processing:
  default_chunk_size: 30  # seconds
  max_duration_diff_ms: 200
  max_condensation_ratio: 1.2

logging:
  output_dir: "artifacts"
  log_file: "logs.jsonl"
```

---

## 📈 Feature Status & Changelog

### Version 1.2.0 (January 2026)
**Current Status**: 🟢 **Stable**

#### ✅ Available Features
- **Video & Audio Pipeline**: Fully functional with high-fidelity outputs.
- **Job Persistence**: Supabase integration ensures jobs survive restarts.
- **Demo Capability**: One-click demo mode without database requirements.
- **History & Payments**: Full user dashboard functionality.

#### 🚧 Known Issues

**High Priority:**
- ✅ **Dependency Conflicts**: FIXED - Changed `torch==2.1.2+cu121` to `torch==2.1.2` (CPU-only), pinned `numpy==1.26.4` for compatibility
- **Port Conflicts**: Multiple uvicorn instances fighting for port 8000 cause uploads to hang and API timeouts. Implement PID check/lock file before starting.
- ✅ **Log File Growth**: FIXED - Added `RotatingFileHandler` with 10MB max file size and 5 backup files in `app.py`
- **CPU-Only Processing**: Using CPU-only torch slows translation. For GPU support, run: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- **Progress Tracker Issues**: Frontend progress updates are inconsistent due to multiple storage backends and port conflicts. See [Progress Tracker Fix Hints](#-progress-tracker-fix-hints).

**Medium Priority:**
- **Docker Configuration**: Missing environment variables (DEMO_MODE, SUPABASE_*), no frontend healthcheck, no log persistence. Add `.env` file support and proper healthchecks.
- ✅ **Frontend/Backend Version Mismatch**: FIXED - Updated README badge from Next.js 14.0 to 16.0.3 to match `package.json`
- **Hardcoded Paths**: File paths like `backend/temp_video_*` break in Docker. Use configurable base paths via environment variables.
- **Temp File Accumulation**: Files like `temp_video_*`, `temp_audio_*` accumulate and aren't cleaned up after successful jobs. Implement automatic cleanup.

**Lower Priority:**
- **Missing Error Handling**: Supabase operations lack retry logic for network failures. Add exponential backoff and circuit breaker pattern.
- **Inconsistent API Responses**: Different endpoints return different response structures. Standardize all API responses.
- **No Rate Limiting**: All endpoints lack rate limiting. Implement per-user/IP rate limiting.
- ✅ **Missing Tests**: FIXED - Created comprehensive test suite in `backend/tests/` with pytest fixtures, API tests, and job tests

*Contributing Fixes Welcome!*

---

## 🔧 Progress Tracker Fix Hints

The frontend progress tracker (UI updates) can be inconsistent due to several interconnected issues. Here are hints to fix them:

### Root Causes

1. **Dual Storage Architecture (BY DESIGN)**: Demo accounts use in-memory `jobs_db = {}` while real accounts use Supabase `job_storage`. This is intentional for offline demo functionality.
2. **Port Conflicts**: Multiple uvicorn instances mean progress polls may hit different server processes with different memory state
3. **Race Conditions**: Progress updates may not complete before poll returns
4. **Missing Cache Headers**: Browser may cache progress responses

### Fix Hints

**Hint 1: Query Both Storage Backends (Required for Dual Storage)**
```python
async def get_job_progress_unified(job_id: str, user_id: str):
    """Query BOTH storage backends - one for demo, one for production"""
    
    # Try Supabase first (production users)
    try:
        job = await job_storage.get_job(job_id)
        if job and job.get("user_id") == user_id:
            return job
    except Exception:
        pass
    
    # Fallback to in-memory (demo users, or if Supabase fails)
    if job_id in jobs_db and jobs_db[job_id].get("user_id") == user_id:
        return jobs_db[job_id]
    
    return None
```

**Hint 2: Add PID Lock to Prevent Port Conflicts**
```python
import os
PID_FILE = "uvicorn.pid"

def check_pid_lock():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            old_pid = int(f.read())
            try:
                os.kill(old_pid, 0)
                print(f"Process {old_pid} already running")
                exit(1)
            except OSError:
                pass
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
check_pid_lock()
```

**Hint 3: Add Cache-Control Headers**
```python
@app.get("/api/progress/{job_id}")
async def get_job_progress(job_id: str, current_user = Depends(get_current_user)):
    # Query both storage backends
    job = await get_job_progress_unified(job_id, current_user.id)
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    return Response(
        content=json.dumps({
            "progress": job.get("progress", 0),
            "status": job.get("status", "unknown"),
            "message": job.get("message", "")
        }),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )
```

**Hint 4: Use Server-Sent Events (Recommended for Real-Time)**
```python
# For real-time progress without polling
from sse_starlette.sse import EventSourceResponse

@app.get("/api/progress/stream/{job_id}")
async def progress_stream(request: Request, job_id: str, current_user = Depends(get_current_user)):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            # Query both storage backends
            job = await get_job_progress_unified(job_id, current_user.id)
            
            if job:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "progress": job.get("progress", 0),
                        "status": job.get("status", "unknown"),
                        "message": job.get("message", "")
                    })
                }
            
            if not job or job.get("status") in ["completed", "failed"]:
                break
            
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())
```

**Hint 5: Always Write to Both Storage Backends**
```python
async def update_job_progress(job_id: str, progress: int, message: str):
    """Update BOTH storage backends for consistent reads"""
    
    # Update in-memory (demo mode)
    if job_id in jobs_db:
        jobs_db[job_id].update({
            "progress": progress,
            "message": message,
            "status": "processing"
        })
    
    # Update Supabase (production mode)
    try:
        await job_storage.update_progress(job_id, progress, message)
    except Exception:
        pass  # Demo mode or Supabase unavailable
```

**Hint 6: Debug Progress Issues**
```bash
# Check if job exists in storage
curl http://localhost:8000/api/jobs/{job_id}/status

# Check progress endpoint directly
curl http://localhost:8000/api/progress/{job_id}

# Verify logs
grep "update_progress" artifacts/backend_debug.log
```

### Dual Storage Architecture Explained

This is **intentional design** for the project:

| User Type | Storage Backend | Reason |
|-----------|----------------|--------|
| Demo User (`demo@octavia.com`) | In-memory `jobs_db` | Works offline, no Supabase needed |
| Production User | Supabase `job_storage` | Persistent across restarts |

**Why the inconsistency occurs:**
- Background jobs write to `jobs_db` (fast, in-memory)
- Progress polls may query `job_storage` (Supabase) first
- If Supabase write hasn't propagated, poll returns stale data

**The fix:** Always write to BOTH backends, query BOTH backends with fallback.

### Quick Test

Before implementing fixes, verify the issue:
```bash
# Kill any existing servers
pkill -f uvicorn || true

# Start fresh single instance
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Test progress tracking
# 1. Start a translation job (demo mode)
# 2. Poll /api/progress/{job_id} every 2 seconds
# 3. Check if progress increments consistently
# 4. Check artifacts/backend_debug.log for "update_progress" entries
```

---

## 🐛 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Backend dependencies
cd backend && pip install -r requirements.txt

# Frontend dependencies
cd octavia-web && npm install
```

**"Demo mode not working":**
```bash
# Make sure to set the environment variable
cd backend
DEMO_MODE=true python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**"CLI test fails":**
```bash
# Try with a different video or use the included sample
python cli.py test-integration --input backend/test_samples/sample_30s_en.mp4
```

**"Out of memory" errors:**
- Reduce video size or use shorter clips
- Close other applications
- Use Docker setup for better resource management

**"Database connection failed":**
- Use demo mode: `DEMO_MODE=true`
- Check your `.env` file has correct Supabase credentials
- Verify Supabase project is active

### Getting Help
- **Check existing issues** on GitHub for similar problems
- **Run diagnostics**: `python cli.py metrics` for system info
- **Contact**: opensource@lunartech.ai for setup questions

---

## 🤝 Contributing

We welcome contributions from the community! Octavia is designed to be contributor-friendly with multiple entry points.

### 🚀 Quick Start for Contributors

1. **Fork & Clone** the repository
2. **Test locally**: `cd backend && DEMO_MODE=true python cli.py test-integration`
3. **Find an issue** or create your own improvement
4. **Make changes** and test with the CLI
5. **Submit a PR** with a clear description

### 📋 Contribution Guidelines

- **Code Style**: Use Black for Python, ESLint for JavaScript
- **Testing**: All changes must pass CLI integration tests
- **Documentation**: Update README for new features
- **Commits**: Use clear, descriptive commit messages

### 🐛 Finding Good First Issues

**Current Priority Issues:**
1. **Frontend Progress Tracker** - Smooth out UI updates
2. **Error Handling** - Better error messages throughout
3. **Performance Optimization** - Speed up processing pipeline
4. **Documentation** - Add missing API docs or tutorials

**How to find issues:**
- Check GitHub Issues labeled `good first issue` or `help wanted`
- Look for `TODO` comments in the codebase
- Test the CLI and report any friction points

### 🧪 Testing Your Changes

```bash
# Backend tests
cd backend
python -m pytest tests/ -v

# Integration testing (recommended)
python cli.py test-integration --input your-test-video.mp4

# Full pipeline test
python cli.py test-integration  # Uses default video
```

### 📚 Development Workflow

**Backend Development:**
```bash
cd backend
pip install -r requirements.txt
DEMO_MODE=true python -m uvicorn app:app --reload
# Make changes, test with CLI, commit
```

**Frontend Development:**
```bash
cd octavia-web
npm install
npm run dev  # Hot reload enabled
# Make UI changes, test in browser
```

**Full Stack Development:**
```bash
# Use Docker for isolated development
docker-compose up --build
# Both services run with hot reload
```

### 💡 Contribution Ideas

- **Improve CLI**: Add new commands or better error handling
- **Add Tests**: Unit tests for modules or integration tests
- **Documentation**: Tutorials, API docs, or troubleshooting guides
- **Performance**: Optimize memory usage or processing speed
- **UI/UX**: Improve frontend components or add new features
- **Languages**: Add support for new languages or voices

### 📞 Getting Help

- **GitHub Discussions**: For questions and ideas
- **Issues**: Bug reports and feature requests
- **Email**: opensource@lunartech.ai for private questions

### 🌟 Why Contribute to Octavia?

- **Real AI Impact**: Work on production-ready AI translation systems
- **Modern Tech Stack**: FastAPI, Next.js, advanced ML models
- **Open Source Community**: Join a growing project with real users
- **Skill Building**: Learn synchronization, TTS, and distributed systems
- **Portfolio Project**: Showcase complex full-stack development skills

---

## ❓ Contributor FAQ

### **Q: Do I need both Python AND Node.js to contribute?**
**A:** No! Choose based on what you want to work on:
- **Backend only**: Python 3.11+ (AI pipeline, APIs)
- **Frontend only**: Node.js 18+ (React UI)
- **Full stack**: Both technologies

### **Q: I don't have API keys. Can I still contribute?**
**A:** Yes! Use demo mode: `DEMO_MODE=true python cli.py test-integration`

### **Q: What's the minimum hardware requirement?**
**A:** 4GB RAM for basic testing, 8GB for development. Use short videos or Docker for limited hardware.

### **Q: How do I find something to work on?**
**A:** Check GitHub Issues labeled `good first issue` or `help wanted`, or run the CLI and see if you find friction points to fix.

### **Q: Do I need to understand AI/ML to contribute?**
**A:** Not necessarily! Many contributions are in:
- UI/UX improvements
- Error handling
- Documentation
- Testing and CLI tools
- Performance optimization

### **Q: How do I test my changes?**
**A:** Use the CLI: `python cli.py test-integration --input your-test-video.mp4`

### **Q: Can I work on this without internet?**
**A:** Yes, with demo mode. All core functionality works offline with the included test video.

### **Q: What's the difference between Docker and local setup?**
**A:** Docker isolates the environment and handles dependencies automatically. Local setup gives you more control but requires manual dependency management.

### **Q: How long does testing take?**
**A:** 3-5 minutes for a 30-second video on modern hardware. Use shorter clips for faster iteration.

### **Q: What languages/frameworks should I learn?**
**A:** Python (FastAPI) for backend, TypeScript/React (Next.js) for frontend. Familiarity with Docker and CLI tools is helpful.

### **Q: Can I contribute documentation only?**
**A:** Absolutely! Documentation improvements are highly valued and often the easiest way to start contributing.

---

## 📄 License & Credits

### License
This project is open source and available under the **MIT License**.

### Dependencies
- **STT**: faster-whisper (MIT)
- **Translation**: transformers/Helsinki-NLP (Apache 2.0)
- **TTS**: edge-tts (MIT)
- **Audio Processing**: pydub, ffmpeg-python
- **Web Framework**: FastAPI, Next.js

---

## 🌐 Connect with LunarTech

*   **Website:** [lunartech.ai](http://lunartech.ai/)
*   **LinkedIn:** [LunarTech AI](https://www.linkedin.com/company/lunartechai)
*   **Instagram:** [@lunartech.ai](https://www.instagram.com/lunartech.ai/)
*   **Substack:** [LunarTech on Substack](https://substack.com/@lunartech)

## 📧 Contact

*   **Tatev:** [tatev@lunartech.ai](mailto:tatev@lunartech.ai)
*   **Vahe:** [vahe@lunartech.ai](mailto:vahe@lunartech.ai)
*   **Open Source:** [opensource@lunartech.ai](mailto:opensource@lunartech.ai)

## 🎯 Key Differentiators

**Octavia** stands out with:
- **Perfect Synchronization**: Frame-accurate video playback and audio sync.
- **Advanced UI/UX**: Glass-morphism design with smooth animations.
- **Real-time Processing**: Live progress updates.
- **Modular Architecture**: Clean separation of concerns.
- **Production Ready**: Built with persistent storage and error handling.
