# Octavia - Standard Video Translator

![LunarTech Logo](documentation/assets/lunartech_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Next.js](https://img.shields.io/badge/Next.js-14.0-black)](https://nextjs.org/)
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

The Octavia CLI provides powerful command-line tools for testing, development, and production use. It's particularly useful for contributors who want to test the system with their own video files.

#### Integration Testing
Test the complete video translation pipeline with various input methods:

```bash
# Interactive mode (prompts for video input)
python cli.py test-integration

# Test with local video file
python cli.py test-integration --input /path/to/your/video.mp4 --target-lang es

# Test with video from URL
python cli.py test-integration --input https://example.com/sample.mp4 --target-lang fr

# Test with different target language
python cli.py test-integration --input video.mp4 --target-lang de
```

**Test Input Options:**
- **Local files**: Any MP4/AVI/MOV video file on your system
- **URLs**: Direct download from HTTP/HTTPS URLs
- **Interactive prompts**: Step-by-step guidance when no arguments provided
- **Default detection**: Automatically offers the included test video if available

#### Other CLI Commands
```bash
# Translate video file directly
python cli.py video --input sample.mp4 --target es

# Generate subtitles only
python cli.py subtitles --input video.mp4 --format srt

# Translate audio file
python cli.py audio --input audio.wav --source-lang en --target-lang es

# Show processing metrics and logs
python cli.py metrics
```

#### CLI for Contributors
The CLI is designed to be contributor-friendly:
- **Flexible input**: Test with your own videos, URLs, or use provided samples
- **No setup required**: Works with existing backend dependencies
- **Comprehensive testing**: Validates all pipeline components
- **Clear reporting**: Detailed test results and performance metrics
- **Professional output**: Clean, technical formatting without distractions

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
- **Frontend Progress Tracker**: UI updates can be inconsistent. *Contributing Fixes Welcome!*

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
