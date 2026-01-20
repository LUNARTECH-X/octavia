# Octavia Docker Build Options

## Quick Comparison

| Dockerfile | Image Size | Build Time | Ollama Included | Best For |
|------------|------------|------------|-----------------|----------|
| `Dockerfile.simple` | 3-4 GB | 20-30 min | ❌ No | Sharing with others |
| `Dockerfile.ollama-runtime` | 4-5 GB | 30-45 min | ⚠️ Runtime | Self-contained, smaller |
| `Dockerfile.with-ollama` | 8-12 GB | 60-90 min | ✅ Pre-baked | Fully portable |

---

## Option 1: Simple (Recommended for Sharing)

```bash
cd backend
docker build -f Dockerfile.simple -t octavia:latest .
docker save octavia:latest -o octavia-simple.tar
```

**Use this when:**
- Sharing with others who have their own Ollama
- Faster builds are important
- You want smaller file sizes

**On another machine:**
```bash
docker load -i octavia-simple.tar
# Must have local Ollama running with TranslateGemma:4b
docker run -p 8000:8000 octavia:latest
```

---

## Option 2: Runtime Ollama (Balanced)

```bash
cd backend
docker build -f Dockerfile.ollama-runtime -t octavia-ollama-runtime:latest .
docker save octavia-ollama-runtime:latest -o octavia-ollama-runtime.tar
```

**Use this when:**
- You want Ollama included but smaller image
- First run downloads models (10-15 min)
- Acceptable for sharing

**On first run:**
- Automatically downloads TranslateGemma:4b (10-15 min)
- Subsequent runs are instant

---

## Option 3: Full Ollama (Fully Portable)

```bash
cd backend
docker build -f Dockerfile.with-ollama -t octavia-full:latest .
docker save octavia-full:latest -o octavia-full.tar
```

**Use this when:**
- You need everything pre-baked
- Image size doesn't matter
- 60-90 min build time is acceptable

**On another machine:**
```bash
docker load -i octavia-full.tar
docker run -p 8000:8000 -p 11434:11434 octavia-full:latest
```

---

## Build Scripts

### Windows
```batch
build-and-save.bat
```

### Linux/Mac
```bash
chmod +x build-and-save.sh
./build-and-save.sh
```

---

## Estimated File Sizes for Sharing

| Option | Compressed (.tar.gz) | Transfer Time (10 Mbps) |
|--------|----------------------|-------------------------|
| Simple | ~1.5 GB | ~20 minutes |
| Runtime Ollama | ~2 GB | ~27 minutes |
| Full Ollama | ~5-7 GB | ~1-1.5 hours |

---

## Recommendations

### For Friday Demo - Quick Sharing
**Use `Dockerfile.simple`**
- Fastest build (20-30 min)
- Smallest file size (~1.5 GB)
- Recipients need their own Ollama

### For Personal Use - Development
**Use `Dockerfile.dev`** (with hot reload)
- No rebuild needed for code changes
- Connect to local Ollama

### For Complete Portability
**Use `Dockerfile.with-ollama`**
- Everything included
- Works anywhere
- Longest build, largest file

---

## Notes

1. **First run on new machines:** AI models (Whisper, etc.) still need to download on first run (~10-20 min)

2. **Ollama port:** If using Ollama images, expose port 11434 alongside 8000

3. **Memory:** TranslateGemma:4b needs 4-8GB RAM to run

4. **Render deployment:** None of these include GPU support - for Render use CPU versions
