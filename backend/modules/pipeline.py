"""
Complete video translation pipeline with subtitle generation
Optimized for FREE deployment using only open source tools
"""

import os
import sys
import json
import uuid
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil

import whisper
import torch
import numpy as np
import asyncio
from pydub import AudioSegment

# Import AI orchestrator
try:
    from modules.ai_orchestrator import AIOchestrator, ProcessingMetrics, AIDecision
    AI_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    AI_ORCHESTRATOR_AVAILABLE = False
    logging.warning("AI Orchestrator not available - using rule-based decisions")

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from modules.audio_translator import AudioTranslator, TranslationConfig, TranslationResult
    from modules.instrumentation import MetricsCollector
    from modules.subtitle_generator import SubtitleGenerator
    from modules.vocal_separator import VocalSeparator, DemucsModel
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Local modules not available, running in simplified mode")

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the video translation pipeline - FREE VERSION"""
    chunk_size: int = 40  # Optimized for faster processing (fewer chunks, more efficiency)
    max_chunk_size: int = 120
    min_chunk_size: int = 10
    timing_tolerance_ms: int = 200
    max_condensation_ratio: float = 1.2
    target_lufs: float = -16.0
    max_peak_db: float = -1.0
    # Use CUDA if available, otherwise CPU (both FREE)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_workers: int = 4  # Use multiple CPU cores (FREE)
    temp_dir: str = "/tmp/octavia"  # Use tmpfs for speed (FREE)
    output_dir: str = "backend/outputs"
    generate_subtitles: bool = True
    subtitle_formats: List[str] = None
    bilingual_subtitles: bool = True
    use_gpu: bool = torch.cuda.is_available()  # FREE if you have GPU
    cache_dir: str = "~/.cache/octavia"  # Local cache (FREE)
    parallel_processing: bool = True
    enable_model_caching: bool = True  # Cache models locally (FREE)
    use_faster_whisper: bool = True  # Open source optimization (FREE)
    crossfade_ms: int = 50  # Crossfade duration between chunks for smooth audio transitions
    use_vad: bool = False  # Disable VAD to preserve timing (default False for translation)
    vad_threshold: float = 0.5
    min_pause_duration_ms: int = 500
    enable_word_timestamps: bool = True
    use_semantic_chunking: bool = True
    enable_vocal_separation: bool = False  # Enable high-quality vocal/background separation
    vocal_separation_mode: str = "auto"    # "demucs", "uvr5", or "auto"
    use_nllb_translation: bool = True     # Use NLLB-200 for higher quality CJK translation

@dataclass
class VideoInfo:
    """Video metadata"""
    path: str
    duration: float
    width: int
    height: int
    codec: str
    audio_codec: str
    frame_rate: float
    bitrate: int
    has_audio: bool

@dataclass
class ChunkInfo:
    """Audio chunk information"""
    id: int
    path: str
    start_ms: float
    end_ms: float
    duration_ms: float
    has_speech: bool = True
    speech_confidence: float = 0.0

class VideoTranslationPipeline:
    """Main video translation pipeline - 100% FREE optimized version"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        if self.config.subtitle_formats is None:
            self.config.subtitle_formats = ["srt", "vtt"]
        
        # Setup device (FREE - uses existing hardware)
        if self.config.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
            logger.info(f"Using FREE GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (FREE)")
        
        self.whisper_model = None
        self.translator = None
        self.subtitle_generator = None
        self.metrics_collector = None
        self.vocal_separator = None
        self.model_cache = {}  # In-memory cache (FREE)

        # Multi-GPU support
        self.available_gpus = self._detect_available_gpus()
        self.ai_orchestrator = None

        # Initialize AI orchestrator if available
        self.ai_mode = "none"
        if AI_ORCHESTRATOR_AVAILABLE:
            try:
                self.ai_orchestrator = AIOchestrator()
                if self.ai_orchestrator.using_ollama:
                    self.ai_mode = "ollama"
                    logger.info("✓ AI Orchestrator: Ollama AI mode ACTIVE (deepseek-r1:latest)")
                elif self.ai_orchestrator.llama_available:
                    self.ai_mode = "llama"
                    logger.info("✓ AI Orchestrator: Llama.cpp AI mode ACTIVE")
                else:
                    self.ai_mode = "rule"
                    logger.info("AI Orchestrator: Rule-based mode (Ollama not available)")
            except Exception as e:
                logger.warning(f"AI Orchestrator initialization failed: {e}")
                self.ai_mode = "rule"

        # Initialize vocal separator if available
        if MODULES_AVAILABLE:
            try:
                self.vocal_separator = VocalSeparator.create(
                    force_cpu_fallback=(self.config.vocal_separation_mode == "uvr5" or not torch.cuda.is_available())
                )
                logger.info(f"VocalSeparator initialized: {self.vocal_separator}")
            except Exception as e:
                logger.warning(f"VocalSeparator initialization failed: {e}")

        # Create directories
        os.makedirs(self.config.temp_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.expanduser(self.config.cache_dir), exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Pre-warm models in background (FREE optimization)
        if self.config.use_gpu:
            asyncio.create_task(self._preload_models_async())
    
    def setup_logging(self):
        """Setup logging for the pipeline"""
        # Check if logger is already configured
        if len(logging.getLogger().handlers) > 0:
            return  # Already configured

        try:
            # Set up basic logging configuration with Unicode-safe format
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler('octavia_pipeline.log', encoding='utf-8')
                ]
            )

            # Add Secret Masker to sanitize logs
            root_logger = logging.getLogger()
            try:
                from services.logging_utils import SecretMasker
                masker = SecretMasker()
                for handler in root_logger.handlers:
                    handler.addFilter(masker)
                logger.info("✓ SecretMasker applied to pipeline logs")
            except ImportError:
                logger.warning("SecretMasker not available, logs will not be sanitized")

            # Set specific log levels for noisy libraries
            logging.getLogger('whisper').setLevel(logging.WARNING)
            logging.getLogger('transformers').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)

            logger.info("Pipeline logging configured successfully")

        except Exception as e:
            # Fallback to basic logging if setup fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.warning(f"Could not setup advanced logging: {e}")
    
    async def _preload_models_async(self):
        """Preload models asynchronously for faster first inference (FREE)"""
        try:
            logger.info("Preloading models in background (FREE optimization)...")
            # Load tiny model first to warm up
            temp_model = whisper.load_model("tiny", device=self.device)
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Models preloaded")
        except Exception as e:
            logger.warning(f"Model preload failed: {e}")
    
    def load_models(self, source_lang: str = "en", target_lang: str = "de"):
        """Load all required models with FREE optimizations"""
        try:
            # Check memory cache first (FREE)
            cache_key = f"models_loaded_{source_lang}_{target_lang}"
            if cache_key in self.model_cache:
                logger.info("Models already loaded (cached in memory)")
                return True
            
            logger.info("Loading Whisper model...")
            
            # Use faster-whisper if available (FREE and faster)
            if self.config.use_faster_whisper:
                try:
                    from faster_whisper import WhisperModel
                    self.whisper_model = WhisperModel(
                        "base",  # Smaller model for speed
                        device="cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu",
                        compute_type="float16" if self.config.use_gpu else "float32",
                        download_root=os.path.expanduser(self.config.cache_dir),
                        cpu_threads=4  # Use multiple CPU threads (FREE)
                    )
                    logger.info("Loaded faster-whisper (FREE optimization)")
                except ImportError:
                    # Fallback to original whisper
                    self.whisper_model = whisper.load_model(
                        "base",  # Smaller for speed
                        device=self.device,
                        download_root=os.path.expanduser(self.config.cache_dir)
                    )
                    logger.info("Loaded standard whisper")
            else:
                self.whisper_model = whisper.load_model(
                    "base",
                    device=self.device,
                    download_root=os.path.expanduser(self.config.cache_dir)
                )
            
            logger.info("Loading subtitle generator...")
            self.subtitle_generator = SubtitleGenerator(model_size="base")
            
            logger.info(f"Loading translation models for {source_lang}->{target_lang}...")
            
            # Setup translator
            translator_config = TranslationConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                auto_detect=True,
                use_gpu=self.config.use_gpu,
                cache_dir=self.config.cache_dir,
                model_size="small",  # Smaller models for speed
                # Pass VAD/Semantic framing config
                use_vad=self.config.use_vad,
                vad_threshold=self.config.vad_threshold,
                min_pause_duration_ms=self.config.min_pause_duration_ms,
                enable_word_timestamps=self.config.enable_word_timestamps,
                use_semantic_chunking=self.config.use_semantic_chunking,
                use_nllb_translation=self.config.use_nllb_translation
            )
            self.translator = AudioTranslator(translator_config)
            
            # Load models
            if not self.translator.load_models():
                raise Exception("Failed to load translation models")
            
            # Cache in memory (FREE)
            self.model_cache[cache_key] = True
            
            logger.info("All models loaded successfully with FREE optimizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def extract_audio_fast(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using FFmpeg with FREE optimizations"""
        try:
            logger.info(f"Extracting audio from {video_path}")
            
            # Use multi-threading for faster processing (FREE)
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                '-threads', '4',  # Use 4 threads (FREE)
                '-loglevel', 'error',
                audio_path
            ]
            
            # Run with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info(f"Audio extracted: {audio_path}")
                return True
            else:
                logger.error("Audio extraction failed: Output file empty")
                return False
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def chunk_audio_parallel(self, audio_path: str) -> List[ChunkInfo]:
        """Split audio into chunks with improved error handling and minimum size checks"""
        try:
            logger.info("Chunking audio with improved logic")

            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)

            # Minimum chunk size for Whisper (at least 1 second)
            min_chunk_size_ms = 2000  # 2 seconds minimum

            # If audio is very short, process as single chunk
            if duration_ms < min_chunk_size_ms:
                logger.info(f"Audio too short ({duration_ms}ms), processing as single chunk")
                chunk_path = os.path.join(self.config.temp_dir, "chunk_0000.wav")
                audio.export(chunk_path, format="wav")

                return [ChunkInfo(
                    id=0,
                    path=chunk_path,
                    start_ms=0,
                    end_ms=duration_ms,
                    duration_ms=duration_ms,
                    has_speech=True  # Assume speech for short clips
                )]

            # Calculate optimal chunk size
            min_chunks = 2
            # Allow up to 1000 chunks for very long videos (enough for ~11 hours at 40s chunks)
            max_chunks = 1000 
            calculated_chunks = duration_ms // (self.config.chunk_size * 1000)
            target_chunk_count = min(max_chunks, max(min_chunks, calculated_chunks))
            actual_chunk_size = duration_ms / target_chunk_count

            # Ensure minimum chunk size
            if actual_chunk_size < min_chunk_size_ms:
                actual_chunk_size = min_chunk_size_ms
                target_chunk_count = int(duration_ms / actual_chunk_size)

            chunks = []

            # Create chunks sequentially for better reliability
            for chunk_id in range(target_chunk_count):
                start_ms = int(chunk_id * actual_chunk_size)
                end_ms = int(min((chunk_id + 1) * actual_chunk_size, duration_ms))

                if end_ms - start_ms < min_chunk_size_ms:
                    continue  # Skip chunks that are too small

                chunk = self._create_chunk_safe(audio, chunk_id, start_ms, end_ms)
                if chunk:
                    chunks.append(chunk)

            # If no chunks were created, create one big chunk
            if not chunks:
                logger.warning("No valid chunks created, creating single chunk")
                chunk_path = os.path.join(self.config.temp_dir, "chunk_0000.wav")
                audio.export(chunk_path, format="wav")

                chunks.append(ChunkInfo(
                    id=0,
                    path=chunk_path,
                    start_ms=0,
                    end_ms=duration_ms,
                    duration_ms=duration_ms,
                    has_speech=True
                ))

            logger.info(f"Created {len(chunks)} audio chunks")
            return chunks

        except Exception as e:
            logger.error(f"Parallel chunking failed: {e}")
            # Fallback to simple chunking
            return self.chunk_audio_simple(audio_path)
    
    def chunk_audio_semantic(self, audio_path: str) -> List[ChunkInfo]:
        """Split audio into chunks using semantic boundaries (VAD + Sentences)"""
        try:
            logger.info("Chunking audio with SEMANTIC logic (VAD + Sentence Boundaries)")
            
            # Ensure models loaded
            if not self.translator:
                # Need translator for this
                self.load_models()
                
            # Get semantic cut points
            semantic_chunks = self.translator.get_semantic_chunks(audio_path)
            
            if not semantic_chunks:
                logger.warning("Semantic analysis returned no chunks, falling back to parallel chunking")
                return self.chunk_audio_parallel(audio_path)
                
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            final_chunks = []
            
            for i, sem_chunk in enumerate(semantic_chunks):
                start_ms = int(sem_chunk['start'] * 1000)
                end_ms = int(sem_chunk['end'] * 1000)
                
                # Safety checks
                if end_ms > duration_ms: end_ms = duration_ms
                if start_ms >= end_ms: continue
                
                chunk_path = os.path.join(self.config.temp_dir, f"chunk_{i:04d}.wav")
                
                # Export chunk
                chunk_audio = audio[start_ms:end_ms]
                chunk_audio.export(chunk_path, format="wav")
                
                # Create ChunkInfo
                final_chunks.append(ChunkInfo(
                    id=i,
                    path=chunk_path,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    has_speech=bool(sem_chunk['text'].strip())  # Empty text = silence
                ))
            
            logger.info(f"Created {len(final_chunks)} semantic audio chunks")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            return self.chunk_audio_parallel(audio_path)

    def chunk_audio_simple(self, audio_path: str) -> List[ChunkInfo]:
        """Simple fallback chunking method"""
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            
            chunks = []
            chunk_id = 0
            
            for start_ms in range(0, duration_ms, self.config.chunk_size * 1000):
                end_ms = min(start_ms + self.config.chunk_size * 1000, duration_ms)
                
                if start_ms >= end_ms:
                    break
                
                chunk = audio[start_ms:end_ms]
                chunk_path = os.path.join(self.config.temp_dir, f"chunk_{chunk_id:04d}.wav")
                chunk.export(chunk_path, format="wav")
                
                chunks.append(ChunkInfo(
                    id=chunk_id,
                    path=chunk_path,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    has_speech=True  # Assume speech for simple method
                ))
                chunk_id += 1
            
            logger.info(f"Created {len(chunks)} audio chunks (simple method)")
            return chunks
        except Exception as e:
            logger.error(f"Simple chunking failed: {e}")
            return []
    
    def _create_chunk(self, audio: AudioSegment, chunk_id: int, start_ms: int, end_ms: int) -> Optional[ChunkInfo]:
        """Create a single chunk"""
        try:
            if end_ms > len(audio):
                end_ms = len(audio)
            if start_ms >= end_ms:
                return None

            chunk = audio[start_ms:end_ms]
            chunk_path = os.path.join(self.config.temp_dir, f"chunk_{chunk_id:04d}.wav")
            chunk.export(chunk_path, format="wav")

            # Quick speech detection
            has_speech = self._quick_speech_check(chunk)

            return ChunkInfo(
                id=chunk_id,
                path=chunk_path,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                has_speech=has_speech
            )
        except Exception as e:
            logger.error(f"Failed to create chunk {chunk_id}: {e}")
            return None

    def _create_chunk_safe(self, audio: AudioSegment, chunk_id: int, start_ms: int, end_ms: int) -> Optional[ChunkInfo]:
        """Create a single chunk with better error handling"""
        try:
            if end_ms > len(audio):
                end_ms = len(audio)
            if start_ms >= end_ms or (end_ms - start_ms) < 1000:  # Minimum 1 second
                return None

            chunk = audio[start_ms:end_ms]
            if len(chunk) < 1000:  # Skip chunks shorter than 1 second
                return None

            chunk_path = os.path.join(self.config.temp_dir, f"chunk_{chunk_id:04d}.wav")
            chunk.export(chunk_path, format="wav")

            # Verify file was created
            if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                return None

            # Quick speech detection
            has_speech = self._quick_speech_check(chunk)

            return ChunkInfo(
                id=chunk_id,
                path=chunk_path,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                has_speech=has_speech
            )
        except Exception as e:
            logger.error(f"Failed to create chunk {chunk_id}: {e}")
            return None
    
    def _detect_available_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs for multi-GPU processing"""
        gpus = []

        try:
            # Check PyTorch CUDA availability
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total": torch.cuda.get_device_properties(i).total_memory,
                        "memory_free": torch.cuda.mem_get_info(i)[0] if hasattr(torch.cuda, 'mem_get_info') else 0,
                        "utilization": 0  # Will be updated during processing
                    }
                    gpus.append(gpu_info)

            # Also check GPUtil for additional info
            try:
                gpu_list = GPUtil.getGPUs()
                for i, gpu in enumerate(gpu_list):
                    if i < len(gpus):
                        gpus[i]["utilization"] = gpu.load * 100
                        gpus[i]["temperature"] = gpu.temperature
            except:
                pass

        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

        logger.info(f"Detected {len(gpus)} GPUs: {[g['name'] for g in gpus]}")
        return gpus

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU metrics
            gpu_metrics = {}
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_metrics[f"gpu_{i}"] = {
                        "utilization": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
            except:
                gpu_metrics = {"error": "GPUtil not available"}

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "gpu_metrics": gpu_metrics
            }

        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
            return {}

    def _quick_speech_check(self, audio_segment: AudioSegment) -> bool:
        """Fast speech detection"""
        try:
            samples = np.array(audio_segment.get_array_of_samples())
            samples = samples.astype(np.float32) / (2**15)

            # Simple RMS check
            rms = np.sqrt(np.mean(samples**2))
            return rms > 0.02
        except:
            return True  # Assume speech by default

    def _detect_source_language(self, video_path: str, min_confidence: float = 0.50) -> Optional[str]:
        """Detect the source language from the video audio with confidence threshold"""
        try:
            logger.info("Detecting source language from video audio...")

            # Try progressively longer samples if confidence is low
            sample_durations = [15, 30, 60]  # Try 15s, 30s, 60s samples
            last_confidence = 0.0
            detected_lang = None

            for duration in sample_durations:
                if duration > 60:
                    break

                # Extract audio sample for language detection
                temp_audio_path = os.path.join(self.config.temp_dir, "lang_detect.wav")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',  # 16kHz for Whisper
                    '-ac', '1',  # Mono
                    '-t', str(duration),
                    '-loglevel', 'error',
                    temp_audio_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0 or not os.path.exists(temp_audio_path):
                    logger.warning(f"Could not extract {duration}s audio sample for language detection")
                    continue

                # Check file size
                if os.path.getsize(temp_audio_path) < 1000:
                    logger.warning(f"Audio sample too small ({os.path.getsize(temp_audio_path)} bytes), trying longer sample")
                    continue

                # Load Whisper model if not already loaded
                if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                    try:
                        from faster_whisper import WhisperModel
                        detect_model = WhisperModel(
                            "tiny",  # Very small model for detection (faster)
                            device="cpu",
                            download_root=os.path.expanduser(self.config.cache_dir)
                        )
                    except ImportError:
                        import whisper
                        detect_model = whisper.load_model("tiny", device="cpu")
                else:
                    detect_model = self.whisper_model

                # Detect language with confidence
                try:
                    if hasattr(detect_model, 'transcribe'):
                        # faster-whisper
                        segments, info = detect_model.transcribe(
                            temp_audio_path,
                            language=None,  # Auto-detect
                            beam_size=1,  # Faster
                            vad_filter=self.config.use_vad  # Use config setting (default False to preserve timing)
                        )
                        detected_lang = info.language
                        confidence = info.language_probability if hasattr(info, 'language_probability') else 0.5
                    else:
                        # Original whisper
                        result = detect_model.transcribe(temp_audio_path, language=None, verbose=False)
                        detected_lang = result.get("language")
                        confidence = result.get("confidence", 0.5)

                    last_confidence = confidence
                    logger.info(f"Language detection ({duration}s sample): {detected_lang} (confidence: {confidence:.2%})")

                    # Clean up temp file
                    try:
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                    except:
                        pass

                    # Check if confidence meets threshold
                    if confidence >= min_confidence:
                        logger.info(f"✓ High confidence language detection: {detected_lang} ({confidence:.2%})")
                        return detected_lang
                    else:
                        logger.warning(f"⚠ Low confidence ({confidence:.2%} < {min_confidence:.0%}), trying longer sample...")
                        continue

                except Exception as detect_error:
                    logger.warning(f"Language detection failed for {duration}s sample: {detect_error}")
                    continue

            # If all samples had low confidence, return the best result we have
            if detected_lang and last_confidence > 0:
                logger.warning(f"⚠ Using best available detection: {detected_lang} ({last_confidence:.2%}) - consider manual source language selection")
                return detected_lang

            logger.warning("All language detection attempts failed")
            return None

        except Exception as e:
            logger.warning(f"Source language detection failed: {e}")
            return None
    
    def process_chunks_batch(self, chunks: List[ChunkInfo], target_lang: str = "de", job_id: str = None, jobs_db: Dict = None):
        """Process chunks in batches for efficiency with real-time progress updates"""
        translated_chunk_paths = []
        all_subtitle_segments = []

        # Group chunks by speech content
        speech_chunks = [c for c in chunks if c.has_speech]
        silent_chunks = [c for c in chunks if not c.has_speech]

        logger.info(f"Processing {len(speech_chunks)} speech chunks, {len(silent_chunks)} silent chunks")

        total_chunks = len(chunks)
        processed_chunks = 0

        # Handle silent chunks quickly
        for chunk in silent_chunks:
            output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
            silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
            silent_audio.export(output_path, format="wav")
            translated_chunk_paths.append((chunk.id, output_path))
            processed_chunks += 1

            # Update progress for silent chunks
            if job_id and jobs_db and job_id in jobs_db:
                progress_percent = 40 + int((processed_chunks / total_chunks) * 40)  # 40-80% range
                jobs_db[job_id]["progress"] = min(progress_percent, 80)
                jobs_db[job_id]["processed_chunks"] = processed_chunks
                jobs_db[job_id]["total_chunks"] = total_chunks
                jobs_db[job_id]["message"] = f"Processing audio chunks... ({processed_chunks}/{total_chunks})"

        # Process speech chunks in parallel for better performance
        logger.info(f"Processing {len(speech_chunks)} speech chunks in parallel")

        def process_chunk_with_progress(chunk):
            """Process a single chunk and return results"""
            chunk_start_time = datetime.now()

            try:
                # Get AI decision if orchestrator is available
                ai_decision = None
                if self.ai_orchestrator and self.ai_orchestrator.llama_available:
                    # Create metrics for this chunk
                    metrics = ProcessingMetrics(
                        chunk_id=chunk.id,
                        audio_duration_ms=chunk.duration_ms,
                        transcription_time_s=0,
                        translation_time_s=0,
                        tts_time_s=0,
                        whisper_confidence=0.0,
                        compression_ratio=1.0
                    )
                    audio_analysis = self.ai_orchestrator.analyze_audio_chunk(chunk.path, chunk.id)
                    ai_decision = self.ai_orchestrator.make_processing_decision(metrics, audio_analysis)
                    if ai_decision:
                        # Log based on actual mode used (check reasoning prefix)
                        reasoning = ai_decision.reasoning
                        if "Llama-optimized" in reasoning or "ollama" in reasoning.lower():
                            logger.info(f"AI Decision (Ollama): chunk={chunk.id}, size={ai_decision.chunk_size_seconds}s, model={ai_decision.whisper_model_size}, workers={ai_decision.parallel_workers}, reason={reasoning}")
                        elif "Llama" in reasoning:
                            logger.info(f"AI Decision (Llama): chunk={chunk.id}, size={ai_decision.chunk_size_seconds}s, model={ai_decision.whisper_model_size}, workers={ai_decision.parallel_workers}, reason={reasoning}")
                        else:
                            logger.info(f"AI Decision (Rule-based): chunk={chunk.id}, size={ai_decision.chunk_size_seconds}s, model={ai_decision.whisper_model_size}, workers={ai_decision.parallel_workers}, reason={reasoning}")
                    else:
                        logger.info(f"AI Decision (Rule-based): chunk={chunk.id} - fallback to defaults")
                else:
                    # Rule-based mode
                    logger.info(f"AI Decision (Rule-based): chunk={chunk.id} - no AI orchestrator available")
                result = self._process_single_chunk(chunk, target_lang)

                # Step 7: Update metrics and store for AI learning
                processing_time = (datetime.now() - chunk_start_time).total_seconds()

                if result:
                    # Save chunk for preview if job_id provided
                    if job_id:
                        try:
                            preview_dir = os.path.join(self.config.output_dir, "previews", job_id)
                            os.makedirs(preview_dir, exist_ok=True)

                            # Copy the translated chunk to preview directory
                            preview_path = os.path.join(preview_dir, f"chunk_{chunk.id:04d}.wav")
                            shutil.copy2(result["path"], preview_path)

                            # Update job with available chunks (thread-safe)
                            if jobs_db and job_id in jobs_db:
                                available_chunks = jobs_db[job_id].get('available_chunks', [])
                                chunk_info = {
                                    "id": chunk.id,
                                    "start_time": chunk.start_ms / 1000.0,  # Convert to seconds
                                    "duration": chunk.duration_ms / 1000.0,
                                    "preview_url": f"/api/download/chunk/{job_id}/{chunk.id}",
                                    "status": "completed",
                                    "confidence_score": result.get("stt_confidence_score", 0.0),
                                    "estimated_wer": result.get("estimated_wer", 0.0),
                                    "quality_rating": result.get("quality_rating", "unknown")
                                }
                                available_chunks.append(chunk_info)

                        except Exception as preview_error:
                            logger.warning(f"Failed to save preview for chunk {chunk.id}: {preview_error}")

                    logger.info(f"[OK] Chunk {chunk.id} processed successfully in {processing_time:.1f}s")
                    return chunk.id, result, processing_time
                else:
                    logger.warning(f"Chunk {chunk.id} returned no result")
                    # Create silent fallback
                    output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                    silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
                    silent_audio.export(output_path, format="wav")
                    return chunk.id, {"path": output_path, "segments": []}, processing_time

            except Exception as e:
                logger.error(f"Chunk {chunk.id} failed: {e}")
                # Create silent fallback
                output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
                silent_audio.export(output_path, format="wav")
                return chunk.id, {"path": output_path, "segments": []}, (datetime.now() - chunk_start_time).total_seconds()

        # Process chunks in parallel using ThreadPoolExecutor
        # Optimized for CPU: If no GPU, force sequential processing to avoid Ollama timeouts
        if not self.config.use_gpu:
            logger.info("CPU mode detected: Limiting to 3 workers to balance speed and stability")
            max_workers = 3
        else:
            # Optimized for 6-core CPU: 5 workers gives best speed (1 core free for system)
            max_workers = min(len(speech_chunks), 5)
            logger.info(f"GPU mode detected: Using {max_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {executor.submit(process_chunk_with_progress, chunk): chunk for chunk in speech_chunks}

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_id, result, processing_time = future.result()

                    # Store results
                    translated_chunk_paths.append((chunk_id, result["path"]))
                    if result.get("segments"):
                        all_subtitle_segments.append(result["segments"])

                    # Update progress
                    processed_chunks += 1
                    if job_id and jobs_db and job_id in jobs_db:
                        progress_percent = 40 + int((processed_chunks / total_chunks) * 40)
                        jobs_db[job_id]["progress"] = min(progress_percent, 85)
                        jobs_db[job_id]["processed_chunks"] = processed_chunks
                        jobs_db[job_id]["total_chunks"] = total_chunks
                        jobs_db[job_id]["message"] = f"Completed chunk {processed_chunks}/{total_chunks}"

                except Exception as exc:
                    logger.error(f'Chunk {chunk.id} generated an exception: {exc}')
                    processed_chunks += 1

        # Final progress update before merging
        if job_id and jobs_db and job_id in jobs_db:
            jobs_db[job_id]["progress"] = 85
            jobs_db[job_id]["message"] = "Merging audio chunks..."

        # Sort by chunk ID
        translated_chunk_paths.sort(key=lambda x: x[0])
        return [path for _, path in translated_chunk_paths], all_subtitle_segments
    
    def _process_single_chunk(self, chunk: ChunkInfo, target_lang: str) -> Optional[Dict]:
        """Process a single chunk"""
        try:
            self.translator.config.target_lang = target_lang

            # FAST-PATH: Skip entire pipeline if source == target language
            if self.translator.config.source_lang == target_lang:
                output_path = os.path.join(self.config.temp_dir, f"passthrough_chunk_{chunk.id:04d}.wav")
                shutil.copy(chunk.path, output_path)
                return {
                    "path": output_path,
                    "segments": [],
                    "stt_confidence_score": 1.0,
                    "estimated_wer": 0.0,
                    "quality_rating": "passthrough"
                }

            result = self.translator.process_audio(chunk.path)
            
            if result.success:
                new_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                shutil.move(result.output_path, new_path)

                return {
                    "path": new_path,
                    "segments": result.timing_segments if result.timing_segments else [],
                    "stt_confidence_score": result.stt_confidence_score,
                    "estimated_wer": result.estimated_wer,
                    "quality_rating": result.quality_rating
                }
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk.id}: {e}")
        
        return None

    def combine_audio_chunks_with_crossfade(self, chunk_paths: List[str], crossfade_ms: int = None) -> Optional[AudioSegment]:
        """Combine audio chunks with crossfade transitions for smooth audio

        Args:
            chunk_paths: List of file paths to audio chunks (in order)
            crossfade_ms: Crossfade duration in milliseconds. If None, uses config.crossfade_ms

        Returns:
            Combined AudioSegment or None if failed
        """
        if crossfade_ms is None:
            crossfade_ms = self.config.crossfade_ms

        if not chunk_paths:
            logger.warning("No chunk paths provided for combining")
            return None

        try:
            chunks = []
            for path in chunk_paths:
                if path and os.path.exists(path):
                    try:
                        chunk = AudioSegment.from_file(path)
                        chunks.append(chunk)
                    except Exception as chunk_error:
                        logger.warning(f"Failed to load chunk {path}: {chunk_error}")
                else:
                    logger.warning(f"Chunk path does not exist: {path}")

            if not chunks:
                logger.error("No valid chunks to combine")
                return None

            if len(chunks) == 1:
                logger.info("Single chunk, no crossfade needed")
                return chunks[0]

            logger.info(f"Combining {len(chunks)} chunks with {crossfade_ms}ms crossfade")

            combined = chunks[0]
            for i, chunk in enumerate(chunks[1:], start=1):
                chunk_length_ms = len(chunk)
                actual_crossfade = min(crossfade_ms, chunk_length_ms // 2)

                if actual_crossfade > 0:
                    combined = combined.append(chunk, crossfade=actual_crossfade)
                else:
                    combined += chunk

                logger.debug(f"Added chunk {i} with {actual_crossfade}ms crossfade")

            logger.info(f"Successfully combined {len(chunks)} chunks ({len(combined)}ms total)")
            return combined

        except Exception as e:
            logger.error(f"Failed to combine audio chunks: {e}")
            return None

    def merge_files_fast(self, video_path: str, audio_path: str, output_path: str, instrumental_path: str = None) -> bool:
        """Merge audio and video quickly, optionally mixing with background audio"""
        try:
            input_audio = audio_path
            
            # If we have an instrumental track, mix it with the translated vocals
            if instrumental_path and os.path.exists(instrumental_path):
                logger.info(f"Mixing translated vocals with background music: {instrumental_path}")
                mixed_audio = os.path.join(self.config.temp_dir, f"mixed_final_{uuid.uuid4().hex[:8]}.wav")
                
                # Combine using ffmpeg amix filter for speed and quality
                cmd = [
                    'ffmpeg', '-y',
                    '-i', audio_path,        # Input 0: Translated vocals
                    '-i', instrumental_path, # Input 1: Original background
                    '-filter_complex', 'amix=inputs=2:duration=first:dropout_transition=2',
                    '-ac', '2',
                    '-ar', '44100',
                    mixed_audio
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if os.path.exists(mixed_audio):
                    input_audio = mixed_audio
                    logger.info("Audio mixing complete")
                else:
                    logger.warning("Audio mixing failed, using translated vocals only")

            # Use FFmpeg with optimized settings to merge with video
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', input_audio,
                '-c:v', 'copy',  # Copy video stream (fast)
                '-c:a', 'aac',
                '-b:a', '128k',  # Lower bitrate for speed
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-threads', '4',  # Multi-threading
                '-loglevel', 'error',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Merged video created: {output_path}")
                return True
            else:
                logger.error(f"Merge failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.config.temp_dir):
                for file in os.listdir(self.config.temp_dir):
                    file_path = os.path.join(self.config.temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except:
                        pass
                logger.info("Cleaned up temp files")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def calculate_duration_metrics(self, original_duration_ms: float, translated_audio_path: str) -> Dict[str, Any]:
        """Calculate duration match metrics between original and translated audio"""
        try:
            if not os.path.exists(translated_audio_path):
                return {
                    "duration_match_percent": 0.0,
                    "duration_diff_ms": 0,
                    "duration_diff_percent": 0.0,
                    "within_tolerance": False
                }
            
            translated_audio = AudioSegment.from_file(translated_audio_path)
            translated_duration_ms = len(translated_audio)
            
            duration_diff_ms = translated_duration_ms - original_duration_ms
            duration_diff_percent = (duration_diff_ms / original_duration_ms * 100) if original_duration_ms > 0 else 0
            
            duration_match_percent = min(100, max(0, 100 - abs(duration_diff_percent)))
            
            tolerance_percent = 15.0  # 15% tolerance
            within_tolerance = abs(duration_diff_percent) <= tolerance_percent
            
            logger.info(f"Duration metrics: original={original_duration_ms:.0f}ms, "
                       f"translated={translated_duration_ms:.0f}ms, "
                       f"diff={duration_diff_percent:+.1f}%, "
                       f"match={duration_match_percent:.1f}%, "
                       f"within_tolerance={within_tolerance}")
            
            return {
                "duration_match_percent": duration_match_percent,
                "duration_diff_ms": duration_diff_ms,
                "duration_diff_percent": duration_diff_percent,
                "original_duration_ms": original_duration_ms,
                "translated_duration_ms": translated_duration_ms,
                "within_tolerance": within_tolerance
            }
            
        except Exception as e:
            logger.error(f"Duration metrics calculation failed: {e}")
            return {
                "duration_match_percent": 0.0,
                "duration_diff_ms": 0,
                "duration_diff_percent": 0.0,
                "within_tolerance": False,
                "error": str(e)
            }
    
    def _update_job_progress(self, job_id: str, progress: int, message: str, chunks_processed: int = None, total_chunks: int = None, available_chunks: list = None, jobs_db: Dict = None):
        """Update job progress in jobs_db"""
        if job_id and jobs_db and job_id in jobs_db:
            try:
                jobs_db[job_id]["progress"] = progress
                jobs_db[job_id]["message"] = message
                if chunks_processed is not None:
                    jobs_db[job_id]["chunks_processed"] = chunks_processed
                if total_chunks is not None:
                    jobs_db[job_id]["total_chunks"] = total_chunks
                if available_chunks is not None:
                    jobs_db[job_id]["available_chunks"] = available_chunks
            except Exception as e:
                logger.warning(f"Failed to update job progress: {e}")
                pass  # Silently fail if jobs_db update fails

    def process_video_fast(self, video_path: str, target_lang: str = "de", source_lang: str = None, job_id: str = None, jobs_db: Dict = None) -> Dict[str, Any]:
        """Fast video translation pipeline - optimized for FREE deployment"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting FAST video translation")
            logger.info(f"Input: {video_path}")
            logger.info(f"Target: {target_lang}")
            logger.info(f"Using GPU: {self.config.use_gpu}")
            
            # Initial progress update
            if job_id:
                self._update_job_progress(job_id, 5, "Initializing translation pipeline...", jobs_db=jobs_db)

            # 1. Load models (cached)
            logger.info("1. Loading models...")
            if job_id:
                self._update_job_progress(job_id, 10, "Loading AI translation models...", jobs_db=jobs_db)
            # Detect source language if not provided
            if source_lang is None:
                source_lang = self._detect_source_language(video_path) or "en"
            logger.info(f"Detected/using source language: {source_lang}")
            if not self.load_models(source_lang=source_lang, target_lang=target_lang):
                return {
                    "success": False,
                    "error": "Failed to load models",
                    "processing_time_s": 0,
                    "output_path": "",
                    "target_language": target_lang
                }
            
            # 2. Extract and pre-process audio
            logger.info("2. Extracting audio...")
            if job_id:
                self._update_job_progress(job_id, 15, "Extracting audio for analysis...", jobs_db=jobs_db)
            
            temp_audio = os.path.join(self.config.temp_dir, f"audio_{uuid.uuid4().hex[:8]}.wav")
            if not self.extract_audio_fast(video_path, temp_audio):
                raise Exception("Audio extraction failed")
            
            # Calculate original audio duration for metrics
            from pydub import AudioSegment
            original_audio = AudioSegment.from_file(temp_audio)
            original_duration_ms = len(original_audio)
            logger.info(f"Original audio duration: {original_duration_ms:.0f}ms")
            
            # --- VOCAL SEPARATION INTEGRATION ---
            instrumental_audio = None
            vocal_audio = temp_audio
            
            if self.config.enable_vocal_separation and self.vocal_separator:
                try:
                    logger.info("🎬 Magic Mode: Separating vocals from background music...")
                    if job_id:
                        self._update_job_progress(job_id, 20, "Separating vocals from background (Magic Mode)...", jobs_db=jobs_db)
                    
                    separation_dir = os.path.join(self.config.temp_dir, "separation")
                    separation_result = self.vocal_separator.separate_with_fallback(vocal_audio, separation_dir)
                    
                    vocal_audio = separation_result.vocals_path
                    instrumental_audio = separation_result.instrumental_path
                    logger.info(f"Vocal separation complete. Vocals: {vocal_audio}, Instrumental: {instrumental_audio}")
                except Exception as e:
                    logger.warning(f"Vocal separation failed, continuing with original audio: {e}")
                    vocal_audio = temp_audio
                    instrumental_audio = None
            # ------------------------------------
            
            # 3. Chunk audio
            logger.info("3. Chunking audio...")
            if job_id:
                self._update_job_progress(job_id, 20, "Splitting video into chunks...", jobs_db=jobs_db)
            
            # Use semantic chunking ONLY if we have separated vocals (as requested)
            if self.config.use_vad and self.config.use_semantic_chunking and instrumental_audio is not None:
                logger.info("🎤 Using Semantic Chunking on Separated Vocals")
                chunks = self.chunk_audio_semantic(vocal_audio)
            else:
                logger.info("⏱️ Using Standard Time-based Chunking")
                chunks = self.chunk_audio_parallel(vocal_audio)
                
            if not chunks:
                return {
                    "success": False,
                    "error": "Chunking failed",
                    "processing_time_s": (datetime.now() - start_time).total_seconds(),
                    "output_path": "",
                    "target_language": target_lang
                }

            total_chunks = len(chunks)
            logger.info(f"Created {total_chunks} audio chunks")
            if job_id:
                self._update_job_progress(job_id, 25, f"Processing {total_chunks} audio chunks...", total_chunks=total_chunks, jobs_db=jobs_db)
            
            # 4. Process chunks in batch
            logger.info(f"4. Processing {len(chunks)} chunks...")
            if job_id:
                self._update_job_progress(job_id, 30, f"Starting TTS generation for {total_chunks} chunks...", jobs_db=jobs_db)
            translated_paths, subtitle_segments = self.process_chunks_batch(chunks, target_lang, job_id, jobs_db)
            
            if len(translated_paths) != len(chunks):
                logger.warning(f"Only {len(translated_paths)}/{len(chunks)} chunks processed successfully")
            
            # 5. Merge chunks
            logger.info("5. Merging audio chunks...")
            if job_id:
                self._update_job_progress(job_id, 80, "Merging translated audio chunks...", jobs_db=jobs_db)
            merged_audio = os.path.join(self.config.temp_dir, "merged.wav")

            # Merge translated audio chunks in correct order
            if translated_paths:
                # Filter out any missing files and sort by chunk ID
                valid_paths = [(i, path) for i, path in enumerate(translated_paths) if path and os.path.exists(path)]
                valid_paths.sort(key=lambda x: x[0])  # Sort by original index

                if valid_paths:
                    try:
                        valid_chunk_paths = [path for _, path in valid_paths]
                        combined = self.combine_audio_chunks_with_crossfade(valid_chunk_paths)

                        if combined:
                            combined.export(merged_audio, format="wav")
                            logger.info(f"Successfully merged {len(valid_paths)} audio chunks to {merged_audio}")
                        else:
                            raise Exception("combine_audio_chunks_with_crossfade returned None")

                    except Exception as merge_error:
                        logger.error(f"Failed to merge audio chunks: {merge_error}")
                        # Fallback: copy first available chunk
                        first_idx, first_path = valid_paths[0]
                        try:
                            shutil.copy2(first_path, merged_audio)
                            logger.warning(f"Fallback: copied first chunk {first_idx} as merged audio")
                        except Exception as fallback_error:
                            logger.error(f"Fallback audio merge also failed: {fallback_error}")
                            return {
                                "success": False,
                                "error": "Audio merging failed",
                                "processing_time_s": (datetime.now() - start_time).total_seconds(),
                                "output_path": "",
                                "target_language": target_lang
                            }
                else:
                    logger.error("No valid translated audio chunks found")
                    return {
                        "success": False,
                        "error": "No translated audio chunks available",
                        "processing_time_s": (datetime.now() - start_time).total_seconds(),
                        "output_path": "",
                        "target_language": target_lang
                    }
            else:
                logger.error("No translated audio paths returned from processing")
                return {
                    "success": False,
                    "processing_time_s": (datetime.now() - start_time).total_seconds(),
                    "output_path": "",
                    "target_language": target_lang
                }
            
            logger.info("6. Merging with video...")
            logger.info(f"Job ID: {job_id}")
            if job_id:
                output_filename = f"translated_video_{job_id}.mp4"
                logger.info(f"Using job_id filename: {output_filename}")
            else:
                output_filename = f"translated_{os.path.basename(video_path)}"
                logger.info(f"Using fallback filename: {output_filename}")
            output_path = os.path.join(self.config.output_dir, output_filename)
            logger.info(f"Final output path: {output_path}")
            
            logger.info(f"Calling merge_files_fast with output_path: {output_path}")
            if not self.merge_files_fast(video_path, merged_audio, output_path, instrumental_path=instrumental_audio):
                return {
                    "error": "Video merge failed",
                    "processing_time_s": (datetime.now() - start_time).total_seconds(),
                    "output_path": "",
                    "target_language": target_lang
                }
            
            # 7. Generate subtitles if requested
            subtitle_files = {}
            if self.config.generate_subtitles and subtitle_segments:
                logger.info("7. Generating subtitles...")
                # Flatten segments
                all_segments = []
                for seg_list in subtitle_segments:
                    all_segments.extend(seg_list)

                base_name = os.path.splitext(output_path)[0]
                srt_path = f"{base_name}.srt"

                # Create simple SRT
                srt_content = ""
                for i, seg in enumerate(all_segments, 1):
                    start = seg.get("start", 0)
                    end = seg.get("end", start + 5)
                    text = seg.get("text", "")

                    def format_time(seconds):
                        h = int(seconds // 3600)
                        m = int((seconds % 3600) // 60)
                        s = int(seconds % 60)
                        ms = int((seconds - int(seconds)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

                    srt_content += f"{i}\n"
                    srt_content += f"{format_time(start)} --> {format_time(end)}\n"
                    srt_content += f"{text}\n\n"

                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)

                subtitle_files["srt"] = srt_path
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate duration metrics
            duration_metrics = self.calculate_duration_metrics(original_duration_ms, merged_audio)
            
            # Cleanup
            self.cleanup_temp_files()
            
            result = {
                "success": True,
                "output_path": output_path,  # Fixed key name
                "target_language": target_lang,
                "processing_time_s": total_time,
                "subtitle_files": subtitle_files,
                "total_chunks": len(chunks),
                "message": f"Translation completed in {total_time:.1f}s",
                # Add missing keys for CLI compatibility
                "successful_chunks": len(chunks),
                "duration_match_within_tolerance": duration_metrics.get("within_tolerance", True),
                "avg_duration_diff_ms": duration_metrics.get("duration_diff_ms", 0),
                "avg_condensation_ratio": 1.0,
                "total_time_seconds": total_time,
                # Duration metrics
                "duration_match_percent": duration_metrics.get("duration_match_percent", 100.0),
                "duration_diff_percent": duration_metrics.get("duration_diff_percent", 0.0),
                "original_duration_ms": duration_metrics.get("original_duration_ms", 0),
                "translated_duration_ms": duration_metrics.get("translated_duration_ms", 0)
            }
            
            logger.info(f"[OK] Translation completed in {total_time:.1f}s")
            logger.info(f"  Output: {output_path}")
            if subtitle_files:
                logger.info(f"  Subtitles: {list(subtitle_files.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            self.cleanup_temp_files()
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_s": (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0,
                "output_path": "",
                "target_language": target_lang,
                "duration_match_within_tolerance": False,
                "duration_match_percent": 0.0,
                "avg_duration_diff_ms": 0,
                "successful_chunks": 0,
                "total_chunks": 0,
                "message": f"Translation failed: {str(e)}"
            }
    
    def process_video(self, video_path: str, target_lang: str = "de") -> Dict[str, Any]:
        """Alias for process_video_fast for compatibility"""
        return self.process_video_fast(video_path, target_lang)
    

# Command-line interface
def main():
    """Command-line interface for video translation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Octavia Video Translator (FREE Optimized)")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output directory (default: outputs)")
    parser.add_argument("--target", "-t", default="de", help="Target language")
    parser.add_argument("--chunk-size", "-c", type=int, default=30, help="Chunk size in seconds")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU even if available")
    parser.add_argument("--fast", action="store_true", help="Use fastest settings")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        use_gpu=not args.no_gpu and torch.cuda.is_available(),
        parallel_processing=args.fast or args.workers > 1
    )
    
    if args.fast:
        config.use_faster_whisper = True
        config.enable_model_caching = True
    
    pipeline = VideoTranslationPipeline(config)
    
    print(f"\n{'='*60}")
    print(f"Octavia Video Translator (FREE Optimized)")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Target language: {args.target}")
    print(f"Using GPU: {config.use_gpu}")
    print(f"Parallel workers: {config.max_workers}")
    print(f"{'='*60}\n")
    
    # Process video
    result = pipeline.process_video_fast(args.input, args.target)
    
    # Print result
    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    
    if result["success"]:
        print(f"[SUCCESS]")
        print(f"  Output: {result['output_video']}")
        print(f"  Processing time: {result['processing_time_s']:.1f}s")
        print(f"  Chunks: {result['chunks_processed']}/{result['total_chunks']}")

        if result.get('subtitle_files'):
            print(f"  Subtitles generated:")
            for name, path in result['subtitle_files'].items():
                if path:
                    print(f"    - {name}: {os.path.basename(path)}")
    else:
        print(f"[FAILED]")
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"{'='*60}\n")
    
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
