"""
AI Orchestrator Module for Octavia Video Translator
Intelligent parameter optimization and dynamic processing
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
import psutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Real-time processing metrics for AI decision making"""
    chunk_id: int = 0
    audio_duration_ms: float = 0.0
    transcription_time_s: float = 0.0
    translation_time_s: float = 0.0
    tts_time_s: float = 0.0
    whisper_confidence: float = 0.0
    vad_speech_ratio: float = 0.5
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    transcription_word_count: int = 0
    translation_word_count: int = 0
    compression_ratio: float = 1.0
    success: bool = True
    error_message: str = ""


@dataclass
class AIDecision:
    """AI-driven parameter adjustments"""
    chunk_size_seconds: float = 30.0
    vad_threshold: float = 0.5
    whisper_model_size: str = "base"
    translation_temperature: float = 0.7
    tts_voice_speed: float = 1.0
    enable_gpu_acceleration: bool = True
    parallel_workers: int = 4
    confidence_threshold: float = 0.8
    enable_preprocessing: bool = True
    enable_quality_validation: bool = True
    reasoning: str = "Initial conservative settings"


class AIOchestrator:
    """
    AI Orchestrator for intelligent video translation processing
    Dynamically adjusts parameters based on real-time analysis
    Falls back to rule-based decisions when Llama is not available
    """

    def __init__(self, llama_model_path: str = None, gpu_layers: int = 0):
        self.llama_model_path = llama_model_path or self._find_llama_model()
        self.gpu_layers = gpu_layers
        self.llama_process = None
        self.llama_available = False
        self.llama = None
        self.using_ollama = False
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "qwen2.5-coder:1.5b"  # Fast, lightweight for quick decisions
        
        # Metrics history for learning
        self.metrics_history: deque = deque(maxlen=100)
        self.decision_history: deque = deque(maxlen=50)
        
        # Performance baselines
        self.baseline_transcription_time = 5.0  # seconds per 30s chunk
        self.baseline_tts_time = 3.0  # seconds per 30s chunk
        
        # Initialize with adaptive defaults
        self.current_decision = AIDecision(
            chunk_size_seconds=30.0,
            vad_threshold=0.5,
            whisper_model_size="base",
            translation_temperature=0.7,
            tts_voice_speed=1.0,
            enable_gpu_acceleration=True,
            parallel_workers=4,
            confidence_threshold=0.8,
            enable_preprocessing=True,
            enable_quality_validation=True,
            reasoning="Initial adaptive settings"
        )
        
        # System state
        self.system_metrics = self._get_system_metrics()
        
        # Try Ollama first (most reliable on Windows)
        if self._test_ollama_connection():
            self.llama_available = True
            self.using_ollama = True
            logger.info("[OK] Connected to Ollama server for AI decisions!")
        # Try to load Llama model directly using llama-cpp-python
        elif self.llama_model_path and os.path.exists(self.llama_model_path):
            self._load_llama_model()
        else:
            logger.info("No Llama/Ollama found - using intelligent rule-based orchestration")

    def _find_llama_model(self) -> Optional[str]:
        """Find available Llama model files"""
        # Model file patterns to search for
        model_patterns = [
            "llama-2-7b-chat.Q5_K_M.gguf",  # Best quality (user has this)
            "llama-2-7b-chat.Q4_K_M.gguf",  # Good quality
            "llama-2-7b-chat.Q5_K_S.gguf",
            "llama-2-7b-chat.Q6_K.gguf",
            "llama-2-7b-chat.gguf",
            "llama-3.2-3b-instruct.gguf",
            "llama-3-8b-instruct.gguf",
        ]
        
        # Base directories to search
        search_dirs = [
            "./backend/models",
            "./models",
            "~/models",
            "/models",
            ".",
        ]
        
        for search_dir in search_dirs:
            expanded_dir = os.path.expanduser(search_dir)
            if os.path.isdir(expanded_dir):
                for pattern in model_patterns:
                    model_path = os.path.join(expanded_dir, pattern)
                    if os.path.exists(model_path):
                        file_size = os.path.getsize(model_path) / (1024**3)  # GB
                        logger.info(f"Found Llama model: {model_path} ({file_size:.1f} GB)")
                        return model_path
        
        # Also check if llama-server is available
        try:
            result = subprocess.run(['where', 'llama-server'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"llama-server found at: {result.stdout.strip().split(chr(10))[0]}")
        except:
            pass
        
        return None

    def start_llama_server(self) -> bool:
        """Start Llama.cpp server for AI decision making"""
        if not self.llama_model_path:
            return False
            
        try:
            cmd = [
                "llama-server",
                "-m", self.llama_model_path,
                "-c", "2048",
                "--host", "127.0.0.1",
                "--port", "8080",
                "-ngl", str(self.gpu_layers),
                "--threads", "4",
                "--mlock",
                "--log-disable"
            ]
            
            logger.info(f"Starting Llama.cpp server with model: {self.llama_model_path}")
            self.llama_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            import time
            time.sleep(3)
            
            if self._test_llama_connection():
                self.llama_available = True
                logger.info("[OK] Llama.cpp server started successfully")
                return True
            else:
                logger.error("Llama.cpp server failed to start properly")
                self.stop_llama_server()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start Llama.cpp server: {e}")
            return False

    def stop_llama_server(self):
        """Stop the Llama.cpp server"""
        if self.llama_process:
            try:
                self.llama_process.terminate()
                self.llama_process.wait(timeout=5)
                logger.info("Llama.cpp server stopped")
            except Exception as e:
                logger.warning(f"Error stopping Llama server: {e}")
                try:
                    self.llama_process.kill()
                except:
                    pass
            finally:
                self.llama_process = None
                self.llama_available = False

    def _load_llama_model(self):
        """Load Llama model directly using llama-cpp-python API (no server needed)"""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading Llama model: {self.llama_model_path}")
            
            # Determine GPU layers (use all if GPU available, 0 for CPU)
            n_gpu_layers = self.gpu_layers if self.gpu_layers > 0 else -1  # -1 = all layers
            
            self.llama = Llama(
                model_path=self.llama_model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            
            self.llama_available = True
            logger.info("[OK] Llama model loaded successfully via llama-cpp-python!")
            
        except ImportError:
            logger.warning("llama-cpp-python not installed - using rule-based decisions")
            self.llama_available = False
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            self.llama_available = False
            self.llama = None

    def _query_llama_direct(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> Optional[AIDecision]:
        """Query loaded Llama model directly (no HTTP server needed)"""
        if not self.llama or not self.llama_available:
            return None
            
        try:
            context = self._prepare_llama_context(metrics, audio_analysis)
            
            prompt = f"""You are an AI orchestrator for video translation. 
Based on this analysis: {context}

Recommend optimal parameters in JSON format:
{{"chunk_size": 30, "model": "base", "workers": 4, "speed": 1.0, "temp": 0.7, "reasoning": "brief"}}"""

            response = self.llama(
                prompt,
                max_tokens=200,
                temperature=0.3,
                stop=["}"]
            )
            
            text = response["choices"][0]["text"]
            
            # Parse JSON from response
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0:
                data = json.loads(text[json_start:json_end])
                return AIDecision(
                    chunk_size_seconds=max(15, min(120, data.get("chunk_size", 30))),
                    whisper_model_size=data.get("model", "base"),
                    parallel_workers=max(1, min(8, data.get("workers", 4))),
                    tts_voice_speed=max(0.8, min(1.3, data.get("speed", 1.0))),
                    translation_temperature=max(0.3, min(0.9, data.get("temp", 0.7))),
                    reasoning=data.get("reasoning", "Llama-optimized")
                )
        except Exception as e:
            logger.warning(f"Direct Llama query error: {e}")
        
        return None

    def _test_ollama_connection(self) -> bool:
        """Test if Ollama server is responding"""
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama server connected successfully")
                return True
            return False
        except:
            return False

    def _query_ollama(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> Optional[AIDecision]:
        """Query Ollama server for processing decision"""
        if not self.llama_available or not self.using_ollama:
            return None

        try:
            import requests

            context = self._prepare_llama_context(metrics, audio_analysis)

            prompt = f"""You are an AI orchestrator for video translation.
Based on this analysis: {context}

Recommend optimal parameters in JSON format:
{{"chunk_size": 30, "model": "base", "workers": 4, "speed": 1.0, "temp": 0.7, "reasoning": "brief"}}"""

            # Respect LLM_TIMEOUT from environment, fallback to 300s
            env_timeout = int(os.getenv('LLM_TIMEOUT', 300))
            timeout = max(env_timeout, 60 if audio_analysis.get('speech_ratio', 0) > 0.6 else 30)

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200
                    }
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                
                # Parse JSON from response
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0:
                    data = json.loads(text[json_start:json_end])
                    return AIDecision(
                        chunk_size_seconds=max(15, min(120, data.get("chunk_size", 30))),
                        whisper_model_size=data.get("model", "base"),
                        parallel_workers=max(1, min(8, data.get("workers", 4))),
                        tts_voice_speed=max(0.8, min(1.3, data.get("speed", 1.0))),
                        translation_temperature=max(0.3, min(0.9, data.get("temp", 0.7))),
                        reasoning=data.get("reasoning", "Ollama-optimized")
                    )
        except Exception as e:
            logger.warning(f"Ollama query error: {e}")
        
        return None

    def _test_llama_connection(self) -> bool:
        """Test if Llama server is responding"""
        try:
            import requests
            response = requests.get("http://127.0.0.1:8080/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            gpu_info = {"available": False}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info["available"] = True
                    gpu_info["utilization"] = gpus[0].load * 100
                    gpu_info["memory_used"] = gpus[0].memoryUsed
                    gpu_info["memory_total"] = gpus[0].memoryTotal
                    gpu_info["temperature"] = gpus[0].temperature
            except:
                pass
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "gpu": gpu_info
            }
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
            return {"cpu_percent": 50, "memory_percent": 50, "memory_available_gb": 4, "gpu": {"available": False}}

    def analyze_audio_chunk(self, audio_path: str, chunk_id: int) -> Dict[str, Any]:
        """Analyze audio chunk for AI decision making"""
        try:
            from pydub import AudioSegment
            
            if not os.path.exists(audio_path):
                return self._default_analysis()
            
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            samples = np.array(audio.get_array_of_samples())
            samples = samples.astype(np.float32) / (2**15)
            
            # Energy-based VAD analysis
            frame_length = int(audio.frame_rate * 0.025)
            hop_length = int(audio.frame_rate * 0.010)
            
            speech_frames = 0
            total_frames = 0
            energies = []
            
            for i in range(0, len(samples) - frame_length, hop_length):
                frame = samples[i:i + frame_length]
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)
                total_frames += 1
                if energy > 0.02:
                    speech_frames += 1
            
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            # Analyze energy distribution for complexity
            energy_std = np.std(energies) if energies else 0
            energy_mean = np.mean(energies) if energies else 0
            
            # Estimate word count (roughly 3 words per second of speech)
            estimated_words = int((duration_ms / 1000) * speech_ratio * 3)
            
            # Calculate complexity score
            complexity_score = speech_ratio * (1 + energy_std * 10)
            
            analysis = {
                "duration_ms": duration_ms,
                "speech_ratio": speech_ratio,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "estimated_words": estimated_words,
                "is_speech": speech_ratio > 0.3,
                "complexity_score": min(1.0, complexity_score),
                "has_quiet_sections": any(e < 0.01 for e in energies),
                "has_loud_sections": any(e > 0.1 for e in energies)
            }
            
            logger.info(f"AI Orchestrator - Chunk {chunk_id} analysis: speech={speech_ratio:.2f}, complexity={complexity_score:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return self._default_analysis()

    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when audio analysis fails"""
        return {
            "duration_ms": 30000,
            "speech_ratio": 0.5,
            "energy_mean": 0.03,
            "energy_std": 0.02,
            "estimated_words": 90,
            "is_speech": True,
            "complexity_score": 0.5,
            "has_quiet_sections": False,
            "has_loud_sections": True
        }

    def make_processing_decision(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> AIDecision:
        """Make intelligent processing decisions based on current state"""
        # Store metrics for learning
        self.metrics_history.append(metrics)
        
        # Update system metrics
        self.system_metrics = self._get_system_metrics()
        
        # Try Ollama first (best for Windows)
        if self.using_ollama and self.llama_available:
            try:
                decision = self._query_ollama(metrics, audio_analysis)
                if decision:
                    self.current_decision = decision
                    self.decision_history.append(decision)
                    logger.info(f"AI Decision (Ollama): {decision.reasoning}")
                    return decision
            except Exception as e:
                logger.warning(f"Ollama query failed: {e}")
        
        # Try direct Llama API (no server needed)
        if self.llama_available and self.llama:
            try:
                decision = self._query_llama_direct(metrics, audio_analysis)
                if decision:
                    self.current_decision = decision
                    self.decision_history.append(decision)
                    logger.info(f"AI Decision (Llama): {decision.reasoning}")
                    return decision
            except Exception as e:
                logger.warning(f"Direct Llama query failed: {e}")
        
        # Fall back to server-based Llama if available
        if self.llama_available and self._test_llama_connection():
            try:
                decision = self._query_llama(metrics, audio_analysis)
                if decision:
                    self.current_decision = decision
                    self.decision_history.append(decision)
                    logger.info(f"AI Decision (Llama Server): {decision.reasoning}")
                    return decision
            except Exception as e:
                logger.warning(f"Llama server query failed: {e}")
        
        # Use intelligent rule-based decisions
        decision = self._adaptive_decision(metrics, audio_analysis)
        self.current_decision = decision
        self.decision_history.append(decision)
        
        return decision

    def _adaptive_decision(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> AIDecision:
        """Intelligent rule-based decision making with learning"""
        speech_ratio = audio_analysis.get('speech_ratio', 0.5)
        complexity = audio_analysis.get('complexity_score', 0.5)
        duration_ms = audio_analysis.get('duration_ms', 30000)
        has_quiet = audio_analysis.get('has_quiet_sections', False)
        has_loud = audio_analysis.get('has_loud_sections', True)
        
        # Calculate performance ratio
        total_time = metrics.transcription_time_s + metrics.translation_time_s + metrics.tts_time_s
        expected_time = (duration_ms / 30000) * (self.baseline_transcription_time + self.baseline_tts_time)
        performance_ratio = total_time / expected_time if expected_time > 0 else 1.0
        
        # Dynamic chunk sizing
        if speech_ratio < 0.2:
            chunk_size = 60.0
            workers = 2
            model_size = "tiny"
            reasoning_parts = ["low speech density"]
        elif speech_ratio > 0.8:
            chunk_size = 20.0
            workers = 6
            model_size = "small"
            reasoning_parts = ["high speech density"]
        else:
            chunk_size = 30.0
            workers = 4
            model_size = "base"
            reasoning_parts = ["medium speech density"]
        
        # Adjust based on complexity
        if complexity > 0.8:
            chunk_size = max(15.0, chunk_size * 0.7)
            model_size = "medium" if model_size in ["base", "small"] else model_size
            reasoning_parts.append("high complexity")
        
        # Adjust based on performance
        if performance_ratio > 1.5:
            chunk_size = max(15.0, chunk_size * 0.8)
            workers = min(workers, 2)
            reasoning_parts.append("slow processing")
        elif performance_ratio < 0.7:
            workers = min(workers + 1, 6)
            reasoning_parts.append("fast processing")
        
        # Adjust based on system resources
        if self.system_metrics.get("gpu", {}).get("available", False):
            gpu_util = self.system_metrics["gpu"].get("utilization", 0)
            if gpu_util > 80:
                workers = min(workers, 3)
                reasoning_parts.append("GPU busy")
            elif gpu_util < 30:
                workers = min(workers + 1, 6)
        
        memory_percent = self.system_metrics.get("memory_percent", 50)
        if memory_percent > 85:
            workers = min(workers, 2)
            model_size = "tiny" if model_size == "base" else model_size
            reasoning_parts.append("low memory")
        
        # Adjust for audio characteristics
        if has_quiet and not has_loud:
            chunk_size = min(chunk_size, 25.0)
            reasoning_parts.append("quiet audio")
        
        # Calculate optimal TTS speed for lip-sync
        tts_speed = 1.0
        if metrics.compression_ratio > 1.2:
            tts_speed = 0.9
            reasoning_parts.append("text expansion")
        elif metrics.compression_ratio < 0.8:
            tts_speed = 1.1
            reasoning_parts.append("text compression")

        if metrics.compression_ratio < 0.6:
            tts_speed = 1.2
            reasoning_parts.append("high compression (e.g., Chinese->English)")

        # Translation temperature based on content type
        temp = 0.7
        if complexity < 0.3:
            temp = 0.5
            reasoning_parts.append("simple content")
        elif complexity > 0.7:
            temp = 0.9
            reasoning_parts.append("complex content")
        
        return AIDecision(
            chunk_size_seconds=min(120.0, max(15.0, chunk_size)),
            vad_threshold=0.5,
            whisper_model_size=model_size,
            translation_temperature=temp,
            tts_voice_speed=min(1.3, max(0.8, tts_speed)),
            enable_gpu_acceleration=bool(self.system_metrics.get("gpu", {}).get("available", False)),
            parallel_workers=min(8, max(1, workers)),
            confidence_threshold=0.8,
            enable_preprocessing=True,
            enable_quality_validation=True,
            reasoning=f"Adaptive: {', '.join(reasoning_parts)}"
        )

    def _query_llama(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> Optional[AIDecision]:
        """Query Llama model for processing decision"""
        try:
            import requests
            
            context = self._prepare_llama_context(metrics, audio_analysis)
            
            prompt = f"""You are an AI orchestrator optimizing video translation. 
Analyze the current state and recommend optimal parameters.

{context}

Respond in JSON:
{{"chunk_size": 15-120, "model": "tiny/base/small/medium", "workers": 1-8, 
"speed": 0.8-1.3, "temp": 0.3-0.9, "reasoning": "brief explanation"}}"""

            response = requests.post(
                "http://127.0.0.1:8080/completion",
                json={"prompt": prompt, "n_predict": 150, "temperature": 0.3},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("content", "")
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start >= 0:
                    data = json.loads(text[json_start:json_end])
                    return AIDecision(
                        chunk_size_seconds=max(15, min(120, data.get("chunk_size", 30))),
                        whisper_model_size=data.get("model", "base"),
                        parallel_workers=max(1, min(8, data.get("workers", 4))),
                        tts_voice_speed=max(0.8, min(1.3, data.get("speed", 1.0))),
                        translation_temperature=max(0.3, min(0.9, data.get("temp", 0.7))),
                        reasoning=data.get("reasoning", "Llama-optimized")
                    )
        except Exception as e:
            logger.warning(f"Llama query error: {e}")
        return None

    def _prepare_llama_context(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> str:
        """Prepare context for Llama query"""
        return f"""
Chunk {metrics.chunk_id}: {metrics.audio_duration_ms:.0f}ms, speech={audio_analysis.get('speech_ratio', 0.5):.2f}
Performance: trans={metrics.transcription_time_s:.1f}s, tts={metrics.tts_time_s:.1f}s
Confidence: {metrics.whisper_confidence:.2f}, GPU: {self.system_metrics.get('gpu', {}).get('available', False)}
System: CPU={self.system_metrics.get('cpu_percent', 0):.0f}%, MEM={self.system_metrics.get('memory_percent', 0):.0f}%
"""
    
    def update_baselines(self, metrics: ProcessingMetrics):
        """Update performance baselines based on actual metrics"""
        if metrics.success:
            alpha = 0.1  # Learning rate
            self.baseline_transcription_time = alpha * metrics.transcription_time_s + (1 - alpha) * self.baseline_transcription_time
            self.baseline_tts_time = alpha * metrics.tts_time_s + (1 - alpha) * self.baseline_tts_time
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "llama_available": self.llama_available,
            "metrics_collected": len(self.metrics_history),
            "decisions_made": len(self.decision_history),
            "current_decision": {
                "chunk_size": self.current_decision.chunk_size_seconds,
                "model": self.current_decision.whisper_model_size,
                "workers": self.current_decision.parallel_workers,
                "reasoning": self.current_decision.reasoning
            },
            "baselines": {
                "transcription_time": round(self.baseline_transcription_time, 2),
                "tts_time": round(self.baseline_tts_time, 2)
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_llama_server()
