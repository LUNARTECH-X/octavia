"""
AI Orchestrator Module for Octavia Video Translator
Uses Llama.cpp for intelligent parameter optimization and dynamic processing
"""

import os
import sys
import json
import logging
import asyncio
import tempfile
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Real-time processing metrics for AI decision making"""
    chunk_id: int
    audio_duration_ms: float
    transcription_time_s: float
    translation_time_s: float
    tts_time_s: float
    whisper_confidence: float
    vad_speech_ratio: float
    memory_usage_mb: float
    gpu_utilization: float
    transcription_word_count: int
    translation_word_count: int
    compression_ratio: float

@dataclass
class AIDecision:
    """AI-driven parameter adjustments"""
    chunk_size_seconds: float
    vad_threshold: float
    whisper_model_size: str
    translation_temperature: float
    tts_voice_speed: float
    enable_gpu_acceleration: bool
    parallel_workers: int
    confidence_threshold: float
    reasoning: str

class AIOchestrator:
    """
    AI Orchestrator using Llama.cpp for intelligent video translation processing
    Dynamically adjusts parameters based on real-time analysis
    """

    def __init__(self, llama_model_path: str = None, gpu_layers: int = 0):
        self.llama_model_path = llama_model_path or self._find_llama_model()
        self.gpu_layers = gpu_layers
        self.llama_process = None
        self.metrics_history: List[ProcessingMetrics] = []
        self.decision_history: List[AIDecision] = []

        # Initialize with conservative defaults
        self.current_params = AIDecision(
            chunk_size_seconds=30.0,
            vad_threshold=0.5,
            whisper_model_size="base",
            translation_temperature=0.7,
            tts_voice_speed=1.0,
            enable_gpu_acceleration=True,
            parallel_workers=4,
            confidence_threshold=0.8,
            reasoning="Initial conservative settings"
        )

    def _find_llama_model(self) -> Optional[str]:
        """Find available Llama model files"""
        possible_paths = [
            "/models/llama-2-7b-chat.gguf",
            "/models/llama-2-13b-chat.gguf",
            "./models/llama-2-7b-chat.gguf",
            "./models/llama-2-13b-chat.gguf",
            "~/models/llama-2-7b-chat.gguf"
        ]

        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                logger.info(f"Found Llama model: {expanded_path}")
                return expanded_path

        logger.warning("No Llama model found - AI orchestration disabled")
        return None

    def start_llama_server(self) -> bool:
        """Start Llama.cpp server for AI decision making"""
        if not self.llama_model_path or not os.path.exists(self.llama_model_path):
            logger.warning("Llama model not available - using rule-based decisions")
            return False

        try:
            # Start Llama.cpp server
            cmd = [
                "llama-server",
                "-m", self.llama_model_path,
                "-c", "2048",  # Context size
                "--host", "127.0.0.1",
                "--port", "8080",
                "-ngl", str(self.gpu_layers),  # GPU layers
                "--threads", "4",
                "--mlock",  # Lock model in memory
                "--log-disable"  # Reduce log noise
            ]

            logger.info(f"Starting Llama.cpp server with model: {self.llama_model_path}")
            self.llama_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start
            import time
            time.sleep(3)

            # Test connection
            if self._test_llama_connection():
                logger.info("✓ Llama.cpp server started successfully")
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

    def _test_llama_connection(self) -> bool:
        """Test if Llama server is responding"""
        try:
            import requests
            response = requests.get("http://127.0.0.1:8080/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def analyze_audio_chunk(self, audio_path: str, chunk_id: int) -> Dict[str, Any]:
        """Analyze audio chunk for AI decision making"""
        try:
            # Load audio for analysis
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)

            # Calculate speech ratio using VAD-like analysis
            samples = np.array(audio.get_array_of_samples())
            samples = samples.astype(np.float32) / (2**15)

            # Simple energy-based VAD
            frame_length = int(audio.frame_rate * 0.025)  # 25ms frames
            hop_length = int(audio.frame_rate * 0.010)    # 10ms hop

            speech_frames = 0
            total_frames = 0

            for i in range(0, len(samples) - frame_length, hop_length):
                frame = samples[i:i + frame_length]
                energy = np.sqrt(np.mean(frame**2))
                total_frames += 1
                if energy > 0.02:  # Speech threshold
                    speech_frames += 1

            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0

            analysis = {
                "duration_ms": len(audio),
                "speech_ratio": speech_ratio,
                "estimated_words": int(len(audio) / 1000 * 3),  # Rough estimate
                "is_speech": speech_ratio > 0.3,
                "complexity_score": speech_ratio * (len(audio) / 30000)  # Duration factor
            }

            logger.info(f"Chunk {chunk_id} analysis: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {
                "duration_ms": 30000,
                "speech_ratio": 0.5,
                "estimated_words": 90,
                "is_speech": True,
                "complexity_score": 0.5
            }

    def make_processing_decision(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> AIDecision:
        """Use AI to make intelligent processing decisions"""
        if not self.llama_process or not self._test_llama_connection():
            # Fallback to rule-based decisions
            return self._rule_based_decision(metrics, audio_analysis)

        try:
            # Prepare context for AI decision
            context = self._prepare_ai_context(metrics, audio_analysis)

            # Query Llama for optimal parameters
            ai_response = self._query_llama_for_decision(context)

            # Parse AI response into decision
            decision = self._parse_ai_decision(ai_response)

            # Validate decision bounds
            decision = self._validate_decision_bounds(decision)

            logger.info(f"AI Decision: {decision.reasoning}")
            return decision

        except Exception as e:
            logger.error(f"AI decision failed, using rule-based: {e}")
            return self._rule_based_decision(metrics, audio_analysis)

    def _prepare_ai_context(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> str:
        """Prepare context data for AI decision making"""
        context = f"""
        CURRENT PROCESSING STATE:
        - Chunk ID: {metrics.chunk_id}
        - Audio Duration: {metrics.audio_duration_ms:.0f}ms
        - Speech Ratio: {audio_analysis['speech_ratio']:.2f}
        - Estimated Words: {audio_analysis['estimated_words']}
        - Complexity Score: {audio_analysis['complexity_score']:.2f}

        PERFORMANCE METRICS:
        - Transcription Time: {metrics.transcription_time_s:.2f}s
        - Translation Time: {metrics.translation_time_s:.2f}s
        - TTS Time: {metrics.tts_time_s:.2f}s
        - Whisper Confidence: {metrics.whisper_confidence:.2f}
        - Memory Usage: {metrics.memory_usage_mb:.0f}MB
        - GPU Utilization: {metrics.gpu_utilization:.1f}%

        PREVIOUS DECISIONS:
        """

        # Add recent decision history
        for i, decision in enumerate(self.decision_history[-3:]):  # Last 3 decisions
            context += f"""
            Decision {i+1}: {decision.reasoning}
            - Chunk Size: {decision.chunk_size_seconds:.1f}s
            - VAD Threshold: {decision.vad_threshold:.2f}
            - Workers: {decision.parallel_workers}
            """

        return context

    def _query_llama_for_decision(self, context: str) -> str:
        """Query Llama model for processing decision"""
        prompt = f"""
        You are an AI orchestrator optimizing video translation processing parameters.
        Analyze the current state and recommend optimal settings for the next chunk.

        {context}

        Based on this analysis, recommend optimal processing parameters:

        Respond in JSON format:
        {{
            "chunk_size_seconds": 15.0-120.0,
            "vad_threshold": 0.1-0.9,
            "whisper_model_size": "tiny|base|small|medium",
            "translation_temperature": 0.1-1.0,
            "tts_voice_speed": 0.8-1.3,
            "enable_gpu_acceleration": true/false,
            "parallel_workers": 1-16,
            "confidence_threshold": 0.5-0.95,
            "reasoning": "brief explanation of your decisions"
        }}

        Consider: performance, quality, resource usage, and speech characteristics.
        """

        try:
            import requests

            response = requests.post(
                "http://127.0.0.1:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 200,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stop": ["}}"]
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                ai_text = result.get("content", "")
                # Extract JSON from response
                json_start = ai_text.find('{')
                json_end = ai_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    return ai_text[json_start:json_end]

            logger.warning("Invalid AI response, using fallback")
            return "{}"

        except Exception as e:
            logger.error(f"Llama query failed: {e}")
            return "{}"

    def _parse_ai_decision(self, ai_response: str) -> AIDecision:
        """Parse AI response into decision object"""
        try:
            data = json.loads(ai_response)

            return AIDecision(
                chunk_size_seconds=max(15.0, min(120.0, data.get("chunk_size_seconds", 30.0))),
                vad_threshold=max(0.1, min(0.9, data.get("vad_threshold", 0.5))),
                whisper_model_size=data.get("whisper_model_size", "base"),
                translation_temperature=max(0.1, min(1.0, data.get("translation_temperature", 0.7))),
                tts_voice_speed=max(0.8, min(1.3, data.get("tts_voice_speed", 1.0))),
                enable_gpu_acceleration=data.get("enable_gpu_acceleration", True),
                parallel_workers=max(1, min(16, data.get("parallel_workers", 4))),
                confidence_threshold=max(0.5, min(0.95, data.get("confidence_threshold", 0.8))),
                reasoning=data.get("reasoning", "AI-optimized parameters")
            )

        except Exception as e:
            logger.error(f"Failed to parse AI decision: {e}")
            return self._rule_based_decision(ProcessingMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), {})

    def _validate_decision_bounds(self, decision: AIDecision) -> AIDecision:
        """Ensure decision parameters are within safe bounds"""
        decision.chunk_size_seconds = max(10.0, min(120.0, decision.chunk_size_seconds))
        decision.vad_threshold = max(0.1, min(0.9, decision.vad_threshold))
        decision.translation_temperature = max(0.1, min(1.0, decision.translation_temperature))
        decision.tts_voice_speed = max(0.7, min(1.4, decision.tts_voice_speed))
        decision.parallel_workers = max(1, min(8, decision.parallel_workers))  # Conservative limit
        decision.confidence_threshold = max(0.3, min(0.95, decision.confidence_threshold))

        return decision

    def _rule_based_decision(self, metrics: ProcessingMetrics, audio_analysis: Dict[str, Any]) -> AIDecision:
        """Fallback rule-based decision making"""
        # Adaptive chunk sizing based on speech density
        speech_ratio = audio_analysis.get('speech_ratio', 0.5)

        if speech_ratio < 0.2:  # Low speech density
            chunk_size = 60.0  # Longer chunks for efficiency
            workers = 2
        elif speech_ratio > 0.8:  # High speech density
            chunk_size = 20.0  # Shorter chunks for precision
            workers = 6
        else:  # Medium density
            chunk_size = 30.0
            workers = 4

        # Adjust based on performance
        if metrics.transcription_time_s > 10:  # Slow processing
            chunk_size = max(15.0, chunk_size * 0.8)  # Smaller chunks
            workers = min(workers, 2)  # Fewer workers

        return AIDecision(
            chunk_size_seconds=chunk_size,
            vad_threshold=0.5,
            whisper_model_size="base",
            translation_temperature=0.7,
            tts_voice_speed=1.0,
            enable_gpu_acceleration=True,
            parallel_workers=workers,
            confidence_threshold=0.8,
            reasoning=f"Rule-based: speech_ratio={speech_ratio:.2f}, adjusted for performance"
        )

    def update_metrics(self, metrics: ProcessingMetrics):
        """Update processing metrics history"""
        self.metrics_history.append(metrics)

        # Keep only recent history
        if len(self.metrics_history) > 50:
            self.metrics_history = self.metrics_history[-50:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for reporting"""
        if not self.metrics_history:
            return {"message": "No metrics available"}

        total_chunks = len(self.metrics_history)
        avg_transcription_time = np.mean([m.transcription_time_s for m in self.metrics_history])
        avg_translation_time = np.mean([m.translation_time_s for m in self.metrics_history])
        avg_tts_time = np.mean([m.tts_time_s for m in self.metrics_history])
        avg_confidence = np.mean([m.whisper_confidence for m in self.metrics_history])

        return {
            "total_chunks_processed": total_chunks,
            "average_transcription_time_s": round(avg_transcription_time, 2),
            "average_translation_time_s": round(avg_translation_time, 2),
            "average_tts_time_s": round(avg_tts_time, 2),
            "average_whisper_confidence": round(avg_confidence, 2),
            "ai_orchestration_active": bool(self.llama_process and self._test_llama_connection()),
            "decisions_made": len(self.decision_history)
        }

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_llama_server()
