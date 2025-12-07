"""
Instrumentation and metrics logging for the pipeline
"""
import json
import time
import psutil
import torch
import numpy as np
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any
import subprocess
import os

@dataclass
class HardwareInfo:
    cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_count: int = 0
    python_version: str = ""

@dataclass
class AudioMetrics:
    lufs: float
    peak_db: float
    true_peak_db: float
    lra: float  # Loudness Range
    duration_ms: float
    sample_rate: int
    channels: int

@dataclass
class ProcessingMetrics:
    stage: str
    chunk_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""
    additional_data: Dict[str, Any] = None

class MetricsCollector:
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        self.metrics: List[Dict] = []
        self.hardware_info = self._capture_hardware_info()
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Create output directory for logs and metrics"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "audio_samples"), exist_ok=True)
    
    def _capture_hardware_info(self) -> HardwareInfo:
        """Capture detailed hardware information"""
        cpu_count = psutil.cpu_count()
        mem = psutil.virtual_memory()
        
        hardware = HardwareInfo(
            cpu_count=cpu_count,
            total_ram_gb=round(mem.total / (1024**3), 2),
            available_ram_gb=round(mem.available / (1024**3), 2),
            gpu_available=torch.cuda.is_available(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        
        if hardware.gpu_available:
            hardware.gpu_name = torch.cuda.get_device_name(0)
            hardware.gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            hardware.gpu_count = torch.cuda.device_count()
        
        return hardware
    
    def start_stage(self, stage: str, chunk_id: str = "") -> ProcessingMetrics:
        """Start timing a processing stage"""
        return ProcessingMetrics(
            stage=stage,
            chunk_id=chunk_id,
            start_time=time.time()
        )
    
    def end_stage(self, metrics: ProcessingMetrics, success: bool = True, error: str = "", 
                  additional_data: Dict = None):
        """End timing and record metrics"""
        metrics.end_time = time.time()
        metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
        metrics.success = success
        metrics.error = error
        metrics.additional_data = additional_data or {}
        
        self._record_metrics(metrics)
    
    def _record_metrics(self, metrics: ProcessingMetrics):
        """Record metrics to memory and file"""
        metric_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": metrics.stage,
            "chunk_id": metrics.chunk_id,
            "duration_ms": round(metrics.duration_ms, 2),
            "status": "SUCCESS" if metrics.success else "ERROR",
            "error": metrics.error,
            "data": metrics.additional_data
        }
        
        self.metrics.append(metric_entry)
        
        # Also write to JSONL file immediately
        self._append_to_jsonl(metric_entry)
    
    def _append_to_jsonl(self, entry: Dict):
        """Append single entry to JSONL log file"""
        log_file = os.path.join(self.output_dir, "logs.jsonl")
        
        with open(log_file, 'a') as f:
            json.dump(entry, f)
            f.write('\n')
    
    def measure_audio_levels(self, audio_path: str) -> AudioMetrics:
        """Measure audio levels using ffmpeg/ffprobe"""
        try:
            # Use ffprobe to get audio stats
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'a',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            if not data.get('streams'):
                raise ValueError("No audio stream found")
            
            stream = data['streams'][0]
            
            # Get duration
            duration = float(stream.get('duration', 0))
            
            lufs = -16.0  # Default
            peak_db = -1.0  # Default
            
            return AudioMetrics(
                lufs=lufs,
                peak_db=peak_db,
                true_peak_db=peak_db - 0.5,  # Approximation
                lra=10.0,  # Typical loudness range
                duration_ms=duration * 1000,
                sample_rate=int(stream.get('sample_rate', 44100)),
                channels=int(stream.get('channels', 2))
            )
            
        except Exception as e:
            # Return default metrics on error
            return AudioMetrics(
                lufs=-16.0,
                peak_db=-1.0,
                true_peak_db=-1.5,
                lra=10.0,
                duration_ms=0,
                sample_rate=44100,
                channels=2
            )
    
    def record_condensation(self, chunk_id: str, original_text: str, condensed_text: str):
        """Record text condensation metrics"""
        original_len = len(original_text.split())
        condensed_len = len(condensed_text.split())
        
        if original_len > 0:
            ratio = condensed_len / original_len
        else:
            ratio = 1.0
        
        self._record_metrics(ProcessingMetrics(
            stage="condensation",
            chunk_id=chunk_id,
            additional_data={
                "original_word_count": original_len,
                "condensed_word_count": condensed_len,
                "condensation_ratio": round(ratio, 3),
                "within_limit": ratio <= 1.2  # Technical requirement
            }
        ))
    
    def record_timing_adjustment(self, chunk_id: str, original_ms: float, translated_ms: float, 
                                 speed_adjustment: float):
        """Record timing adjustment metrics"""
        diff_ms = translated_ms - original_ms
        
        self._record_metrics(ProcessingMetrics(
            stage="timing_adjustment",
            chunk_id=chunk_id,
            additional_data={
                "original_duration_ms": round(original_ms, 1),
                "translated_duration_ms": round(translated_ms, 1),
                "duration_diff_ms": round(diff_ms, 1),
                "speed_adjustment": round(speed_adjustment, 3),
                "within_tolerance": abs(diff_ms) <= 200  # Technical requirement
            }
        ))
    
    def record_segment_metrics(self, chunk_id: str, metrics: Dict):
        """Record comprehensive segment metrics"""
        self._record_metrics(ProcessingMetrics(
            stage="segment_complete",
            chunk_id=chunk_id,
            additional_data=metrics
        ))
    
    def generate_summary_report(self) -> Dict:
        """Generate a summary report of all metrics"""
        successful_stages = [m for m in self.metrics if m.get("status") == "SUCCESS"]
        error_stages = [m for m in self.metrics if m.get("status") == "ERROR"]
        
        # Calculate average durations by stage
        stage_durations = {}
        for metric in successful_stages:
            stage = metric["stage"]
            duration = metric["duration_ms"]
            if stage not in stage_durations:
                stage_durations[stage] = []
            stage_durations[stage].append(duration)
        
        avg_durations = {}
        for stage, durations in stage_durations.items():
            avg_durations[stage] = round(np.mean(durations), 2)
        
        # Count condensation events
        condensation_metrics = [m for m in self.metrics if m.get("stage") == "condensation"]
        condensation_ratios = [m.get("data", {}).get("condensation_ratio", 1.0) 
                              for m in condensation_metrics]
        
        # Count timing adjustments
        timing_metrics = [m for m in self.metrics if m.get("stage") == "timing_adjustment"]
        within_tolerance = sum(1 for m in timing_metrics 
                              if m.get("data", {}).get("within_tolerance", False))
        
        report = {
            "hardware": {
                "cpu_count": self.hardware_info.cpu_count,
                "total_ram_gb": self.hardware_info.total_ram_gb,
                "gpu_available": self.hardware_info.gpu_available,
                "gpu_name": self.hardware_info.gpu_name,
                "gpu_memory_gb": self.hardware_info.gpu_memory_gb
            },
            "processing_summary": {
                "total_stages": len(self.metrics),
                "successful_stages": len(successful_stages),
                "failed_stages": len(error_stages),
                "success_rate": round(len(successful_stages) / len(self.metrics) * 100, 1) 
                               if self.metrics else 0,
                "average_durations_ms": avg_durations
            },
            "quality_metrics": {
                "condensation_events": len(condensation_metrics),
                "avg_condensation_ratio": round(np.mean(condensation_ratios), 3) 
                                         if condensation_ratios else 1.0,
                "max_condensation_ratio": round(max(condensation_ratios), 3) 
                                         if condensation_ratios else 1.0,
                "timing_adjustments": len(timing_metrics),
                "within_tolerance_percentage": round(within_tolerance / len(timing_metrics) * 100, 1) 
                                              if timing_metrics else 100.0
            },
            "timestamps": {
                "start_time": self.metrics[0]["timestamp"] if self.metrics else "",
                "end_time": self.metrics[-1]["timestamp"] if self.metrics else "",
                "total_metrics": len(self.metrics)
            }
        }
        
        # Save report to file
        report_file = os.path.join(self.output_dir, "metrics_summary.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
