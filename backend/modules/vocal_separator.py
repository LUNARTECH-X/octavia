"""
UVR5 Vocal Separator Module with CPU Fallback
Separates vocals from background music for "Magic Mode" dubbing

This module provides automatic fallback:
- Uses Demucs on GPU for best quality
- Falls back to UVR5-ONNX (audio-separator) on CPU for memory efficiency
"""

import os
import sys
import logging
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file using soundfile to avoid torchcodec issues"""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        return audio, sr
    except ImportError:
        try:
            import scipy.io.wavfile as wav
            sr, audio = wav.read(audio_path)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise RuntimeError(f"Failed to load audio: {e}")


def save_audio(audio_path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio file using wave module to avoid torchcodec issues"""
    try:
        import wave
        
        audio_normalized = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(audio_path, 'w') as w:
            w.setnchannels(audio.shape[0] if audio.ndim > 1 else 1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio_int16.tobytes())
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise RuntimeError(f"Failed to save audio: {e}")


class DemucsModel(Enum):
    """Available Demucs models for vocal separation"""
    HTDEMUCS = "htdemucs"  # Fast, good quality
    HTDEMUCS_FT = "htdemucs_ft"  # Fine-tuned, best quality (recommended)
    HTDEMUCS_6S = "htdemucs_6s"  # 6-stem separation (vocals, drums, bass, other, guitar, piano)
    HDEMUCS_MMI = "hdemucs_mmi"  # Hybrid, experimental


@dataclass
class SeparationResult:
    """Result of vocal separation"""
    vocals_path: str
    instrumental_path: str
    other_stems: Dict[str, str]  # Additional stems if using 6s model
    processing_time: float
    model_used: str
    device_used: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'vocals_path': self.vocals_path,
            'instrumental_path': self.instrumental_path,
            'other_stems': self.other_stems,
            'processing_time': self.processing_time,
            'model_used': self.model_used,
            'device_used': self.device_used
        }


class VocalSeparator:
    """
    UVR5 Vocal Separator with CPU Fallback
    
    Separates vocals from background music and sound effects for video translation.
    Essential for "Magic Mode" dubbing where original background music is preserved.
    
    Features:
    - Uses Demucs on GPU for best quality
    - Falls back to UVR5-ONNX (audio-separator) on CPU for memory efficiency
    - Multiple Demucs model support (htdemucs, htdemucs_ft, htdemucs_6s)
    - Automatic CUDA/CPU device detection
    - CJK language audio support
    - High-quality vocal extraction
    - Background music preservation
    """
    
    def __init__(
        self,
        model: DemucsModel = DemucsModel.HTDEMUCS_FT,
        device: Optional[str] = None,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        force_cpu_fallback: bool = False
    ):
        """
        Initialize VocalSeparator
        
        Args:
            model: Demucs model to use (default: htdemucs_ft for best quality)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            shifts: Number of random shifts for better quality (higher = slower but better)
            overlap: Overlap between segments (0.25 = 25%)
            split: Split audio into chunks for processing (recommended for long audio)
            segment: Segment size in seconds (None = auto)
            force_cpu_fallback: Force use of UVR5-ONNX instead of Demucs (useful for CPU)
        """
        self.model_name = model.value
        self.shifts = shifts
        self.overlap = overlap
        self.split = split
        self.segment = segment
        self.force_cpu_fallback = force_cpu_fallback
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"VocalSeparator initialized with model={self.model_name}, device={self.device}")
        
        self.model = None
        self._model_loaded = False
        self._uvr_separator = None
        self._using_fallback = False
        self._uvr5_separator = None
    
    def load_model(self):
        """Load Demucs model"""
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from demucs.pretrained import get_model
            
            logger.info(f"Loading Demucs model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self._model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {e}")
            raise RuntimeError(f"Failed to load Demucs model: {e}")
    
    def separate(
        self,
        audio_path: str,
        output_dir: str,
        output_format: str = "wav"
    ) -> SeparationResult:
        """
        Separate vocals from background music
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated stems
            output_format: Output audio format (wav, mp3, flac)
        
        Returns:
            SeparationResult with paths to separated audio files
        """
        import time
        from demucs.apply import apply_model
        
        start_time = time.time()
        
        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Separating audio: {audio_path}")
        
        try:
            # Load audio using soundfile to avoid torchcodec issues
            wav_np, sr = load_audio(audio_path)
            wav_tensor = torch.from_numpy(wav_np).float()
            
            # Resample if needed (Demucs expects 44100 Hz)
            if sr != self.model.samplerate:
                logger.info(f"Resampling from {sr} Hz to {self.model.samplerate} Hz")
                resampler = torchaudio.transforms.Resample(sr, self.model.samplerate)
                wav_tensor = resampler(wav_tensor)
                sr = self.model.samplerate
            
            # Move to device
            wav_tensor = wav_tensor.to(self.device)
            
            # Ensure correct shape for Demucs: (batch, channels, samples)
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)
            elif wav_tensor.dim() == 2:
                wav_tensor = wav_tensor.unsqueeze(0)
            
            # Apply model
            logger.info("Running vocal separation...")
            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    wav_tensor,
                    shifts=self.shifts,
                    split=self.split,
                    overlap=self.overlap,
                    segment=self.segment,
                    device=self.device
                )
            
            # Get source names from model
            source_names = self.model.sources
            
            # Save separated stems
            output_paths = {}
            base_name = Path(audio_path).stem
            
            for i, source_name in enumerate(source_names):
                output_path = os.path.join(
                    output_dir,
                    f"{base_name}_{source_name}.{output_format}"
                )
                
                # Get source audio (remove batch dimension)
                source_audio = sources[0, i].cpu().numpy()
                
                # Save audio using soundfile
                save_audio(
                    output_path,
                    source_audio,
                    self.model.samplerate
                )
                
                output_paths[source_name] = output_path
                logger.info(f"Saved {source_name}: {output_path}")
            
            # Determine vocals and instrumental paths
            vocals_path = output_paths.get('vocals', '')
            
            # For instrumental, combine all non-vocal stems
            if 'no_vocals' in output_paths:
                instrumental_path = output_paths['no_vocals']
            else:
                # For models that separate into multiple stems, we need to mix non-vocal stems
                instrumental_path = self._create_instrumental_mix(
                    output_paths,
                    output_dir,
                    base_name,
                    output_format
                )
            
            # Get other stems (excluding vocals and instrumental)
            other_stems = {
                k: v for k, v in output_paths.items()
                if k not in ['vocals', 'no_vocals']
            }
            
            processing_time = time.time() - start_time
            
            result = SeparationResult(
                vocals_path=vocals_path,
                instrumental_path=instrumental_path,
                other_stems=other_stems,
                processing_time=processing_time,
                model_used=self.model_name,
                device_used=self.device
            )
            
            logger.info(f"Separation completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {e}")
            raise RuntimeError(f"Vocal separation failed: {e}")
    
    def _create_instrumental_mix(
        self,
        output_paths: Dict[str, str],
        output_dir: str,
        base_name: str,
        output_format: str
    ) -> str:
        """
        Create instrumental mix by combining all non-vocal stems
        
        Args:
            output_paths: Dictionary of stem paths
            output_dir: Output directory
            base_name: Base filename
            output_format: Output format
        
        Returns:
            Path to instrumental mix
        """
        try:
            # Get all non-vocal stems
            non_vocal_stems = [
                path for name, path in output_paths.items()
                if name != 'vocals'
            ]
            
            if not non_vocal_stems:
                logger.warning("No non-vocal stems found")
                return ""
            
            # If only one non-vocal stem, just return it
            if len(non_vocal_stems) == 1:
                return non_vocal_stems[0]
            
            # Mix multiple stems
            logger.info("Mixing non-vocal stems to create instrumental track")
            
            mixed_audio = None
            sr: int = 44100
            
            for stem_path in non_vocal_stems:
                audio_np, sample_rate = load_audio(stem_path)
                audio_tensor = torch.from_numpy(audio_np).float()
                
                if mixed_audio is None:
                    mixed_audio = audio_tensor
                    sr = int(sample_rate)
                else:
                    # Ensure same length
                    min_len = min(mixed_audio.shape[0], audio_tensor.shape[0])
                    mixed_audio = mixed_audio[:min_len] + audio_tensor[:min_len]
            
            # Normalize to prevent clipping
            if mixed_audio is not None:
                mixed_audio = mixed_audio / len(non_vocal_stems)
            
            # Save mixed instrumental
            instrumental_path = os.path.join(
                output_dir,
                f"{base_name}_instrumental.{output_format}"
            )
            
            # Convert to numpy and save
            if mixed_audio is not None:
                mixed_audio_np = mixed_audio.cpu().numpy()
                save_audio(instrumental_path, mixed_audio_np, sr if sr is not None else 44100)
                logger.info(f"Created instrumental mix: {instrumental_path}")
                return instrumental_path
            else:
                return non_vocal_stems[0] if non_vocal_stems else ""
            
        except Exception as e:
            logger.error(f"Failed to create instrumental mix: {e}")
            if non_vocal_stems:
                return non_vocal_stems[0]
            return ""
    
    def is_available(self) -> bool:
        """Check if Demucs is available"""
        try:
            import demucs
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available Demucs models"""
        return [model.value for model in DemucsModel]
    
    @staticmethod
    def create(force_cpu_fallback: bool = False) -> 'VocalSeparator':
        """
        Factory method to create VocalSeparator with automatic backend selection
        
        Args:
            force_cpu_fallback: If True, use UVR5-ONNX even if GPU is available
        
        Returns:
            VocalSeparator instance with appropriate backend
        """
        if force_cpu_fallback:
            return VocalSeparator(force_cpu_fallback=True)
        
        if torch.cuda.is_available():
            return VocalSeparator(device="cuda")
        else:
            return VocalSeparator(force_cpu_fallback=True)
    
    def _load_uvr5_fallback(self) -> bool:
        """Load UVR5-ONNX fallback for CPU processing"""
        try:
            from modules.uvr5_separator import UVRSeparator, UVR5Model
            
            logger.info("Loading UVR5-ONNX fallback (audio-separator)")
            self._uvr5_separator = UVRSeparator(
                model=UVR5Model.HP_UVR,
                device="cpu"
            )
            self._uvr5_separator.load_model()
            self._using_fallback = True
            logger.info("UVR5-ONNX fallback loaded successfully")
            return True
        except ImportError as e:
            logger.error(f"UVR5 separator not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load UVR5 fallback: {e}")
            return False
    
    def separate_with_fallback(
        self,
        audio_path: str,
        output_dir: str,
        output_format: str = "wav"
    ) -> SeparationResult:
        """
        Separate vocals with automatic fallback from Demucs to UVR5-ONNX
        
        This method attempts to use Demucs first (if GPU available), then falls back
        to UVR5-ONNX for CPU-friendly processing.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated stems
            output_format: Output audio format (wav, mp3, flac)
        
        Returns:
            SeparationResult with paths to separated audio files
        """
        import time
        
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we should use Demucs (GPU available and not forced to fallback)
        use_demucs = (self.device == "cuda" and torch.cuda.is_available()) and not self.force_cpu_fallback
        
        if use_demucs:
            try:
                logger.info("Attempting Demucs separation (GPU)...")
                result = self.separate(audio_path, output_dir, output_format)
                result.processing_time = time.time() - start_time
                logger.info(f"Demucs separation completed in {result.processing_time:.2f}s")
                return result
            except RuntimeError as e:
                if "not enough memory" in str(e).lower() or " DefaultCPUAllocator" in str(e):
                    logger.warning("Demucs ran out of memory, falling back to UVR5-ONNX...")
                else:
                    logger.warning(f"Demucs failed: {e}, falling back to UVR5-ONNX...")
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Demucs separation failed: {e}, falling back to UVR5-ONNX...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Use UVR5-ONNX fallback
        logger.info("Using UVR5-ONNX (audio-separator) for CPU processing...")
        
        if not self._uvr5_separator:
            if not self._load_uvr5_fallback():
                raise RuntimeError("Failed to load UVR5 fallback and Demucs failed")
        
        try:
            uvr5_result = self._uvr5_separator.separate(audio_path, output_dir, output_format)
            
            # Create compatible SeparationResult
            result = SeparationResult(
                vocals_path=uvr5_result.vocals_path,
                instrumental_path=uvr5_result.instrumental_path,
                other_stems=uvr5_result.other_stems,
                processing_time=time.time() - start_time,
                model_used=self._uvr5_separator.model_name,
                device_used="cpu"
            )
            
            logger.info(f"UVR5 separation completed in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"UVR5 separation failed: {e}")
            raise RuntimeError(f"Vocal separation failed: {e}")
    
    def __repr__(self) -> str:
        backend = "UVR5-ONNX" if self._using_fallback else "Demucs"
        return f"VocalSeparator(model={self.model_name}, device={self.device}, backend={backend})"
