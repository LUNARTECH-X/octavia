"""
UVR5 Vocal Separator Module using ONNX-based models (audio-separator)
CPU-friendly vocal separation for "Magic Mode" dubbing
Cross-platform compatible with Windows/Linux
"""

import os
import sys
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import torch

logger = logging.getLogger(__name__)


def get_crossplatform_cache_dir() -> str:
    """Get cross-platform cache directory for models"""
    if sys.platform == "win32":
        cache_dir = os.path.join(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()), 'audio-separator-models')
    else:
        cache_dir = tempfile.gettempdir()
    
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


class UVR5Model(Enum):
    """Available UVR5 models for vocal separation"""
    HP_UVR = "1_HP-UVR.pth"  # Basic UVR model
    HP_VOCAL_UVR = "3_HP-Vocal-UVR.pth"  # Vocal extraction
    HP2_UVR = "7_HP2-UVR.pth"  # HP2 UVR model
    DEMUCS_4_STEM = "htdemucs_4.yaml"  # 4-stem Demucs
    DEMUCS_6_STEM = "htdemucs_6s.yaml"  # 6-stem Demucs


@dataclass
class SeparationResult:
    """Result of vocal separation"""
    vocals_path: str
    instrumental_path: str
    other_stems: Dict[str, str]
    processing_time: float
    model_used: str
    device_used: str
    
    def to_dict(self) -> dict:
        return {
            'vocals_path': self.vocals_path,
            'instrumental_path': self.instrumental_path,
            'other_stems': self.other_stems,
            'processing_time': self.processing_time,
            'model_used': self.model_used,
            'device_used': self.device_used
        }


class UVRSeparator:
    """
    UVR5 Vocal Separator using ONNX-based models
    
    CPU-friendly vocal separation for video translation.
    Essential for "Magic Mode" dubbing where original background music is preserved.
    
    Features:
    - Multiple model support (VR, MDX, Demucs)
    - Automatic CPU/GPU detection
    - Low memory footprint (works on CPU)
    - Fast inference with ONNX runtime
    - CJK language audio support
    """
    
    def __init__(
        self,
        model: UVR5Model = UVR5Model.HP_UVR,
        device: Optional[str] = None,
        output_format: str = "wav"
    ):
        """
        Initialize UVRSeparator
        
        Args:
            model: UVR5 model to use (default: HP-UVR)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            output_format: Output audio format (wav, flac, mp3)
        """
        self.model_name = model.value
        self.output_format = output_format
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"UVRSeparator initialized with model={self.model_name}, device={self.device}")
        
        self.separator = None
        self._model_loaded = False
    
    def load_model(self):
        """Load UVR5 model with cross-platform cache directory"""
        if self._model_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            from audio_separator.separator import Separator
            
            logger.info(f"Loading UVR5 model: {self.model_name}")
            
            # Set up cross-platform model cache directory
            cache_dir = get_crossplatform_cache_dir()
            os.environ['AUDIO_SEPARATOR_MODEL_DIR'] = cache_dir
            logger.info(f"Using model cache directory: {cache_dir}")
            
            self.separator = Separator(
                output_format=self.output_format.upper()
            )
            
            # Load the model (audio-separator handles cross-platform paths internally)
            self.separator.load_model(self.model_name)
            
            self._model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model: {self.separator.model_friendly_name}")
            
        except Exception as e:
            logger.error(f"Failed to load UVR5 model: {e}")
            raise RuntimeError(f"Failed to load UVR5 model: {e}")
    
    def separate(
        self,
        audio_path: str,
        output_dir: str,
        output_format: Optional[str] = None
    ) -> SeparationResult:
        """
        Separate vocals from background music
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save separated stems
            output_format: Override output format
        
        Returns:
            SeparationResult with paths to separated audio files
        """
        import time
        
        start_time = time.time()
        
        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Separating audio: {audio_path}")
        
        try:
            # Run separation
            logger.info("Running vocal separation...")
            
            # audio-separator saves files in the CWD by default.
            # We want them in output_dir.
            results = self.separator.separate(audio_path)
            
            processing_time = time.time() - start_time
            
            # Parse output files and move them to output_dir
            vocals_path = ""
            instrumental_path = ""
            other_stems = {}
            
            base_name = Path(audio_path).stem
            
            for output_file in results:
                # Move file to output_dir if it's not already there
                filename = os.path.basename(output_file)
                final_path = os.path.join(output_dir, filename)
                
                if os.path.abspath(output_file) != os.path.abspath(final_path):
                    if os.path.exists(final_path):
                        os.remove(final_path)
                    os.rename(output_file, final_path)
                
                stem_lower = Path(final_path).stem.lower()
                
                if 'vocals' in stem_lower:
                    vocals_path = final_path
                elif 'instrumental' in stem_lower or 'no_vocals' in stem_lower or 'accompaniment' in stem_lower or 'backing' in stem_lower:
                    instrumental_path = final_path
                elif 'drums' in stem_lower:
                    other_stems['drums'] = final_path
                elif 'bass' in stem_lower:
                    other_stems['bass'] = final_path
                elif 'other' in stem_lower:
                    other_stems['other'] = final_path
                else:
                    stem_name = Path(final_path).stem.replace(f"{base_name}_", "").replace(f"_{base_name}", "")
                    if stem_name and 'vocals' not in stem_name and 'instrumental' not in stem_name:
                        other_stems[stem_name] = final_path
            
            # If instrumental not found, look for remaining file
            if not instrumental_path and len(results) > 1:
                for f in results:
                    # Need to use the moved path
                    moved_path = os.path.join(output_dir, os.path.basename(f))
                    if moved_path != vocals_path:
                        instrumental_path = moved_path
                        break
            
            logger.info(f"Separation completed in {processing_time:.2f}s")
            
            result = SeparationResult(
                vocals_path=vocals_path,
                instrumental_path=instrumental_path,
                other_stems=other_stems,
                processing_time=processing_time,
                model_used=self.model_name,
                device_used=self.device
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Vocal separation failed: {e}")
            raise RuntimeError(f"Vocal separation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if UVR5 is available"""
        try:
            from audio_separator.separator import Separator
            return True
        except ImportError:
            return False
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available UVR5 models"""
        return [model.value for model in UVR5Model]
    
    @staticmethod
    def get_all_supported_models() -> Dict[str, str]:
        """Get all supported models with descriptions"""
        return {
            "1_HP-UVR.pth": "VR Arch Single: Basic instrumental/vocal separation (CPU-friendly)",
            "3_HP-Vocal-UVR.pth": "VR Arch Single: Vocal extraction focused (CPU-friendly)",
            "7_HP2-UVR.pth": "VR Arch Single: HP2 model, improved quality (CPU-friendly)",
            "htdemucs_4.yaml": "Demucs 4-stem: vocals, drums, bass, other (requires GPU)",
            "htdemucs_6s.yaml": "Demucs 6-stem: 6 separate stems (requires GPU)",
        }
    
    def __repr__(self) -> str:
        return f"UVRSeparator(model={self.model_name}, device={self.device})"


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file using soundfile"""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise RuntimeError(f"Failed to load audio: {e}")


def save_audio(audio_path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio file using wave module"""
    try:
        import wave
        
        audio_normalized = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        with wave.open(audio_path, 'w') as w:
            channels = audio.shape[0] if audio.ndim > 1 else 1
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio_int16.tobytes())
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        raise RuntimeError(f"Failed to save audio: {e}")
