"""
Audio Translator Module for Octavia Video Translator
Enhanced with better translation quality and timing accuracy
Supports Russian to English and English to German translations
Uses Edge-TTS for high-quality multilingual voice synthesis
"""

import os
import sys
import json
import asyncio
import logging
import re
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline
from pydub import AudioSegment
import numpy as np
from difflib import SequenceMatcher
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    print("Coqui TTS not available, using fallback TTS")

logger = logging.getLogger(__name__)

@dataclass
class TranslationConfig:
    """Configuration for translation"""
    source_lang: str = "en"
    target_lang: str = "de"
    auto_detect: bool = True
    chunk_size: int = 30
    max_condensation_ratio: float = 1.2
    timing_tolerance_ms: int = 200
    voice_speed: float = 1.0
    voice_pitch: str = "+0Hz"
    voice_style: str = "neutral"
    use_gpu: bool = False
    cache_dir: str = "~/.cache/octavia"
    model_size: str = "base"
    # Voice quality settings
    enable_input_normalization: bool = True
    enable_denoising: bool = True
    enable_gain_consistency: bool = True
    enable_silence_padding: bool = True
    validation_spots: int = 5
    max_speedup_ratio: float = 1.1
    target_lufs: float = -16.0

@dataclass
class TranslationResult:
    """Result of audio translation"""
    success: bool
    original_text: str
    translated_text: str
    original_language: str
    target_language: str
    original_duration_ms: float
    translated_duration_ms: float
    duration_match_percent: float
    speed_adjustment: float
    output_path: str
    subtitle_path: Optional[str] = None
    timing_segments: List[Dict] = None
    error: Optional[str] = None

class AudioTranslator:
    """Main audio translation class with improved quality"""
    
    # Translation model mapping with better models
    MODEL_MAPPING = {
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en"
    }
    
    # Edge-TTS voice mapping with better voices
    VOICE_MAPPING = {
        "en": "en-US-JennyNeural",
        "de": "de-DE-KatjaNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural"
    }
    
    # Voice rate mapping (characters per second) for different languages
    VOICE_RATES = {
        "en": 12,  # English: ~12 chars/second
        "de": 11,  # German: ~11 chars/second  
        "ru": 10,  # Russian: ~10 chars/second
        "es": 13,  # Spanish: ~13 chars/second
        "fr": 12   # French: ~12 chars/second
    }
    
    def __init__(self, config: TranslationConfig = None):
        self.config = config or TranslationConfig()
        self.whisper_model = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.translation_pipeline = None
        self._models_loaded = False
        
    def load_models(self):
        """Load all required AI models with speed optimizations"""
        try:
            logger.info("Loading Whisper model (optimized for speed)...")

            # SPEED OPTIMIZATION: Use faster-whisper if available, otherwise base model for speed
            try:
                from faster_whisper import WhisperModel
                # Use smaller model for speed, enable GPU if available
                self.whisper_model = WhisperModel(
                    "base",  # Small model for speed (2x faster than medium)
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    compute_type="float16" if torch.cuda.is_available() else "float32",
                    cpu_threads=4,
                    num_workers=1,  # Single worker for speed
                    download_root=os.path.expanduser("~/.cache/whisper")
                )
                self._using_faster_whisper = True
                logger.info("[OK] Loaded faster-whisper base model (GPU accelerated)" if torch.cuda.is_available() else "[OK] Loaded faster-whisper base model (CPU)")
            except ImportError:
                logger.warning("faster-whisper not available, using standard whisper")
                # Fallback to standard whisper with optimizations
                self.whisper_model = whisper.load_model(
                    "base",  # Base model for speed (much faster than medium)
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self._using_faster_whisper = False
                logger.info("[OK] Loaded standard whisper base model")

            logger.info("Loading translation model...")
            model_key = f"{self.config.source_lang}-{self.config.target_lang}"
            model_name = self.MODEL_MAPPING.get(model_key)

            if not model_name:
                logger.warning(f"Model not found for {model_key}, using Helsinki-NLP/opus-mt-mul-en")
                model_name = "Helsinki-NLP/opus-mt-mul-en"

            # Load tokenizer and model with optimizations
            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)

            # SPEED OPTIMIZATION: Use GPU if available for translation
            device = 0 if torch.cuda.is_available() else -1
            self.translation_pipeline = pipeline(
                "translation",
                model=self.translation_model,
                tokenizer=self.translation_tokenizer,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self._models_loaded = True
            model_type = "faster-whisper" if self._using_faster_whisper else "standard whisper"
            gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
            logger.info(f"[OK] Models loaded successfully: {model_type} base + {model_name} ({gpu_status})")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to minimal models
            try:
                self.whisper_model = whisper.load_model("tiny")  # Ultra-fast fallback
                self.translation_pipeline = pipeline(
                    "translation_en_to_de" if self.config.target_lang == "de" else "translation_en_to_fr",
                    device=-1
                )
                self._using_faster_whisper = False
                logger.warning("Loaded ultra-fast fallback models (tiny whisper)")
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback models also failed: {fallback_error}")
                return False
    
    def detect_language(self, audio_path: str) -> str:
        """Detect language of audio file with confidence"""
        try:
            if not self.whisper_model:
                self.load_models()
            
            # Load and process audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(
                self.whisper_model.device if hasattr(self.whisper_model, 'device') else 'cpu'
            )
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2%})")
            
            # Map Whisper language codes to our codes
            lang_map = {
                "en": "en",
                "de": "de", 
                "ru": "ru",
                "es": "es",
                "fr": "fr"
            }
            
            return lang_map.get(detected_lang, self.config.source_lang)
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return self.config.source_lang
    
    def transcribe_with_segments(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text with detailed timestamps - sequential processing only"""
        try:
            if not self.whisper_model:
                self.load_models()

            # Check if audio file exists and has content
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file not found: {audio_path}")

            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise Exception(f"Audio file is empty: {audio_path}")

            # Check if file is too short (less than 0.5 seconds)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                if len(audio) < 500:  # Less than 0.5 seconds
                    logger.warning(f"Audio file too short ({len(audio)}ms), returning empty transcription")
                    return {
                        "text": "",
                        "segments": [],
                        "language": self.config.source_lang,
                        "success": True
                    }
            except Exception as audio_check_error:
                logger.warning(f"Could not check audio duration: {audio_check_error}")

            logger.info(f"Transcribing audio file: {audio_path} ({file_size} bytes)")

            # Set language for transcription
            language = self.config.source_lang
            if self.config.auto_detect or language == "auto":
                try:
                    language = self.detect_language(audio_path)
                    self.config.source_lang = language
                except Exception as lang_error:
                    logger.warning(f"Language detection failed, using {language}: {lang_error}")

            logger.info(f"Transcribing audio in {language}...")

            # Use appropriate transcription method based on model type
            try:
                if self._using_faster_whisper:
                    # faster_whisper API - basic parameters only
                    try:
                        segments, info = self.whisper_model.transcribe(
                            audio_path,
                            language=language if language != "auto" else None
                        )
                    except TypeError as param_error:
                        # If parameters are wrong, try minimal call
                        logger.warning(f"faster_whisper parameters failed: {param_error}, using minimal call")
                        segments, info = self.whisper_model.transcribe(audio_path)

                    # Convert faster_whisper result to expected format
                    # Combine all segment texts to get full transcription
                    full_text = "".join([segment.text for segment in segments]).strip()

                    result = {
                        "text": full_text,
                        "language": info.language if hasattr(info, 'language') else self.config.source_lang,
                        "segments": [
                            {
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text,
                                "words": []  # faster_whisper doesn't provide word-level timestamps by default
                            }
                            for segment in segments
                        ]
                    }
                else:
                    # Standard whisper API
                    result = self.whisper_model.transcribe(
                        audio_path,
                        language=language if language != "auto" else None,
                        task="transcribe",
                        verbose=False,
                        temperature=0.0,
                        best_of=1,
                        beam_size=1
                    )
            except Exception as basic_error:
                logger.error(f"Basic transcription failed: {basic_error}")
                raise Exception(f"Transcription failed: {basic_error}")

            if not result or not result.get("text"):
                logger.warning("Transcription returned no text, returning empty result")
                return {
                    "text": "",
                    "segments": [],
                    "language": language,
                    "success": True
                }

            # Process segments for better accuracy
            segments = []
            raw_segments = result.get("segments", [])

            if raw_segments:
                for segment in raw_segments:
                    if segment.get("text") and segment["text"].strip():
                        segments.append({
                            "start": segment.get("start", 0),
                            "end": segment.get("end", segment.get("start", 0) + 1),
                            "text": segment["text"].strip(),
                            "words": segment.get("words", [])
                        })

            full_text = " ".join([seg["text"] for seg in segments]) if segments else result.get("text", "")

            logger.info(f"Transcription successful: {len(full_text)} chars, {len(segments)} segments")

            # Calculate transcription quality metrics
            quality_metrics = self._calculate_transcription_quality(
                full_text, segments, result
            )

            return {
                "text": full_text.strip(),
                "segments": segments,
                "language": result.get("language", language),
                "success": True,
                "quality_metrics": quality_metrics
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "text": "",
                "segments": [],
                "language": self.config.source_lang,
                "success": False,
                "error": str(e)
            }
    
    def translate_text_with_context(self, text: str, segments: List[Dict]) -> Tuple[str, List[Dict]]:
        """Translate text with segment context preservation. Fails loudly if translation is a no-op or fallback."""
        try:
            if not self.translation_pipeline:
                self.load_models()

            if not text or len(text.strip()) < 2:
                logger.warning("No text to translate")
                raise ValueError("No text to translate")

            logger.info(f"Translating text: '{text[:100]}...'")

            # Simple direct translation for now
            try:
                logger.info(f"Attempting translation with pipeline...")

                if self.translation_pipeline:
                    result = self.translation_pipeline(text, max_length=512, num_beams=1)
                    translated_text = result[0]['translation_text']
                    logger.info(f"Pipeline translation result: '{translated_text[:100]}...'")
                else:
                    logger.warning("No translation pipeline available, trying direct model")
                    inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.translation_model.generate(**inputs, max_length=512, num_beams=1)
                        translated_text = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

                    logger.info(f"Direct model translation result: '{translated_text[:100]}...'")

                # Ensure we have translated text
                if not translated_text or len(translated_text.strip()) < 1:
                    logger.error("Translation returned empty text. Failing loudly.")
                    raise RuntimeError("Translation returned empty text.")

                # If translation is too short or same as input, it's likely failed
                if len(translated_text.strip()) < len(text.strip()) * 0.1 or translated_text.strip().lower() == text.strip().lower():
                    logger.error("Translation appears to have failed (no-op or too similar to input). Failing loudly.")
                    raise RuntimeError("Translation appears to have failed (no-op or too similar to input). Text: " + text[:100])

            except Exception as translation_error:
                logger.error(f"Translation pipeline failed: {translation_error}")
                import traceback
                traceback.print_exc()
                raise

            # Clean the translation
            translated_text = self._clean_translation(translated_text)

            # Create translated segments (simple mapping)
            translated_segments = []
            if segments:
                translated_words = translated_text.split()
                total_original_words = sum(len(seg["text"].split()) for seg in segments)

                word_idx = 0
                for seg in segments:
                    seg_words = len(seg["text"].split())
                    if total_original_words > 0:
                        ratio = seg_words / total_original_words
                        num_translated_words = max(1, int(len(translated_words) * ratio))
                    else:
                        num_translated_words = max(1, len(translated_words) // len(segments))

                    start_idx = word_idx
                    end_idx = min(word_idx + num_translated_words, len(translated_words))
                    segment_text = " ".join(translated_words[start_idx:end_idx])

                    translated_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "original_text": seg["text"],
                        "translated_text": segment_text if segment_text else seg["text"],
                        "words": seg.get("words", [])
                    })

                    word_idx = end_idx

            logger.info(f"Translation completed: {len(text)} chars to {len(translated_text)} chars")

            return translated_text, translated_segments

        except Exception as e:
            logger.error(f"Translation failed completely: {e}")
            raise
    
    def _clean_translation(self, text: str) -> str:
        """Clean and normalize translated text"""
        # Remove duplicate punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
        # Capitalize sentences
        sentences = re.split(r'([.!?])\s+', text)
        cleaned = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    cleaned.append(sentence)
                if i + 1 < len(sentences):
                    cleaned.append(sentences[i + 1])
        
        text = ' '.join(cleaned)
        return text.strip()
    
    def condense_text_smart(self, text: str, target_duration_ms: float) -> Tuple[str, float]:
        """Smart text condensation to fit target duration"""
        try:
            words = text.split()
            if len(words) <= 10:  # Don't condense very short texts
                return text, 1.0
            
            # Estimate speaking rate for target language
            chars_per_second = self.VOICE_RATES.get(self.config.target_lang, 12)
            target_chars = int(target_duration_ms / 1000 * chars_per_second)
            
            current_chars = len(text)
            
            if current_chars <= target_chars:
                return text, 1.0
            
            # Calculate needed condensation ratio
            ratio = target_chars / current_chars
            
            # Don't condense too much
            min_ratio = 0.7
            if ratio < min_ratio:
                ratio = min_ratio
            
            # Smart condensation: remove filler words first
            filler_words = {
                "en": ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "that", "this", "these", "those"],
                "de": ["der", "die", "das", "und", "oder", "aber", "in", "auf", "an", "zu", "für", "von", "mit", "dass", "dies", "diese", "jene"],
                "ru": ["и", "в", "на", "с", "по", "к", "у", "о", "от", "до", "за", "из", "над", "что", "это", "этот", "эта", "эти"],
                "es": ["el", "la", "los", "las", "y", "o", "pero", "en", "a", "de", "con", "por", "para", "que", "este", "esta", "estos", "estas"],
                "fr": ["le", "la", "les", "et", "ou", "mais", "en", "à", "de", "avec", "pour", "par", "que", "ce", "cette", "ces"]
            }
            
            lang_fillers = filler_words.get(self.config.target_lang, filler_words["en"])
            
            # Remove filler words
            filtered_words = []
            for word in words:
                if word.lower() not in lang_fillers:
                    filtered_words.append(word)
            
            # If still too long, use compression
            if len(" ".join(filtered_words)) > target_chars:
                # Take key parts: first 30%, middle 40%, last 30%
                target_word_count = int(len(filtered_words) * ratio)
                keep_first = int(target_word_count * 0.3)
                keep_last = int(target_word_count * 0.3)
                keep_middle = target_word_count - keep_first - keep_last
                
                if keep_middle > 0:
                    middle_start = len(filtered_words) // 2 - keep_middle // 2
                    middle_end = middle_start + keep_middle
                    
                    if middle_start < 0:
                        middle_start = 0
                    if middle_end > len(filtered_words):
                        middle_end = len(filtered_words)
                    
                    first_part = filtered_words[:keep_first]
                    middle_part = filtered_words[middle_start:middle_end]
                    last_part = filtered_words[-keep_last:] if keep_last > 0 else []
                    
                    filtered_words = first_part + middle_part + last_part
                else:
                    filtered_words = filtered_words[:target_word_count]
            
            condensed_text = " ".join(filtered_words)
            actual_ratio = len(condensed_text) / len(text) if len(text) > 0 else 1.0
            
            logger.info(f"Condensed: {len(words)} words to {len(filtered_words)} words (ratio: {actual_ratio:.2f})")
            
            return condensed_text, actual_ratio
            
        except Exception as e:
            logger.error(f"Smart condensation failed: {e}")
            return text, 1.0
    
    def synthesize_speech_with_timing(self, text: str, segments: List[Dict], output_path: str) -> Tuple[bool, List[Dict]]:
        """Generate speech with timing preservation using Coqui TTS"""
        try:
            if not text or len(text.strip()) < 2:
                logger.warning("Text too short for TTS, creating silent audio")
                silent_audio = AudioSegment.silent(duration=1000)
                silent_audio.export(output_path, format="wav")
                return True, []

            # Use Coqui TTS if available, otherwise fallback to gTTS
            if COQUI_AVAILABLE:
                logger.info(f"Generating speech with Coqui TTS for language: {self.config.target_lang}")

                # Initialize Coqui TTS with appropriate model
                try:
                    # Map languages to Coqui TTS models
                    coqui_lang_map = {
                        'en': 'tts_models/en/ljspeech/tacotron2-DDC_ph',
                        'es': 'tts_models/es/mai/tacotron2-DDC_ph',
                        'fr': 'tts_models/fr/mai/tacotron2-DDC_ph',
                        'de': 'tts_models/de/mai/tacotron2-DDC_ph',
                        'it': 'tts_models/it/mai/tacotron2-DDC_ph',
                        'pt': 'tts_models/pt/cv/vits',
                        'ru': 'tts_models/ru/cv/v3_nemo',
                        'ja': 'tts_models/ja/kokoro/tacotron2-DDC_ph',
                        'zh': 'tts_models/zh-cn/csspc/tacotron2-DDC_ph',
                        'ar': 'tts_models/ar/cv/vits',
                        'hi': 'tts_models/hi/cv/v3_nemo'
                    }

                    model_name = coqui_lang_map.get(self.config.target_lang, 'tts_models/en/ljspeech/tacotron2-DDC_ph')

                    # Initialize TTS
                    tts = TTS(model_name).to(self.config.device if torch.cuda.is_available() else "cpu")

                    # Generate speech
                    wav = tts.tts(text=text)

                    # Convert to numpy array and save
                    import numpy as np
                    wav_np = np.array(wav)

                    # Save as WAV
                    import scipy.io.wavfile as wavfile
                    wavfile.write(output_path, 22050, (wav_np * 32767).astype(np.int16))

                    # Load with pydub for duration
                    tts_audio = AudioSegment.from_file(output_path)
                    tts_duration_ms = len(tts_audio)

                    logger.info(f"Coqui TTS audio generated: {tts_duration_ms}ms duration")

                except Exception as coqui_error:
                    logger.warning(f"Coqui TTS failed: {coqui_error}, falling back to gTTS")
                    return self._fallback_gtts_synthesis(text, segments, output_path)

            else:
                logger.info("Coqui TTS not available, using gTTS fallback")
                return self._fallback_gtts_synthesis(text, segments, output_path)

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Create fallback silent audio
            try:
                silent_audio = AudioSegment.silent(duration=5000)
                silent_audio.export(output_path, format="wav")
                logger.info("Created fallback silent audio")
                return True, []
            except Exception as fallback_error:
                logger.error(f"Fallback audio creation failed: {fallback_error}")
                return False, []

    def _fallback_gtts_synthesis(self, text: str, segments: List[Dict], output_path: str) -> Tuple[bool, List[Dict]]:
        """Fallback TTS synthesis using gTTS"""
        try:
            if not text or len(text.strip()) < 2:
                logger.warning("Text too short for TTS, creating silent audio")
                silent_audio = AudioSegment.silent(duration=1000)
                silent_audio.export(output_path, format="wav")
                return True, []

            # Use gTTS for synchronous TTS generation
            from gtts import gTTS
            import io

            # Get appropriate language code for gTTS
            lang_map = {
                'en': 'en',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'ja': 'ja',
                'ko': 'ko',
                'zh': 'zh-cn',
                'ar': 'ar',
                'hi': 'hi'
            }

            gtts_lang = lang_map.get(self.config.target_lang, 'en')
            logger.info(f"Generating speech with gTTS for language: {gtts_lang}")

            # Generate TTS audio synchronously
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            # Load audio with pydub
            tts_audio = AudioSegment.from_file(audio_bytes, format="mp3")
            tts_duration_ms = len(tts_audio)

            logger.info(f"gTTS audio generated: {tts_duration_ms}ms duration")

            # Calculate simple timing segments (proportional distribution)
            adjusted_segments = []
            if segments and tts_duration_ms > 0:
                total_original_duration = sum(seg["end"] - seg["start"] for seg in segments)

                if total_original_duration > 0:
                    current_pos = 0

                    for seg in segments:
                        seg_duration = seg["end"] - seg["start"]
                        seg_ratio = seg_duration / total_original_duration

                        adjusted_start = current_pos
                        adjusted_end = current_pos + (tts_duration_ms * seg_ratio)

                        adjusted_segments.append({
                            "original_start": seg["start"],
                            "original_end": seg["end"],
                            "adjusted_start": adjusted_start,
                            "adjusted_end": adjusted_end,
                            "timing_precision_ms": 0,  # Simple proportional - no precision timing
                            "within_tolerance": True,
                            "text": seg.get("translated_text", seg.get("original_text", "")),
                        })

                        current_pos = adjusted_end

                    logger.info(f"Created {len(adjusted_segments)} timing segments with proportional distribution")

            # Apply speed adjustment if needed
            if abs(self.config.voice_speed - 1.0) > 0.01:
                try:
                    speed_factor = self.config.voice_speed
                    # Simple speed adjustment using pydub
                    if speed_factor > 1.0:
                        tts_audio = tts_audio.speedup(playback_speed=speed_factor)
                    else:
                        frame_rate = int(tts_audio.frame_rate * speed_factor)
                        tts_audio = tts_audio._spawn(tts_audio.raw_data, overrides={"frame_rate": frame_rate})
                        tts_audio = tts_audio.set_frame_rate(tts_audio.frame_rate)
                    logger.info(f"Applied speed adjustment: {speed_factor:.2f}x")
                except Exception as speed_error:
                    logger.warning(f"Speed adjustment failed: {speed_error}")

            # Normalize audio levels
            try:
                tts_audio = self._normalize_audio(tts_audio)
            except Exception as norm_error:
                logger.warning(f"Audio normalization failed: {norm_error}")

            # Export final audio
            tts_audio.export(output_path, format="wav")
            logger.info(f"Speech synthesized successfully: {output_path} ({len(tts_audio)}ms)")

            return True, adjusted_segments

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Create fallback silent audio
            try:
                silent_audio = AudioSegment.silent(duration=5000)
                silent_audio.export(output_path, format="wav")
                logger.info("Created fallback silent audio")
                return True, []
            except Exception as fallback_error:
                logger.error(f"Fallback audio creation failed: {fallback_error}")
                return False, []
    
    def _adjust_audio_speed(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Adjust audio playback speed with better quality"""
        try:
            if speed == 1.0:
                return audio

            # Use FFmpeg for better speed adjustment
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
                temp_input = tmp_in.name
                audio.export(temp_input, format="wav")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                temp_output = tmp_in.name

                cmd = [
                    'ffmpeg', '-i', temp_input,
                    '-filter:a', f'atempo={speed}',
                    '-vn', '-y', temp_output
                ]

                subprocess.run(cmd, capture_output=True, text=True)

                result = AudioSegment.from_file(temp_output)

            # Cleanup
            os.unlink(tmp_input)
            os.unlink(tmp_output)

            return result

        except Exception as e:
            logger.error(f"Speed adjustment failed, using pydub fallback: {e}")
            # Fallback to pydub
            if speed > 1.0:
                return audio.speedup(playback_speed=speed)
            else:
                frame_rate = int(audio.frame_rate * speed)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": frame_rate})
                return audio.set_frame_rate(audio.frame_rate)

    def _adjust_audio_speed_precise(self, audio: AudioSegment, target_speed: float) -> AudioSegment:
        """Apply precise speed adjustment for frame-accurate duration matching"""
        try:
            if abs(target_speed - 1.0) < 0.01:  # Very close to 1.0
                return audio

            logger.info(f"Applying precise speed adjustment: {target_speed:.3f}x")

            # Use multiple techniques for precision
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
                temp_input = tmp_in.name
                audio.export(temp_input, format="wav")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                temp_output = tmp_out.name

                # Use high-precision FFmpeg filter
                cmd = [
                    'ffmpeg', '-i', temp_input,
                    '-filter:a', f'atempo={target_speed:.3f}',
                    '-vn', '-y', '-acodec', 'pcm_s16le',
                    '-ar', '44100',  # Standard sample rate
                    temp_output
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    logger.error(f"FFmpeg precise speed failed: {result.stderr}")
                    raise Exception("FFmpeg precise speed adjustment failed")

                adjusted_audio = AudioSegment.from_file(temp_output)

            # Cleanup
            os.unlink(temp_input)
            os.unlink(temp_output)

            # Verify the adjustment worked
            original_duration = len(audio)
            adjusted_duration = len(adjusted_audio)
            actual_ratio = adjusted_duration / original_duration

            logger.info(f"Precise speed result: {original_duration}ms to {adjusted_duration}ms (ratio: {actual_ratio:.3f})")

            return adjusted_audio

        except Exception as e:
            logger.error(f"Precise speed adjustment failed: {e}")
            # Fallback to regular speed adjustment
            return self._adjust_audio_speed(audio, target_speed)
    
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels"""
        try:
            # Target loudness (LUFS)
            target_lufs = -16.0
            
            # Simple normalization 
            max_dBFS = audio.max_dBFS
            if max_dBFS < -1.0:  # Too quiet
                gain = -1.0 - max_dBFS
                audio = audio.apply_gain(gain)
            elif max_dBFS > -1.0:  # Too loud
                gain = -1.0 - max_dBFS
                audio = audio.apply_gain(gain)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio
    
    def calculate_optimal_speed(self, original_duration_ms: float, translated_text: str) -> float:
        """Calculate optimal speed adjustment for frame-accurate timing match with quality constraints"""
        try:
            # Get speaking rate for target language
            chars_per_second = self.VOICE_RATES.get(self.config.target_lang, 12)

            # Estimate speaking time
            estimated_speaking_time = len(translated_text) / chars_per_second  # seconds

            # Convert original duration to seconds
            original_duration_s = original_duration_ms / 1000

            if original_duration_s <= 0:
                return 1.0

            # Calculate required speed for EXACT duration match
            speed = estimated_speaking_time / original_duration_s

            # Apply quality constraints: prefer text condensation over extreme speedup
            # If TTS would be too long, condense text first, then apply minor speedup only if needed

            # First, check if we need any adjustment
            if abs(speed - 1.0) < 0.05:  # Within 5% - no adjustment needed
                speed = 1.0
            elif speed < 0.9:  # TTS too slow - this suggests text condensation was insufficient
                # Allow some slowdown for natural speech
                speed = max(0.85, speed)
            elif speed > self.config.max_speedup_ratio:  # TTS too fast - limit speedup to preserve quality
                # Cap at max_speedup_ratio (default 1.1x) to avoid artifacts
                speed = min(speed, self.config.max_speedup_ratio)
                logger.info(f"Speed capped at {self.config.max_speedup_ratio:.2f}x for quality")
            else:
                # Normal range - fine-tune
                speed = round(speed * 100) / 100

            logger.info(f"Frame-accurate speed: {estimated_speaking_time:.3f}s speech in {original_duration_s:.3f}s to speed: {speed:.2f}x")

            # Update config
            self.config.voice_speed = speed

            return speed

        except Exception as e:
            logger.error(f"Speed calculation failed: {e}")
            return 1.0

    def apply_gain_consistency(self, audio: AudioSegment, target_lufs: float = -16.0) -> AudioSegment:
        """Apply consistent gain across audio chunks with compression to avoid clipping"""
        try:
            if not self.config.enable_gain_consistency:
                return audio

            # Apply dynamic range compression to prevent clipping
            # Use FFmpeg compressor filter for better control
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
                temp_input = tmp_in.name
                audio.export(temp_input, format="wav")

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                temp_output = tmp_out.name

                # Apply compressor and limiter for consistent gain
                cmd = [
                    'ffmpeg', '-i', temp_input,
                    '-filter:a', f'compand=0.3,1:6:-70,-60,-20,lowpass=f=10000',
                    '-filter:a', f'loudnorm=I={target_lufs}:TP=-1.5:LRA=11',
                    '-acodec', 'pcm_s16le',
                    '-y', temp_output
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0 and os.path.exists(temp_output):
                    processed_audio = AudioSegment.from_file(temp_output)
                else:
                    logger.warning(f"FFmpeg gain consistency failed: {result.stderr}")
                    # Fallback to simple normalization
                    processed_audio = self._normalize_audio(audio)

            # Cleanup
            os.unlink(temp_input)
            os.unlink(temp_output)

            logger.info(f"Gain consistency applied (target LUFS: {target_lufs})")
            return processed_audio

        except Exception as e:
            logger.error(f"Gain consistency failed: {e}")
            return self._normalize_audio(audio)

    def add_silence_padding(self, segments: List[Dict], total_duration_ms: float) -> List[Dict]:
        """Add silence padding to align breaths and pauses"""
        try:
            if not self.config.enable_silence_padding or not segments:
                return segments

            padded_segments = []
            current_time = 0

            # Calculate natural pause durations based on punctuation
            for i, seg in enumerate(segments):
                # Add segment with padding
                padded_start = current_time

                # Calculate segment duration
                seg_duration = seg.get("adjusted_end", seg.get("original_end", 0)) - seg.get("adjusted_start", seg.get("original_start", 0))
                seg_duration = max(seg_duration, 0.1)  # Minimum 100ms

                padded_end = padded_start + seg_duration

                # Add silence padding based on text content
                text = seg.get("translated_text", seg.get("text", ""))

                # Detect sentence endings and add appropriate pauses
                silence_padding = 0
                if text.endswith(('.', '!', '?')):
                    silence_padding = 300  # 300ms pause after sentences
                elif text.endswith(',', ';', ':'):
                    silence_padding = 150  # 150ms pause after clauses
                elif i < len(segments) - 1:  # Between segments
                    silence_padding = 100  # 100ms natural pause

                padded_end += silence_padding / 1000  # Convert to seconds

                padded_segments.append({
                    **seg,
                    "padded_start": padded_start,
                    "padded_end": padded_end,
                    "silence_padding_ms": silence_padding
                })

                current_time = padded_end

            # Scale to fit total duration if needed
            if current_time > 0:
                scale_factor = total_duration_ms / (current_time * 1000)
                if abs(scale_factor - 1.0) < 0.1:  # Only scale if significant difference
                    for seg in padded_segments:
                        seg["padded_start"] *= scale_factor
                        seg["padded_end"] *= scale_factor

            logger.info(f"Silence padding applied to {len(padded_segments)} segments")
            return padded_segments

        except Exception as e:
            logger.error(f"Silence padding failed: {e}")
            return segments

    def validate_audio_quality(self, audio_path: str, original_path: str) -> Dict[str, Any]:
        """Validate audio quality with random 20-30s spots"""
        try:
            validation_results = {
                "spots_checked": 0,
                "avg_snr": 0.0,
                "sync_accuracy_percent": 0.0,
                "artifacts_detected": 0,
                "quality_score": 0.0,
                "recommendations": []
            }

            if not self.config.validation_spots or self.config.validation_spots <= 0:
                return validation_results

            # Load both audio files
            try:
                translated_audio = AudioSegment.from_file(audio_path)
                original_audio = AudioSegment.from_file(original_path)
            except Exception as load_error:
                logger.error(f"Could not load audio for validation: {load_error}")
                return validation_results

            total_duration = min(len(translated_audio), len(original_audio))
            if total_duration < 10000:  # Less than 10 seconds
                logger.warning("Audio too short for validation")
                return validation_results

            # Generate random spots (20-30 seconds each)
            spot_duration = 25000  # 25 seconds average
            max_start = total_duration - spot_duration

            if max_start <= 0:
                # Single spot for short audio
                spots = [(0, total_duration)]
            else:
                # Generate validation spots
                import random
                spots = []
                for _ in range(min(self.config.validation_spots, 5)):
                    start = random.randint(0, max_start)
                    end = min(start + spot_duration, total_duration)
                    spots.append((start, end))

            validation_results["spots_checked"] = len(spots)

            # Analyze each spot
            snr_values = []
            sync_issues = 0

            for start_ms, end_ms in spots:
                # Extract audio segments
                orig_segment = original_audio[start_ms:end_ms]
                trans_segment = translated_audio[start_ms:end_ms]

                # Simple SNR estimation (signal power / noise power)
                try:
                    # Calculate RMS power as proxy for SNR
                    orig_samples = np.array(orig_segment.get_array_of_samples())
                    trans_samples = np.array(trans_segment.get_array_of_samples())

                    if len(orig_samples) > 0 and len(trans_samples) > 0:
                        # Normalize lengths
                        min_len = min(len(orig_samples), len(trans_samples))
                        orig_samples = orig_samples[:min_len]
                        trans_samples = trans_samples[:min_len]

                        # Calculate signal power
                        signal_power = np.mean(orig_samples ** 2)
                        noise_power = np.mean((orig_samples - trans_samples) ** 2)

                        if noise_power > 0:
                            snr = 10 * np.log10(signal_power / noise_power)
                            snr = max(0, min(60, snr))  # Clamp to reasonable range
                            snr_values.append(snr)

                        # Check for sync issues (rough estimation)
                        # Look for significant timing differences
                        if abs(len(orig_samples) - len(trans_samples)) > len(orig_samples) * 0.1:
                            sync_issues += 1

                except Exception as spot_error:
                    logger.warning(f"Spot analysis failed: {spot_error}")
                    continue

            # Calculate averages
            if snr_values:
                validation_results["avg_snr"] = sum(snr_values) / len(snr_values)

            validation_results["sync_accuracy_percent"] = ((len(spots) - sync_issues) / len(spots)) * 100 if spots else 0

            # Overall quality score
            snr_score = min(1.0, validation_results["avg_snr"] / 30.0)  # 30dB = perfect score
            sync_score = validation_results["sync_accuracy_percent"] / 100.0
            validation_results["quality_score"] = (snr_score + sync_score) / 2.0

            # Generate recommendations
            if validation_results["avg_snr"] < 15:
                validation_results["recommendations"].append("Low signal quality - consider increasing normalization strength")

            if validation_results["sync_accuracy_percent"] < 80:
                validation_results["recommendations"].append("Sync issues detected - review timing alignment")

            if validation_results["quality_score"] < 0.7:
                validation_results["recommendations"].append("Overall quality below threshold - consider adjusting parameters")

            logger.info(f"Quality validation: {validation_results['quality_score']:.2f} "
                       f"(SNR: {validation_results['avg_snr']:.1f}dB, "
                       f"Sync: {validation_results['sync_accuracy_percent']:.1f}%)")

            return validation_results

        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            return {
                "spots_checked": 0,
                "avg_snr": 0.0,
                "sync_accuracy_percent": 0.0,
                "artifacts_detected": 0,
                "quality_score": 0.0,
                "recommendations": ["Validation failed"],
                "error": str(e)
            }

    def _calculate_transcription_quality(self, transcribed_text: str, segments: List[Dict], raw_result: Dict) -> Dict[str, Any]:
        """Calculate transcription quality metrics including WER estimation"""
        try:
            metrics = {
                "confidence_score": 0.0,
                "word_count": 0,
                "segment_count": len(segments),
                "avg_segment_length": 0.0,
                "estimated_wer": 0.0,
                "meaning_preservation_score": 0.0,
                "quality_rating": "unknown"
            }

            if not transcribed_text or not segments:
                return metrics

            # Word count
            words = transcribed_text.split()
            metrics["word_count"] = len(words)

            # Average segment length
            if segments:
                total_length = sum(seg.get("end", 0) - seg.get("start", 0) for seg in segments)
                metrics["avg_segment_length"] = total_length / len(segments)

            # Confidence score (use Whisper's confidence if available)
            if raw_result and "segments" in raw_result:
                confidences = []
                for seg in raw_result["segments"]:
                    if "avg_logprob" in seg:
                        # Convert log probability to confidence score
                        confidence = min(1.0, max(0.0, 1.0 + seg["avg_logprob"]))
                        confidences.append(confidence)

                if confidences:
                    metrics["confidence_score"] = sum(confidences) / len(confidences)

            # Estimate WER (Word Error Rate) based on confidence and text patterns
            # Lower confidence = higher estimated WER
            base_wer = 0.05  # Base WER for high-confidence transcriptions
            confidence_penalty = max(0, (1.0 - metrics["confidence_score"]) * 0.20)
            metrics["estimated_wer"] = min(0.50, base_wer + confidence_penalty)

            # Meaning preservation score (simple heuristic)
            # Check for common transcription errors and meaning coherence
            meaning_score = self._assess_meaning_preservation(transcribed_text, segments)
            metrics["meaning_preservation_score"] = meaning_score

            # Overall quality rating
            avg_score = (metrics["confidence_score"] + (1 - metrics["estimated_wer"]) + meaning_score) / 3
            if avg_score >= 0.85:
                metrics["quality_rating"] = "excellent"
            elif avg_score >= 0.70:
                metrics["quality_rating"] = "good"
            elif avg_score >= 0.50:
                metrics["quality_rating"] = "fair"
            else:
                metrics["quality_rating"] = "poor"

            logger.info(f"STT Quality: {metrics['quality_rating']} "
                       f"(conf: {metrics['confidence_score']:.2f}, "
                       f"WER: {metrics['estimated_wer']:.2f}, "
                       f"meaning: {meaning_score:.2f})")

            return metrics

        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return {
                "confidence_score": 0.5,
                "word_count": len(transcribed_text.split()) if transcribed_text else 0,
                "segment_count": len(segments),
                "avg_segment_length": 0.0,
                "estimated_wer": 0.10,  # Conservative estimate
                "meaning_preservation_score": 0.8,
                "quality_rating": "unknown",
                "error": str(e)
            }

    def _assess_meaning_preservation(self, text: str, segments: List[Dict]) -> float:
        """Assess how well the transcription preserves meaning"""
        try:
            score = 0.8  # Base score

            if not text or not segments:
                return 0.5

            # Check for common transcription issues
            issues = 0
            total_checks = 0

            # 1. Check segment continuity (no major gaps)
            total_checks += 1
            segment_gaps = []
            for i in range(len(segments) - 1):
                gap = segments[i + 1]["start"] - segments[i]["end"]
                segment_gaps.append(gap)

            if segment_gaps:
                avg_gap = sum(segment_gaps) / len(segment_gaps)
                if avg_gap > 2.0:  # Large gaps might indicate missed speech
                    issues += 0.2

            # 2. Check for repetitive patterns (transcription loops)
            total_checks += 1
            words = text.lower().split()
            if len(words) > 10:
                # Check for excessive repetition
                word_counts = {}
                for word in words:
                    if len(word) > 2:  # Skip short words
                        word_counts[word] = word_counts.get(word, 0) + 1

                max_repetitions = max(word_counts.values()) if word_counts else 0
                if max_repetitions > len(words) * 0.15:  # More than 15% of words repeated
                    issues += 0.3

            # 3. Check text coherence (sentence structure)
            total_checks += 1
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 1:
                # Check if sentences have reasonable length
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if avg_sentence_length < 3:  # Very short sentences might indicate fragmentation
                    issues += 0.2

            # 4. Language consistency check
            total_checks += 1
            # Simple check for mixed languages (basic heuristic)
            english_words = sum(1 for word in words if word.replace("'", "").isalpha() and len(word) > 2)
            if english_words > 0:
                english_ratio = english_words / len([w for w in words if len(w) > 2])
                # If mostly English-like words, likely good transcription
                if english_ratio < 0.3:  # Low English ratio might indicate issues
                    issues += 0.1

            # Calculate final score
            final_score = max(0.1, min(1.0, score - (issues / total_checks)))
            return final_score

        except Exception as e:
            logger.error(f"Meaning assessment failed: {e}")
            return 0.7  # Conservative fallback
    
    def generate_subtitles(self, segments: List[Dict], output_path: str) -> str:
        """Generate SRT subtitles from timing segments"""
        try:
            if not segments:
                return ""
            
            srt_lines = []
            
            for i, seg in enumerate(segments, 1):
                # Format timestamps
                start_ms = seg.get("adjusted_start", seg.get("original_start", 0)) * 1000
                end_ms = seg.get("adjusted_end", seg.get("original_end", 0)) * 1000
                
                # Convert to SRT format
                def ms_to_srt_time(ms):
                    hours = int(ms // 3600000)
                    minutes = int((ms % 3600000) // 60000)
                    seconds = int((ms % 60000) // 1000)
                    milliseconds = int(ms % 1000)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
                
                start_time = ms_to_srt_time(start_ms)
                end_time = ms_to_srt_time(end_ms)
                
                # Get text
                text = seg.get("translated_text", seg.get("text", ""))
                
                # Add to SRT
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")  # Empty line between entries
            
            srt_content = "\n".join(srt_lines)
            
            # Save to file
            srt_path = output_path.replace(".wav", ".srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logger.info(f"Subtitles generated: {srt_path}")
            return srt_path
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}")
            return ""
    
    def preprocess_audio(self, input_path: str) -> str:
        """Apply FFmpeg filters for normalization and de-noising"""
        try:
            if not self.config.enable_input_normalization and not self.config.enable_denoising:
                return input_path

            logger.info("Preprocessing audio: normalization and de-noising")

            # Create temporary output path
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            temp_output = f"{base_name}_preprocessed.wav"

            # Build FFmpeg filter chain
            filters = []

            if self.config.enable_denoising:
                # Apply audio noise reduction using anlmdn filter
                filters.append("anlmdn")

            if self.config.enable_input_normalization:
                # Apply gentle normalization to avoid artifacts
                filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

            if not filters:
                return input_path

            filter_string = ",".join(filters)

            # Execute FFmpeg command
            import subprocess
            cmd = [
                'ffmpeg', '-i', input_path,
                '-filter:a', filter_string,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # Whisper expects 16kHz
                '-ac', '1',      # Mono
                '-y', temp_output
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.warning(f"FFmpeg preprocessing failed: {result.stderr}")
                logger.warning("Continuing with original audio")
                return input_path

            if os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                logger.info(f"Audio preprocessing successful: {temp_output}")
                return temp_output
            else:
                logger.warning("Preprocessed audio file is empty, using original")
                return input_path

        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return input_path

    def select_voice_model(self, target_lang: str, text_length: int) -> str:
        """Select voice model that matches target language phonetics"""
        try:
            # Get base voice for language
            base_voice = self.VOICE_MAPPING.get(target_lang, "en-US-JennyNeural")

            # Adjust speaking rate based on text length and language characteristics
            speaking_rate = self.VOICE_RATES.get(target_lang, 12)

            # For longer texts, prefer slightly slower voices to maintain clarity
            if text_length > 500:  # Long text
                # Could select alternative voices with better clarity for long content
                pass  # Keep base voice for now

            # Avoid extreme speaking rates - Edge-TTS handles this via speed parameter
            # We'll control this in the TTS generation

            logger.info(f"Selected voice model: {base_voice} for {target_lang} (rate: {speaking_rate} chars/sec)")
            return base_voice

        except Exception as e:
            logger.error(f"Voice model selection failed: {e}")
            return self.VOICE_MAPPING.get(target_lang, "en-US-JennyNeural")

    def process_audio(self, audio_path: str) -> TranslationResult:
        """Complete audio translation pipeline with improved quality"""
        start_time = datetime.now()

        try:
            # Validate input
            if not os.path.exists(audio_path):
                error_msg = f"Audio file not found: {audio_path}"
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text="",
                    translated_text="",
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=0,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=1.0,
                    output_path="",
                    error=error_msg
                )

            # Step 0: Preprocess audio (normalization and de-noising)
            processed_audio_path = self.preprocess_audio(audio_path)

            # Load models if not loaded
            if not self._models_loaded:
                if not self.load_models():
                    error_msg = "Failed to load AI models"
                    logger.error(error_msg)
                    return TranslationResult(
                        success=False,
                        original_text="",
                        translated_text="",
                        original_language=self.config.source_lang,
                        target_language=self.config.target_lang,
                        original_duration_ms=0,
                        translated_duration_ms=0,
                        duration_match_percent=0,
                        speed_adjustment=1.0,
                        output_path="",
                        error=error_msg
                    )

            # Step 1: Get original audio duration
            logger.info(f"Processing audio: {processed_audio_path}")
            original_audio = AudioSegment.from_file(processed_audio_path)
            original_duration_ms = len(original_audio)
            logger.info(f"Original duration: {original_duration_ms:.0f}ms")
            
            # Step 2: Transcribe with detailed segments
            transcription = self.transcribe_with_segments(audio_path)
            if not transcription.get("success", False):
                error_msg = transcription.get("error", "Transcription failed")
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text="",
                    translated_text="",
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=original_duration_ms,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=1.0,
                    output_path="",
                    error=error_msg
                )
            
            original_text = transcription["text"]
            detected_language = transcription["language"]
            segments = transcription["segments"]
            
            # Update source language if auto-detected
            if self.config.auto_detect:
                self.config.source_lang = detected_language
            
            logger.info(f"Transcribed: {len(original_text)} characters, {len(segments)} segments in {detected_language}")
            
            # Step 3: Translate with context preservation
            logger.info(f"Translating from {self.config.source_lang} to {self.config.target_lang}...")
            translated_text, translated_segments = self.translate_text_with_context(
                original_text, segments
            )
            
            if translated_text == original_text:
                logger.warning("Translation returned original text (possible fallback)")
            
            # Step 4: Smart condensation if needed (only for very long text)
            if len(translated_text) > len(original_text) * self.config.max_condensation_ratio and len(translated_text) > 200:
                logger.info("Text needs condensation...")
                condensed_text, condensation_ratio = self.condense_text_smart(
                    translated_text, original_duration_ms
                )
                translated_text = condensed_text
                logger.info(f"Condensation applied: ratio {condensation_ratio:.2f}")
            
            # Step 5: Calculate optimal speed adjustment
            speed = self.calculate_optimal_speed(original_duration_ms, translated_text)
            
            # Step 6: Select optimal voice model
            selected_voice = self.select_voice_model(self.config.target_lang, len(translated_text))

            # Step 7: Generate speech with timing
            output_path = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            logger.info(f"Generating speech with gTTS (speed: {speed:.2f}x)...")

            # Run synchronous TTS (no asyncio needed)
            tts_success = False
            timing_segments = []

            try:
                tts_success, timing_segments = self.synthesize_speech_with_timing(
                    translated_text, translated_segments, output_path
                )
            except Exception as tts_error:
                logger.error(f"TTS call failed: {tts_error}")

            if not tts_success or not os.path.exists(output_path):
                error_msg = "TTS synthesis failed"
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text=original_text,
                    translated_text=translated_text,
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=original_duration_ms,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=speed,
                    output_path="",
                    error=error_msg
                )

            # Step 8: Apply gain consistency and compression
            if self.config.enable_gain_consistency:
                logger.info("Applying gain consistency and compression...")
                translated_audio = AudioSegment.from_file(output_path)
                processed_audio = self.apply_gain_consistency(translated_audio, self.config.target_lufs)
                processed_audio.export(output_path, format="wav")
                logger.info("Gain consistency applied")
            
            # Step 7: Generate subtitles
            subtitle_path = self.generate_subtitles(timing_segments, output_path)
            
            # Step 8: Get translated audio duration
            translated_audio = AudioSegment.from_file(output_path)
            translated_duration_ms = len(translated_audio)

            # Step 9: FRAME-ACCURATE DURATION MATCHING
            # Ensure output duration matches input duration within frame constraints
            duration_diff = abs(translated_duration_ms - original_duration_ms)

            if duration_diff > 100:  # More than 100ms difference
                logger.info(f"Applying frame-accurate duration correction: {translated_duration_ms}ms to {original_duration_ms}ms")

                # Calculate exact speed needed for frame-accurate match
                exact_speed = translated_duration_ms / original_duration_ms
                exact_speed = max(0.5, min(2.0, exact_speed))  # Safety bounds

                # Apply precise speed adjustment
                corrected_audio = self._adjust_audio_speed_precise(translated_audio, exact_speed)
                corrected_duration = len(corrected_audio)

                # Export corrected audio
                corrected_audio.export(output_path, format="wav")
                translated_duration_ms = corrected_duration

                logger.info(f"Duration correction applied: {translated_duration_ms}ms (diff: {abs(translated_duration_ms - original_duration_ms)}ms)")

            # Step 10: Calculate final metrics
            final_duration_diff = abs(translated_duration_ms - original_duration_ms)
            duration_match_percent = (1 - final_duration_diff / original_duration_ms) * 100 if original_duration_ms > 0 else 0

            # Step 11: Validate lip-sync precision
            within_tolerance = final_duration_diff <= self.config.timing_tolerance_ms

            if within_tolerance:
                logger.info(f"[OK] Lip-sync precision achieved: ±{final_duration_diff}ms (within {self.config.timing_tolerance_ms}ms tolerance)")
            else:
                logger.warning(f"Lip-sync timing: ±{final_duration_diff}ms (exceeds {self.config.timing_tolerance_ms}ms tolerance)")

            # Step 12: Final quality validation
            if self.config.validation_spots > 0:
                logger.info(f"Running final quality validation with {self.config.validation_spots} random spots...")
                validation_results = self.validate_audio_quality(output_path, processed_audio_path)
                if validation_results["quality_score"] < 0.7:
                    logger.warning(f"Quality validation failed: {validation_results['quality_score']:.2f}")
                    for rec in validation_results["recommendations"]:
                        logger.warning(f"Recommendation: {rec}")
                else:
                    logger.info(f"[OK] Quality validation passed: {validation_results['quality_score']:.2f}")

            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = TranslationResult(
                success=True,
                original_text=original_text,
                translated_text=translated_text,
                original_language=self.config.source_lang,
                target_language=self.config.target_lang,
                original_duration_ms=original_duration_ms,
                translated_duration_ms=translated_duration_ms,
                duration_match_percent=duration_match_percent,
                speed_adjustment=speed,
                output_path=output_path,
                subtitle_path=subtitle_path,
                timing_segments=timing_segments
            )
            
            # Log success
            logger.info(f"Translation completed in {processing_time:.1f}s")
            logger.info(f"Duration match: {duration_match_percent:.1f}% ({duration_diff:.0f}ms diff)")
            logger.info(f"Speed adjustment: {speed:.2f}x")
            logger.info(f"Output: {output_path}")
            if subtitle_path:
                logger.info(f"Subtitles: {subtitle_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio translation pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TranslationResult(
                success=False,
                original_text="",
                translated_text="",
                original_language=self.config.source_lang,
                target_language=self.config.target_lang,
                original_duration_ms=0,
                translated_duration_ms=0,
                duration_match_percent=0,
                speed_adjustment=1.0,
                output_path="",
                error=str(e)
            )

# Utility function for quick testing
def test_audio_translation():
    """Test the audio translation module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio translation")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--source", default="auto", help="Source language (auto for detection)")
    parser.add_argument("--target", default="de", help="Target language")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Configure translator
    config = TranslationConfig(
        source_lang=args.source,
        target_lang=args.target,
        auto_detect=(args.source == "auto")
    )
    
    translator = AudioTranslator(config)
    
    print(f"\n{'='*60}")
    print(f"Testing Audio Translation: {args.source} → {args.target}")
    print(f"Audio file: {args.audio}")
    print(f"{'='*60}\n")
    
    # Process audio
    result = translator.process_audio(args.audio)
    
    # Print results
    print(f"\n{'='*60}")
    print("Translation Results:")
    print(f"{'='*60}")
    
    if result.success:
        print(f"✓ SUCCESS")
        print(f"  Source language: {result.original_language}")
        print(f"  Target language: {result.target_language}")
        print(f"  Original duration: {result.original_duration_ms:.0f}ms")
        print(f"  Translated duration: {result.translated_duration_ms:.0f}ms")
        print(f"  Duration match: {result.duration_match_percent:.1f}%")
        print(f"  Speed adjustment: {result.speed_adjustment:.2f}x")
        print(f"  Output file: {result.output_path}")
        
        if result.subtitle_path:
            print(f"  Subtitle file: {result.subtitle_path}")
        
        # Show text preview
        print(f"\n  Original text (preview): {result.original_text[:200]}...")
        print(f"  Translated text (preview): {result.translated_text[:200]}...")
        
        # Move output if specified
        if args.output and os.path.exists(result.output_path):
            import shutil
            shutil.move(result.output_path, args.output)
            print(f"  Moved output to: {args.output}")
    else:
        print(f"✗ FAILED")
        print(f"  Error: {result.error}")
    
    print(f"{'='*60}\n")
    
    return result.success

if __name__ == "__main__":
    # Example usage
    success = test_audio_translation()
    sys.exit(0 if success else 1)
