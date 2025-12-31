"""
Unit tests for audio translation functionality
Tests translation, TTS, and audio processing pipeline
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.audio_translator import AudioTranslator, TranslationConfig, TranslationResult

class TestAudioTranslator:
    """Test class for audio translation functionality"""

    @pytest.fixture
    def translator_config(self):
        """Fixture for translation configuration"""
        return TranslationConfig(
            source_lang="en",
            target_lang="es",
            auto_detect=False,
            voice_speed=1.0,
            enable_gain_consistency=False,  # Disable for faster testing
            validation_spots=0  # Disable validation for testing
        )

    @pytest.fixture
    def translator(self, translator_config):
        """Fixture for AudioTranslator instance"""
        return AudioTranslator(translator_config)

    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary sample audio file for testing"""
        import numpy as np
        from pydub import AudioSegment

        # Create a simple 2-second audio file
        sample_rate = 22050
        duration = 2  # seconds
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3  # Low volume sine wave

        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create AudioSegment
        audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format='wav')

        yield temp_path

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

    def test_initialization(self, translator_config):
        """Test AudioTranslator initialization"""
        translator = AudioTranslator(translator_config)

        assert translator.config.source_lang == "en"
        assert translator.config.target_lang == "es"
        assert translator._models_loaded == False
        assert translator.whisper_model is None
        assert translator.translation_pipeline is None

    @patch('modules.audio_translator.whisper')
    def test_load_models_success(self, mock_whisper, translator):
        """Test successful model loading"""
        # Mock the whisper module
        mock_whisper.load_model.return_value = MagicMock()
        mock_whisper.pad_or_trim.return_value = MagicMock()
        mock_whisper.log_mel_spectrogram.return_value = MagicMock()

        # Mock transformers pipeline
        with patch('modules.audio_translator.pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()

            result = translator.load_models()

            assert result == True
            assert translator._models_loaded == True
            assert translator.whisper_model is not None
            assert translator.translation_pipeline is not None

    def test_detect_language(self, translator, sample_audio_file):
        """Test language detection"""
        # Skip if models not loaded (would need actual model for full test)
        if not translator._models_loaded:
            pytest.skip("Models not loaded - skipping language detection test")

        result = translator.detect_language(sample_audio_file)
        assert isinstance(result, str)
        assert len(result) == 2  # Language codes are 2 characters

    def test_transcribe_with_segments_empty_file(self, translator):
        """Test transcription with empty/nonexistent file"""
        result = translator.transcribe_with_segments("nonexistent_file.wav")

        assert result["success"] == False
        assert result["text"] == ""
        assert result["segments"] == []
        assert "error" in result

    def test_translate_text_with_context_empty(self, translator):
        """Test translation with empty text"""
        result_text, result_segments = translator.translate_text_with_context("", [])

        assert result_text == ""
        assert result_segments == []

    def test_condense_text_smart(self, translator):
        """Test text condensation"""
        long_text = "This is a very long text that should be condensed because it exceeds the maximum allowed length for the target duration."
        target_duration = 1000  # 1 second

        condensed, ratio = translator.condense_text_smart(long_text, target_duration)

        assert isinstance(condensed, str)
        assert ratio <= 1.0
        assert ratio >= 0.0

    def test_calculate_optimal_speed(self, translator):
        """Test speed calculation"""
        original_duration = 2000  # 2 seconds
        text = "Hello world"

        speed = translator.calculate_optimal_speed(original_duration, text)

        assert 0.5 <= speed <= 2.0  # Reasonable speed range

    @patch('modules.audio_translator.AudioSegment')
    def test_synthesize_speech_with_timing_empty_text(self, mock_audio_segment, translator):
        """Test TTS synthesis with empty text"""
        mock_audio_segment.silent.return_value = MagicMock()

        success, segments = translator.synthesize_speech_with_timing("", [], "output.wav")

        # Should handle empty text gracefully
        assert isinstance(success, bool)
        assert isinstance(segments, list)

    def test_voice_rates_mapping(self, translator):
        """Test voice rates are properly defined"""
        assert "en" in translator.VOICE_RATES
        assert "es" in translator.VOICE_RATES
        assert "de" in translator.VOICE_RATES
        assert "fr" in translator.VOICE_RATES

        # All rates should be reasonable (10-15 chars/second)
        for lang, rate in translator.VOICE_RATES.items():
            assert 8 <= rate <= 20, f"Unreasonable rate for {lang}: {rate}"

    def test_voice_mapping(self, translator):
        """Test voice mappings are defined"""
        assert "en" in translator.VOICE_MAPPING
        assert "es" in translator.VOICE_MAPPING
        assert "de" in translator.VOICE_MAPPING

        # All voices should be Edge-TTS voice names
        for lang, voice in translator.VOICE_MAPPING.items():
            assert "Neural" in voice, f"Voice {voice} doesn't appear to be Edge-TTS format"

    def test_select_voice_model(self, translator):
        """Test voice model selection"""
        voice = translator.select_voice_model("es", 100)

        assert isinstance(voice, str)
        assert len(voice) > 0

    @patch('modules.audio_translator.AudioSegment')
    def test_preprocess_audio(self, mock_audio_segment, translator, sample_audio_file):
        """Test audio preprocessing"""
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.__len__.return_value = 1000

        result = translator.preprocess_audio(sample_audio_file)

        # Should return the input path if no preprocessing needed
        assert result == sample_audio_file

    def test_process_audio_file_not_found(self, translator):
        """Test processing with nonexistent file"""
        result = translator.process_audio("nonexistent_file.wav")

        assert result.success == False
        assert "not found" in result.error.lower()
        assert result.output_path == ""

    def test_translation_config_validation(self):
        """Test TranslationConfig validation"""
        config = TranslationConfig(
            source_lang="en",
            target_lang="es",
            chunk_size=30,
            timing_tolerance_ms=200
        )

        assert config.source_lang == "en"
        assert config.target_lang == "es"
        assert config.chunk_size == 30
        assert config.timing_tolerance_ms == 200

    def test_translation_result_structure(self):
        """Test TranslationResult structure"""
        result = TranslationResult(
            success=True,
            original_text="Hello",
            translated_text="Hola",
            original_language="en",
            target_language="es",
            original_duration_ms=1000,
            translated_duration_ms=1100,
            duration_match_percent=90.9,
            speed_adjustment=1.1,
            output_path="output.wav"
        )

        assert result.success == True
        assert result.original_text == "Hello"
        assert result.translated_text == "Hola"
        assert result.duration_match_percent == 90.9
        assert result.speed_adjustment == 1.1

    def test_cleanup_temp_files(self, translator):
        """Test temporary file cleanup"""
        # Create a test file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        # Verify file exists
        assert os.path.exists(temp_path)

        # This would normally clean up temp files, but we can't easily test it
        # without setting up the temp_dir structure
        translator.cleanup_temp_files()

        # File should still exist (since it's not in temp_dir)
        assert os.path.exists(temp_path)

        # Clean up
        os.unlink(temp_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
