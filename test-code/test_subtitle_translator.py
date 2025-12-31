"""
Unit tests for subtitle translation functionality
Tests Helsinki NLP translation integration
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.subtitle_translator import SubtitleTranslator

class TestSubtitleTranslator:
    """Test class for subtitle translation functionality"""

    @pytest.fixture
    def translator(self):
        """Fixture for SubtitleTranslator instance"""
        return SubtitleTranslator()

    @pytest.fixture
    def sample_srt_content(self):
        """Sample SRT content for testing"""
        return """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test.

2
00:00:05,000 --> 00:00:10,000
How are you doing today?

3
00:00:10,000 --> 00:00:15,000
Thank you for watching.
"""

    def test_initialization(self, translator):
        """Test SubtitleTranslator initialization"""
        assert translator.source_lang == "en"
        assert translator.target_lang == "es"
        assert translator.translator is None

    @patch('modules.subtitle_translator.pipeline')
    def test_load_translator_success(self, mock_pipeline, translator):
        """Test successful translator loading"""
        mock_translator = MagicMock()
        mock_pipeline.return_value = mock_translator

        result = translator.load_translator("en", "es")

        assert result == True
        assert translator.translator == mock_translator

    @patch('modules.subtitle_translator.pipeline')
    def test_load_translator_failure(self, mock_pipeline, translator):
        """Test translator loading failure"""
        mock_pipeline.side_effect = Exception("Model not found")

        result = translator.load_translator("en", "xx")

        assert result == False
        assert translator.translator is None

    def test_parse_srt(self, translator, sample_srt_content):
        """Test SRT parsing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(sample_srt_content)
            srt_path = f.name

        try:
            segments = translator.parse_srt(srt_path)

            assert len(segments) == 3

            # Check first segment
            assert segments[0]['index'] == 1
            assert segments[0]['start'] == 1.0
            assert segments[0]['end'] == 5.0
            assert segments[0]['text'] == "Hello world, this is a test."

            # Check second segment
            assert segments[1]['index'] == 2
            assert segments[1]['start'] == 5.0
            assert segments[1]['end'] == 10.0
            assert segments[1]['text'] == "How are you doing today?"

        finally:
            os.unlink(srt_path)

    def test_parse_srt_empty_file(self, translator):
        """Test parsing empty SRT file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write("")
            srt_path = f.name

        try:
            segments = translator.parse_srt(srt_path)
            assert segments == []
        finally:
            os.unlink(srt_path)

    def test_translate_subtitle_text(self, translator):
        """Test individual subtitle text translation"""
        # Mock the translator
        translator.translator = MagicMock()
        translator.translator.return_value = [{"translation_text": "Hola mundo"}]

        result = translator.translate_subtitle_text("Hello world")

        assert result == "Hola mundo"
        translator.translator.assert_called_once()

    def test_translate_subtitle_text_no_translator(self, translator):
        """Test translation without loaded translator"""
        translator.translator = None

        result = translator.translate_subtitle_text("Hello world")

        # Should return original text as fallback
        assert result == "Hello world"

    @patch('modules.subtitle_translator.pipeline')
    def test_translate_subtitles_full_process(self, mock_pipeline, translator, sample_srt_content):
        """Test complete subtitle translation process"""
        # Setup mock translator
        mock_translator = MagicMock()
        mock_translator.return_value = [{"translation_text": "Texto traducido"}]
        mock_pipeline.return_value = mock_translator

        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as input_file:
            input_file.write(sample_srt_content)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as output_file:
            output_path = output_file.name

        try:
            result = translator.translate_subtitles(input_path, output_path, "en", "es")

            assert result['success'] == True
            assert 'translated_segments' in result
            assert len(result['translated_segments']) == 3

            # Check that output file was created
            assert os.path.exists(output_path)

            # Read output file
            with open(output_path, 'r', encoding='utf-8') as f:
                output_content = f.read()

            # Should contain translated content
            assert "Texto traducido" in output_content

        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_translate_subtitles_file_not_found(self, translator):
        """Test translation with nonexistent input file"""
        result = translator.translate_subtitles("nonexistent.srt", "output.srt", "en", "es")

        assert result['success'] == False
        assert 'error' in result

    def test_format_timestamp(self, translator):
        """Test timestamp formatting"""
        from datetime import timedelta

        # Test various timestamps
        test_cases = [
            (timedelta(seconds=1, milliseconds=500), "00:00:01,500"),
            (timedelta(minutes=1, seconds=30), "00:01:30,000"),
            (timedelta(hours=1, minutes=2, seconds=3, milliseconds=456), "01:02:03,456"),
        ]

        for td, expected in test_cases:
            result = translator.format_timestamp(td)
            assert result == expected

    def test_write_srt(self, translator):
        """Test SRT file writing"""
        segments = [
            {
                'index': 1,
                'start': 1.0,
                'end': 5.0,
                'text': 'Hello world'
            },
            {
                'index': 2,
                'start': 5.0,
                'end': 10.0,
                'text': 'How are you?'
            }
        ]

        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as output_file:
            output_path = output_file.name

        try:
            translator.write_srt(segments, output_path)

            # Read back the file
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Verify structure
            lines = content.strip().split('\n')
            assert lines[0] == '1'
            assert '00:00:01,000 --> 00:00:05,000' in content
            assert 'Hello world' in content
            assert lines[4] == '2'
            assert '00:00:05,000 --> 00:00:10,000' in content
            assert 'How are you?' in content

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_get_available_languages(self, translator):
        """Test available languages list"""
        languages = translator.get_available_languages()

        assert isinstance(languages, dict)
        assert 'en' in languages
        assert 'es' in languages
        assert 'de' in languages

        # Check that each language has a proper name
        for code, name in languages.items():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_is_language_pair_supported(self, translator):
        """Test language pair support checking"""
        # Supported pairs
        assert translator.is_language_pair_supported('en', 'es') == True
        assert translator.is_language_pair_supported('en', 'de') == True

        # Unsupported pairs
        assert translator.is_language_pair_supported('en', 'xx') == False
        assert translator.is_language_pair_supported('xx', 'en') == False

    def test_estimate_translation_time(self, translator, sample_srt_content):
        """Test translation time estimation"""
        # Count characters in sample content
        char_count = len(sample_srt_content)

        estimated_time = translator.estimate_translation_time(char_count)

        # Should be a reasonable positive number
        assert estimated_time > 0
        assert isinstance(estimated_time, float)

        # Time should scale with content length
        longer_time = translator.estimate_translation_time(char_count * 2)
        assert longer_time > estimated_time

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
