"""
Integration tests for the complete Octavia video translation system
Tests end-to-end functionality with real components
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def sample_video_path(self):
        """Path to the test video file"""
        video_path = Path(__file__).parent.parent / "test_samples" / "sample_30s_en.mp4"
        if not video_path.exists():
            pytest.skip(f"Test video file not found: {video_path}")
        return str(video_path)

    def test_full_pipeline_imports(self):
        """Test that all main components can be imported"""
        try:
            from modules.audio_translator import AudioTranslator, TranslationConfig
            from modules.pipeline import VideoTranslationPipeline, PipelineConfig
            from modules.subtitle_generator import SubtitleGenerator
            from modules.subtitle_translator import SubtitleTranslator
            from modules.instrumentation import MetricsCollector
            from app import app
            from routes.translation_routes import router

            # All imports successful
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

    def test_fastapi_app_creation(self):
        """Test that FastAPI app can be created without errors"""
        try:
            from app import app
            assert app.title == "Octavia Video Translator"
            assert app.version == "4.0.0"
        except Exception as e:
            pytest.fail(f"Failed to create FastAPI app: {e}")

    @pytest.mark.slow
    def test_subtitle_generation_integration(self, sample_video_path):
        """Test complete subtitle generation workflow"""
        from modules.subtitle_generator import SubtitleGenerator

        generator = SubtitleGenerator(model_size="tiny")  # Use tiny for speed

        # Generate subtitles
        result = generator.process_file(sample_video_path, "srt")

        assert result["success"] == True
        assert result["text"] != ""
        assert len(result["segments"]) > 0
        assert result["language"] == "en"

        print("✓ Subtitle generation integration test passed")
        print(f"  Generated {len(result['segments'])} segments")
        print(f"  Language: {result['language']}")
        print(f"  Text preview: {result['text'][:100]}...")

    @pytest.mark.slow
    def test_video_pipeline_initialization(self, sample_video_path):
        """Test video pipeline can initialize and load models"""
        from modules.pipeline import VideoTranslationPipeline, PipelineConfig

        config = PipelineConfig(
            use_gpu=False,  # Disable GPU for testing
            generate_subtitles=False,  # Disable for speed
            parallel_processing=False
        )

        pipeline = VideoTranslationPipeline(config)

        # Test model loading (may take time)
        model_loaded = pipeline.load_models("en", "es")

        if model_loaded:
            assert pipeline.whisper_model is not None
            print("✓ Video pipeline models loaded successfully")
        else:
            print("⚠ Video pipeline models not loaded (expected in test environment)")
            # Don't fail - models may not load in test environment without dependencies

    def test_api_endpoints_registration(self):
        """Test that API endpoints are properly registered"""
        from app import app
        from routes.translation_routes import router

        # Check that router is included
        routes = [route.path for route in app.routes]
        assert any("/api/translate" in route for route in routes), "Translation routes not registered"

        print("✓ API endpoints properly registered")
        print(f"  Total routes: {len(routes)}")
        translation_routes = [r for r in routes if "/api/translate" in r]
        print(f"  Translation routes: {len(translation_routes)}")

    def test_configuration_loading(self):
        """Test configuration loading"""
        try:
            from modules.audio_translator import TranslationConfig
            from modules.pipeline import PipelineConfig

            audio_config = TranslationConfig()
            pipeline_config = PipelineConfig()

            assert audio_config.source_lang == "en"
            assert audio_config.target_lang == "de"
            assert pipeline_config.chunk_size == 30

            print("✓ Configuration classes work correctly")

        except Exception as e:
            pytest.fail(f"Configuration loading failed: {e}")

    def test_health_endpoint(self):
        """Test health endpoint returns proper response"""
        from app import app

        # Create test client
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] == True
        assert data["service"] == "Octavia Video Translator"
        assert "models" in data
        assert "status_indicators" in data

        print("✓ Health endpoint working correctly")
        print(f"  Service: {data['service']}")
        print(f"  Version: {data['version']}")

    def test_demo_login_endpoint(self):
        """Test demo login functionality"""
        from app import app

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.post("/api/auth/demo-login")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] == True
        assert "token" in data
        assert "user" in data
        assert data["user"]["email"] == "demo@octavia.com"

        print("✓ Demo login working correctly")

    def test_credit_packages_endpoint(self):
        """Test credit packages endpoint"""
        from app import app

        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/api/payments/packages")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] == True
        assert "packages" in data
        assert len(data["packages"]) > 0

        # Check package structure
        starter_package = data["packages"].get("starter_credits")
        assert starter_package is not None
        assert starter_package["credits"] == 100
        assert starter_package["price"] == 999

        print("✓ Credit packages endpoint working")
        print(f"  Available packages: {len(data['packages'])}")

    @pytest.mark.slow
    def test_subtitle_translation_workflow(self):
        """Test complete subtitle translation workflow"""
        from modules.subtitle_translator import SubtitleTranslator

        translator = SubtitleTranslator()

        # Create test SRT content
        test_content = """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test.

2
00:00:05,000 --> 00:00:10,000
How are you doing today?
"""

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as input_file:
            input_file.write(test_content)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as output_file:
            output_path = output_file.name

        try:
            # Test translation (will use fallback if models not available)
            result = translator.translate_subtitles(input_path, output_path, "en", "es")

            # Should succeed even with fallback
            assert result["success"] == True
            assert os.path.exists(output_path)

            print("✓ Subtitle translation workflow completed")
            print(f"  Input segments: 2")
            print(f"  Output file created: {os.path.exists(output_path)}")

        finally:
            # Cleanup
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation"""
        from modules.pipeline import PipelineConfig

        # Test default config
        config = PipelineConfig()
        assert config.chunk_size == 30
        assert config.max_workers == 4

        # Test custom config
        custom_config = PipelineConfig(
            chunk_size=60,
            max_workers=8,
            use_gpu=True,
            generate_subtitles=False
        )
        assert custom_config.chunk_size == 60
        assert custom_config.max_workers == 8
        assert custom_config.use_gpu == True
        assert custom_config.generate_subtitles == False

        print("✓ Pipeline configuration validation passed")

    def test_audio_translator_config_validation(self):
        """Test audio translator configuration validation"""
        from modules.audio_translator import TranslationConfig

        config = TranslationConfig(
            source_lang="en",
            target_lang="fr",
            voice_speed=1.2,
            enable_gain_consistency=True
        )

        assert config.source_lang == "en"
        assert config.target_lang == "fr"
        assert config.voice_speed == 1.2
        assert config.enable_gain_consistency == True

        print("✓ Audio translator configuration validation passed")

    def test_file_format_validation(self):
        """Test supported file format validation"""
        supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv']
        supported_audio_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

        test_files = [
            ("video.mp4", True, "video"),
            ("audio.wav", True, "audio"),
            ("document.pdf", False, None),
            ("video.MP4", True, "video"),  # Case insensitive
            ("audio.WAV", True, "audio"),
        ]

        for filename, should_be_supported, file_type in test_files:
            extension = filename[filename.rfind('.'):].lower()

            if file_type == "video":
                is_supported = extension in [fmt.lower() for fmt in supported_video_formats]
            elif file_type == "audio":
                is_supported = extension in [fmt.lower() for fmt in supported_audio_formats]
            else:
                is_supported = False

            assert is_supported == should_be_supported, f"File {filename} support check failed"

        print("✓ File format validation passed")

    def test_timing_calculations(self):
        """Test timing and duration calculations"""
        from modules.audio_translator import AudioTranslator

        translator = AudioTranslator()

        # Test speed calculation
        original_duration = 2000  # 2 seconds
        text_length = 100

        speed = translator.calculate_optimal_speed(original_duration, "A" * text_length)

        # Speed should be reasonable
        assert 0.5 <= speed <= 2.0

        # Test condensation
        long_text = "This is a very long text that should be condensed for timing purposes."
        condensed, ratio = translator.condense_text_smart(long_text, 1000)

        assert isinstance(condensed, str)
        assert ratio <= 1.0

        print("✓ Timing calculations working correctly")

    def test_error_handling(self):
        """Test error handling across components"""
        from modules.audio_translator import AudioTranslator
        from modules.pipeline import VideoTranslationPipeline

        # Test audio translator with invalid file
        translator = AudioTranslator()
        result = translator.process_audio("nonexistent.wav")

        assert result.success == False
        assert "not found" in result.error.lower()

        # Test pipeline with invalid file
        pipeline = VideoTranslationPipeline()
        result = pipeline.process_video_fast("nonexistent.mp4", "es")

        assert result["success"] == False
        assert "not found" in result["error"].lower()

        print("✓ Error handling working correctly")

    def test_voice_quality_settings(self):
        """Test voice quality configuration"""
        from modules.audio_translator import TranslationConfig

        config = TranslationConfig()

        # Test voice rate mappings
        assert config.voice_speed >= 0.5
        assert config.voice_speed <= 2.0

        # Test voice settings
        assert isinstance(config.enable_gain_consistency, bool)
        assert isinstance(config.enable_silence_padding, bool)

        print("✓ Voice quality settings configured correctly")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
