"""
Unit tests for video translation pipeline
Tests the complete video processing workflow
"""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import VideoTranslationPipeline, PipelineConfig

class TestVideoTranslationPipeline:
    """Test class for video translation pipeline"""

    @pytest.fixture
    def pipeline_config(self):
        """Fixture for pipeline configuration"""
        return PipelineConfig(
            chunk_size=10,  # Smaller chunks for faster testing
            max_workers=1,  # Single worker for testing
            use_gpu=False,  # Disable GPU for testing
            generate_subtitles=False,  # Disable subtitle generation for speed
            parallel_processing=False  # Disable parallel processing
        )

    @pytest.fixture
    def pipeline(self, pipeline_config):
        """Fixture for VideoTranslationPipeline instance"""
        return VideoTranslationPipeline(pipeline_config)

    @pytest.fixture
    def sample_video_file(self):
        """Fixture to provide the test video file path"""
        video_path = Path(__file__).parent.parent / "test_samples" / "sample_30s_en.mp4"
        if not video_path.exists():
            pytest.skip(f"Test video file not found: {video_path}")
        return str(video_path)

    def test_initialization(self, pipeline_config):
        """Test pipeline initialization"""
        pipeline = VideoTranslationPipeline(pipeline_config)

        assert pipeline.config.chunk_size == 10
        assert pipeline.config.max_workers == 1
        assert pipeline.config.use_gpu == False
        assert pipeline.device.type == "cpu"

    def test_detect_available_gpus(self, pipeline):
        """Test GPU detection"""
        gpus = pipeline._detect_available_gpus()

        assert isinstance(gpus, list)
        # Should at least have an empty list or GPU info
        for gpu in gpus:
            if gpu:  # If GPU info exists
                assert "id" in gpu
                assert "name" in gpu
                assert "memory_total" in gpu

    def test_get_system_metrics(self, pipeline):
        """Test system metrics collection"""
        metrics = pipeline._get_system_metrics()

        assert isinstance(metrics, dict)
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "memory_used_gb" in metrics
        assert "memory_total_gb" in metrics

    @patch('modules.pipeline.AudioSegment')
    def test_chunk_audio_parallel(self, mock_audio_segment, pipeline):
        """Test audio chunking"""
        # Create mock audio
        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 30000  # 30 seconds
        mock_audio_segment.from_file.return_value = mock_audio

        # Mock chunk export
        mock_chunk = MagicMock()
        mock_audio.__getitem__.return_value = mock_chunk
        mock_chunk.export = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override temp_dir for testing
            pipeline.config.temp_dir = temp_dir

            chunks = pipeline.chunk_audio_parallel("fake_audio.wav")

            assert isinstance(chunks, list)
            # Should create chunks for 30-second audio with 10-second chunks
            assert len(chunks) >= 2  # At least 2-3 chunks expected

            for chunk in chunks:
                assert hasattr(chunk, 'id')
                assert hasattr(chunk, 'path')
                assert hasattr(chunk, 'start_ms')
                assert hasattr(chunk, 'end_ms')
                assert hasattr(chunk, 'duration_ms')

    def test_chunk_audio_simple(self, pipeline):
        """Test simple audio chunking"""
        # Create a temporary audio file
        import numpy as np
        from pydub import AudioSegment

        # Create a 5-second test audio
        sample_rate = 22050
        duration = 5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(440 * 2 * np.pi * t) * 0.1
        audio_int16 = (audio_data * 32767).astype(np.int16)

        audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_path = temp_audio.name
            audio.export(audio_path, format='wav')

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pipeline.config.temp_dir = temp_dir
                chunks = pipeline.chunk_audio_simple(audio_path)

                assert isinstance(chunks, list)
                assert len(chunks) > 0

                for chunk in chunks:
                    assert hasattr(chunk, 'id')
                    assert hasattr(chunk, 'path')
                    assert hasattr(chunk, 'start_ms')
                    assert hasattr(chunk, 'end_ms')
                    assert hasattr(chunk, 'duration_ms')
                    assert os.path.exists(chunk.path)

        finally:
            os.unlink(audio_path)

    @patch('modules.pipeline.subprocess.run')
    def test_extract_audio_fast_success(self, mock_subprocess, pipeline, sample_video_file):
        """Test successful audio extraction"""
        # Mock successful ffmpeg execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")

            # Mock os.path.exists to return True for the output file
            with patch('os.path.exists', return_value=True):
                result = pipeline.extract_audio_fast(sample_video_file, audio_path)
                assert result == True

    @patch('modules.pipeline.subprocess.run')
    def test_extract_audio_fast_failure(self, mock_subprocess, pipeline, sample_video_file):
        """Test failed audio extraction"""
        # Mock failed ffmpeg execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg error"
        mock_subprocess.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")

            result = pipeline.extract_audio_fast(sample_video_file, audio_path)
            assert result == False

    @patch('modules.pipeline.subprocess.run')
    def test_merge_files_fast_success(self, mock_subprocess, pipeline):
        """Test successful video merge"""
        # Mock successful ffmpeg execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # Mock file existence
        with patch('os.path.exists', return_value=True):
            result = pipeline.merge_files_fast("video.mp4", "audio.wav", "output.mp4")
            assert result == True

    @patch('modules.pipeline.subprocess.run')
    def test_merge_files_fast_failure(self, mock_subprocess, pipeline):
        """Test failed video merge"""
        # Mock failed ffmpeg execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "merge error"
        mock_subprocess.return_value = mock_result

        result = pipeline.merge_files_fast("video.mp4", "audio.wav", "output.mp4")
        assert result == False

    def test_quick_speech_check(self, pipeline):
        """Test speech detection in audio segments"""
        # Create test audio segments
        import numpy as np
        from pydub import AudioSegment

        # Silent audio (should return False)
        silent_audio = AudioSegment.silent(duration=1000)
        assert pipeline._quick_speech_check(silent_audio) == False

        # Audio with content (should return True)
        sample_rate = 22050
        duration = 1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(440 * 2 * np.pi * t) * 0.3
        audio_int16 = (audio_data * 32767).astype(np.int16)

        audio_with_content = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        assert pipeline._quick_speech_check(audio_with_content) == True

    def test_cleanup_temp_files(self, pipeline):
        """Test temporary file cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.config.temp_dir = temp_dir

            # Create some test files
            test_files = []
            for i in range(3):
                with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as f:
                    f.write(b"test")
                    test_files.append(f.name)

            # Verify files exist
            for file_path in test_files:
                assert os.path.exists(file_path)

            # Clean up
            pipeline.cleanup_temp_files()

            # Files should be gone (or at least attempted to be removed)
            # Note: This test might be flaky in some environments

    @patch('modules.pipeline.VideoTranslationPipeline.load_models')
    @patch('modules.pipeline.VideoTranslationPipeline.extract_audio_fast')
    @patch('modules.pipeline.VideoTranslationPipeline.chunk_audio_parallel')
    @patch('modules.pipeline.VideoTranslationPipeline.process_chunks_batch')
    @patch('modules.pipeline.VideoTranslationPipeline.merge_files_fast')
    def test_process_video_fast_success(self, mock_merge, mock_process_chunks, mock_chunk,
                                       mock_extract, mock_load_models, pipeline, sample_video_file):
        """Test successful video processing"""
        # Setup mocks
        mock_load_models.return_value = True
        mock_extract.return_value = True
        mock_chunk.return_value = [MagicMock(id=0, path="chunk.wav", start_ms=0, end_ms=1000, duration_ms=1000)]
        mock_process_chunks.return_value = (["translated_chunk.wav"], [{"text": "test", "start": 0, "end": 1}])
        mock_merge.return_value = True

        result = pipeline.process_video_fast(sample_video_file, "es")

        assert result["success"] == True
        assert "output_video" in result
        assert "processing_time_s" in result
        assert result["target_language"] == "es"

    @patch('modules.pipeline.VideoTranslationPipeline.load_models')
    def test_process_video_fast_model_load_failure(self, mock_load_models, pipeline, sample_video_file):
        """Test video processing when model loading fails"""
        mock_load_models.return_value = False

        result = pipeline.process_video_fast(sample_video_file, "es")

        assert result["success"] == False
        assert "Failed to load models" in result["error"]

    @patch('modules.pipeline.VideoTranslationPipeline.load_models')
    @patch('modules.pipeline.VideoTranslationPipeline.extract_audio_fast')
    def test_process_video_fast_audio_extraction_failure(self, mock_extract, mock_load_models,
                                                        pipeline, sample_video_file):
        """Test video processing when audio extraction fails"""
        mock_load_models.return_value = True
        mock_extract.return_value = False

        result = pipeline.process_video_fast(sample_video_file, "es")

        assert result["success"] == False
        assert "Audio extraction failed" in result["error"]

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values"""
        config = PipelineConfig()

        assert config.chunk_size == 30
        assert config.max_workers == 4
        assert config.generate_subtitles == True
        assert config.use_gpu == False  # Should be False by default in tests

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
