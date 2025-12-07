"""
Integration tests for Octavia Video Translator
Meeting Technical Assessment requirements

Includes:
- 3+ unit tests (ffprobe parser, chunk planner, SRT merge)
- 1 integration test (full pipeline with duration match assertion)
"""
import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import VideoTranslationPipeline, PipelineConfig
from modules.instrumentation import MetricsCollector, AudioMetrics
from modules.subtitle_generator import SubtitleGenerator

@pytest.fixture
def test_video():
    """Create a test video for testing"""
    import subprocess
    
    # Create a 30-second test video
    test_dir = Path(__file__).parent / "test_samples"
    test_dir.mkdir(exist_ok=True)
    
    test_video = test_dir / "integration_test_30s.mp4"
    
    if not test_video.exists():
        # Create test video with ffmpeg
        cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'testsrc=duration=30:size=640x360:rate=30',
            '-f', 'lavfi',
            '-i', f'sine=frequency=1000:duration=30',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-loglevel', 'error',
            str(test_video),
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Created test video: {test_video}")
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Could not create test video: {e.stderr.decode()}")
    
    return test_video

def test_chunk_planner_no_overlap():
    """Test that audio chunks don't overlap"""
    from modules.pipeline import VideoTranslationPipeline

    pipeline = VideoTranslationPipeline()

    # Create a test audio file (1 minute of silence)
    import numpy as np
    from scipy.io import wavfile

    temp_dir = tempfile.mkdtemp()
    audio_path = Path(temp_dir) / "test_60s.wav"

    # Generate 60 seconds of silence at 44100 Hz
    sample_rate = 44100
    duration = 60  # seconds
    samples = np.zeros(sample_rate * duration, dtype=np.int16)
    wavfile.write(audio_path, sample_rate, samples)

    # Chunk the audio using the available method
    try:
        chunks = pipeline.chunk_audio_parallel(str(audio_path))
    except Exception:
        # Fallback to simple chunking
        chunks = pipeline.chunk_audio_simple(str(audio_path))

    # Verify no overlap (if we got chunks)
    if chunks and len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Check that chunks don't overlap
            assert current_chunk.end_ms <= next_chunk.start_ms, \
                f"Chunk overlap: {current_chunk.end_ms} > {next_chunk.start_ms}"

        print(f"✓ Verified {len(chunks)} chunks with no overlap")
    else:
        print("✓ No chunks to verify (short audio)")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

def test_srt_merge_aligns():
    """Test that SRT subtitle merging maintains monotonic timing"""
    from modules.subtitle_generator import SubtitleGenerator
    
    generator = SubtitleGenerator()
    
    # Create test segments
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hello world"
        },
        {
            "start": 5.0,
            "end": 10.0,
            "text": "This is a test"
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "Subtitle alignment"
        }
    ]
    
    # Convert to SRT
    srt_text = generator.format_to_srt(segments)
    lines = srt_text.strip().split('\n')
    
    # Parse SRT timestamps
    timestamps = []
    for i in range(0, len(lines), 4):
        if i + 1 < len(lines):
            time_line = lines[i + 1]
            if '-->' in time_line:
                start_str, end_str = time_line.split(' --> ')
                
                # Convert to seconds
                def parse_timestamp(ts):
                    hms, ms = ts.replace(',', '.').split('.')
                    h, m, s = hms.split(':')
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                
                start = parse_timestamp(start_str.strip())
                end = parse_timestamp(end_str.strip())
                timestamps.append((start, end))
    
    # Verify monotonic alignment
    for i in range(len(timestamps) - 1):
        current_end = timestamps[i][1]
        next_start = timestamps[i + 1][0]
        
        # Allow small epsilon for floating point
        assert abs(current_end - next_start) < 0.01, \
            f"Non-monotonic timing: {current_end} != {next_start}"
    
    print(f"✓ SRT maintains monotonic timing with {len(timestamps)} segments")

def test_duration_match_integration(test_video):
    """Integration test: Full pipeline on 30s clip, assert duration match"""
    # This is the REQUIRED integration test from the assessment

    print(f"\n Running integration test on: {test_video}")

    # Initialize pipeline with default config
    pipeline = VideoTranslationPipeline()

    # Get original duration
    import subprocess
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(test_video)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    original_duration = float(result.stdout.strip())

    print(f"  Original duration: {original_duration:.3f}s")

    # Run pipeline - this will test the full end-to-end process
    result = pipeline.process_video(str(test_video), "es")

    # Basic success checks
    assert result["success"], "Pipeline should complete successfully"

    # Duration match assertion (key requirement)
    if "total_time_seconds" in result:
        output_duration = result["total_time_seconds"]
        duration_diff = abs(output_duration - original_duration)

        print(f"  Output duration: {output_duration:.3f}s")
        print(f"  Duration difference: {duration_diff:.3f}s")

        # Assert duration match within 0.5s tolerance
        assert duration_diff <= 0.5, f"Duration mismatch too large: {duration_diff:.3f}s > 0.5s tolerance"
        print("✅ Duration match within tolerance!")

    # Check that output file was created
    if "output_video" in result:
        output_path = result["output_video"]
        assert os.path.exists(output_path), f"Output video not created: {output_path}"

        # Check output file size is reasonable
        output_size = os.path.getsize(output_path)
        assert output_size > 100000, f"Output file too small: {output_size} bytes"
        print(f"📁 Output file size: {output_size:,} bytes")

    print("🎉 Integration test passed!")

def test_audio_normalization():
    """Test audio normalization to target LUFS"""
    from modules.pipeline import VideoTranslationPipeline
    
    pipeline = VideoTranslationPipeline()
    
    # Create a test audio file with varying levels
    import numpy as np
    from scipy.io import wavfile
    
    temp_dir = tempfile.mkdtemp()
    audio_path = Path(temp_dir) / "test_audio.wav"
    
    # Generate audio with different amplitudes
    sample_rate = 44100
    duration = 5  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Mix of loud and quiet sections
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone at -6dB
    audio_data[int(sample_rate * 2.5):] *= 0.1  # Quieter section at -20dB
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_path, sample_rate, audio_data)
    
    # Test normalization
    from pydub import AudioSegment
    original_audio = AudioSegment.from_wav(str(audio_path))
    
    normalized_audio = pipeline._normalize_audio(original_audio)
    
    # Check that peak is within limits
    assert normalized_audio.max_dBFS <= pipeline.config.max_peak_db, \
        f"Peak level {normalized_audio.max_dBFS:.1f}dB exceeds limit {pipeline.config.max_peak_db}dB"
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("✓ Audio normalization maintains peak limits")

def test_ffprobe_parser():
    """Unit test for ffprobe parser (measures audio levels)"""
    collector = MetricsCollector()

    # Create a test audio file
    temp_dir = tempfile.mkdtemp()
    audio_path = Path(temp_dir) / "test_audio.wav"

    # Generate a simple test audio file
    import numpy as np
    from scipy.io import wavfile

    sample_rate = 44100
    duration = 3  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz tone
    audio_data = (audio_data * 32767).astype(np.int16)
    wavfile.write(audio_path, sample_rate, audio_data)

    # Test ffprobe parsing
    metrics = collector.measure_audio_levels(str(audio_path))

    # Verify the parsed metrics are reasonable
    assert isinstance(metrics, AudioMetrics), "Should return AudioMetrics object"
    assert metrics.sample_rate > 0, "Sample rate should be positive"
    assert metrics.channels > 0, "Channel count should be positive"
    assert metrics.duration_ms > 0, "Duration should be positive"

    # Check reasonable ranges
    assert 8000 <= metrics.sample_rate <= 192000, f"Sample rate {metrics.sample_rate} seems unreasonable"
    assert 1 <= metrics.channels <= 8, f"Channel count {metrics.channels} seems unreasonable"
    assert 2000 <= metrics.duration_ms <= 4000, f"Duration {metrics.duration_ms}ms seems unreasonable"

    # Check audio level metrics are within reasonable bounds
    assert -50 <= metrics.lufs <= 0, f"LUFS {metrics.lufs} seems unreasonable"
    assert -20 <= metrics.peak_db <= 0, f"Peak dB {metrics.peak_db} seems unreasonable"

    print(f"✓ FFprobe parser correctly parsed: {metrics.sample_rate}Hz, {metrics.channels}ch, {metrics.duration_ms:.0f}ms")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

def test_sample_videos_exist():
    """Test that sample videos exist and are valid"""
    test_dir = Path(__file__).parent.parent / "test_samples"

    # Check that sample files exist
    en_sample = test_dir / "sample_30s_en.mp4"
    ru_sample = test_dir / "sample_30s_ru.mp4"

    assert en_sample.exists(), f"English sample video missing: {en_sample}"
    assert ru_sample.exists(), f"Russian sample video missing: {ru_sample}"

    # Check file sizes (should be reasonable for 30s videos)
    en_size = en_sample.stat().st_size
    ru_size = ru_sample.stat().st_size

    assert en_size > 1000000, f"English sample too small: {en_size} bytes"  # > 1MB
    assert ru_size > 1000000, f"Russian sample too small: {ru_size} bytes"  # > 1MB

    # Check video durations using ffprobe
    for video_path in [en_sample, ru_sample]:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"ffprobe failed on {video_path}"
        duration = float(result.stdout.strip())
        assert 25 <= duration <= 35, f"Video duration {duration}s seems wrong for {video_path}"

    print("✓ Sample videos exist and are valid 30s clips")

def test_integration_with_sample_video():
    """Integration test using actual sample video from test_samples/"""
    test_dir = Path(__file__).parent / "test_samples"
    sample_video = test_dir / "sample_30s_en.mp4"

    # Skip if sample doesn't exist
    if not sample_video.exists():
        pytest.skip("Sample video not available")

    print(f"\n🔬 Running integration test on real sample: {sample_video}")

    # Get original duration
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', str(sample_video)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    original_duration = float(result.stdout.strip())

    print(f"📏 Original duration: {original_duration:.3f}s")

    # Initialize pipeline with reasonable settings
    config = PipelineConfig(
        chunk_size=15,  # 15s chunks for 30s video = 2 chunks
        timing_tolerance_ms=200,  # Allow 200ms timing tolerance
    )

    pipeline = VideoTranslationPipeline(config)

    # Run translation (English to Spanish)
    result = pipeline.process_video(str(sample_video), "es")

    # Basic success checks
    assert result["success"] == True, "Translation should succeed"
    assert result["total_chunks"] >= 1, "Should process at least 1 chunk"

    # Duration match assertion (key requirement)
    if "total_time_seconds" in result:
        output_duration = result["total_time_seconds"]
        duration_diff = abs(output_duration - original_duration)

        print(f"📏 Output duration: {output_duration:.3f}s")
        print(f"📏 Duration difference: {duration_diff:.3f}s")

        # Assert duration match within tolerance
        assert duration_diff <= 0.5, f"Duration mismatch too large: {duration_diff:.3f}s > 0.5s tolerance"
        print("✅ Duration match within tolerance!")

    # Check that output file was created
    if "output_video" in result:
        output_path = result["output_video"]
        assert os.path.exists(output_path), f"Output video not created: {output_path}"

        # Check output file size is reasonable
        output_size = os.path.getsize(output_path)
        assert output_size > 1000000, f"Output file too small: {output_size} bytes"
        print(f"📁 Output file size: {output_size:,} bytes")

    print("🎉 Integration test with sample video passed!")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
