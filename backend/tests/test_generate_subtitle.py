"""
Unit test for subtitle generation functionality
Tests the SubtitleGenerator.process_file method using the sample_30s_en.mp4 video file
"""
import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.subtitle_generator import SubtitleGenerator

class TestSubtitleGeneration:
    """Test class for subtitle generation functionality"""

    @pytest.fixture
    def subtitle_generator(self):
        """Fixture to provide SubtitleGenerator instance"""
        return SubtitleGenerator(model_size="tiny")  # Use tiny model for faster testing

    @pytest.fixture
    def test_audio_file(self):
        """Fixture to provide the test video file path"""
        video_path = Path(__file__).parent.parent / "test_samples" / "sample_30s_en.mp4"
        assert video_path.exists(), f"Test video file not found: {video_path}"
        return str(video_path)

    def test_subtitle_generation_success(self, subtitle_generator, test_audio_file):
        """Test that subtitle generation completes successfully"""
        # Process the video file
        result = subtitle_generator.process_file(test_audio_file, output_format="srt")

        # Assert success
        assert result["success"] == True, f"Subtitle generation failed: {result.get('error', 'Unknown error')}"

        # Assert that text was transcribed
        assert result["text"] != "", "No text was transcribed"
        assert len(result["text"].strip()) > 0, "Transcribed text is empty"

        # Assert segments were created
        assert result["segment_count"] > 0, "No subtitle segments were generated"
        assert len(result["segments"]) > 0, "Segments list is empty"

        # Assert language detection
        assert result["language"] is not None, "Language not detected"
        assert result["language"] == "en", f"Expected English language, got: {result['language']}"

        print("✓ Subtitle generation completed successfully")
        print(f"  Language: {result['language']}")
        print(f"  Segments: {result['segment_count']}")
        print(f"  Text length: {len(result['text'])} characters")

    def test_subtitle_file_creation(self, subtitle_generator, test_audio_file):
        """Test that subtitle files are created on disk"""
        # Clean up any existing subtitle files first
        current_dir = Path.cwd()
        for ext in ['.srt', '.vtt']:
            subtitle_file = current_dir / f"subtitles{ext}"
            if subtitle_file.exists():
                subtitle_file.unlink()

        # Process the audio file
        result = subtitle_generator.process_file(test_audio_file, output_format="srt", generate_all=True)

        # Assert success
        assert result["success"] == True, f"Subtitle generation failed: {result.get('error', 'Unknown error')}"

        # Check that output files were created
        assert "output_files" in result, "No output_files in result"
        output_files = result["output_files"]

        # Should have SRT file at minimum
        assert "srt" in output_files, "SRT file not created"
        srt_path = Path(output_files["srt"])
        assert srt_path.exists(), f"SRT file not found: {srt_path}"

        # Check file has content
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        assert len(srt_content.strip()) > 0, "SRT file is empty"

        # Verify SRT format has proper structure
        lines = srt_content.strip().split('\n')
        assert len(lines) >= 3, "SRT file doesn't have minimum required lines"

        # First line should be a number (segment index)
        try:
            int(lines[0].strip())
        except ValueError:
            pytest.fail("SRT file doesn't start with segment number")

        print("✓ Subtitle files created successfully")
        print(f"  SRT file: {srt_path}")
        print(f"  File size: {srt_path.stat().st_size} bytes")

        # Clean up created files
        for file_path in output_files.values():
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass

    def test_subtitle_segments_structure(self, subtitle_generator, test_audio_file):
        """Test that subtitle segments have proper structure"""
        result = subtitle_generator.process_file(test_audio_file, output_format="srt")

        assert result["success"] == True, f"Subtitle generation failed: {result.get('error', 'Unknown error')}"

        segments = result["segments"]
        assert len(segments) > 0, "No segments generated"

        # Check each segment has required fields
        for i, segment in enumerate(segments):
            assert "start" in segment, f"Segment {i} missing 'start' field"
            assert "end" in segment, f"Segment {i} missing 'end' field"
            assert "text" in segment, f"Segment {i} missing 'text' field"

            # Check timing values are reasonable
            assert segment["start"] >= 0, f"Segment {i} has negative start time: {segment['start']}"
            assert segment["end"] > segment["start"], f"Segment {i} end time {segment['end']} not greater than start {segment['start']}"

            # Check text is not empty
            assert segment["text"].strip() != "", f"Segment {i} has empty text"

            # Check confidence if present
            if "confidence" in segment:
                assert 0.0 <= segment["confidence"] <= 1.0, f"Segment {i} confidence out of range: {segment['confidence']}"

        print("✓ Subtitle segments have proper structure")
        print(f"  Total segments: {len(segments)}")
        print(f"  First segment: {segments[0]['start']:.2f}s - {segments[0]['end']:.2f}s")
        print(f"  Last segment: {segments[-1]['start']:.2f}s - {segments[-1]['end']:.2f}s")

    def test_multiple_format_generation(self, subtitle_generator, test_audio_file):
        """Test generating subtitles in multiple formats"""
        result = subtitle_generator.process_file(test_audio_file, output_format="srt", generate_all=True)

        assert result["success"] == True, f"Subtitle generation failed: {result.get('error', 'Unknown error')}"

        output_files = result["output_files"]

        # Should have both SRT and VTT
        assert "srt" in output_files, "SRT file not generated"
        assert "vtt" in output_files, "VTT file not generated"

        # Check both files exist and have content
        for format_name, file_path in output_files.items():
            file_obj = Path(file_path)
            assert file_obj.exists(), f"{format_name.upper()} file not created: {file_path}"

            with open(file_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            assert len(content.strip()) > 0, f"{format_name.upper()} file is empty"

        print("✓ Multiple subtitle formats generated successfully")
        for format_name, file_path in output_files.items():
            file_size = Path(file_path).stat().st_size
            print(f"  {format_name.upper()}: {file_path} ({file_size} bytes)")

        # Clean up
        for file_path in output_files.values():
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
