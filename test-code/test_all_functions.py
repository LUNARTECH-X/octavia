#!/usr/bin/env python3
"""
End-to-end test script for all Octavia translation functions
Tests subtitle generation, translation, subtitle-to-audio, and video translation
using the sample 30-second English video.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.subtitle_generator import SubtitleGenerator
from modules.subtitle_translator import SubtitleTranslator
from modules.audio_translator import AudioTranslator, TranslationConfig
from modules.pipeline import VideoTranslationPipeline, PipelineConfig

def test_subtitle_generation():
    """Test subtitle generation from video"""
    print("\n" + "="*60)
    print("TESTING: Subtitle Generation")
    print("="*60)

    video_path = "test_samples/sample_30s_en.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return False

    try:
        # Initialize subtitle generator
        generator = SubtitleGenerator(model_size="tiny")
        print("✓ SubtitleGenerator initialized")

        # Generate subtitles
        start_time = time.time()
        result = generator.process_file(video_path, "srt")
        end_time = time.time()

        print(".1f")
        print(f"  Language: {result['language']}")
        print(f"  Segments: {result['segment_count']}")
        print(f"  Success: {result['success']}")

        if result['success'] and result['segment_count'] > 0:
            print("✅ Subtitle generation PASSED")

            # Check if SRT file was created
            if 'output_files' in result and 'srt' in result['output_files']:
                srt_path = result['output_files']['srt']
                if os.path.exists(srt_path):
                    print(f"✅ SRT file created: {srt_path}")

                    # Read first few lines
                    with open(srt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        print("✅ SRT content preview:")
                        lines = content.split('\n')[:10]
                        for line in lines:
                            print(f"    {line}")
                else:
                    print("❌ SRT file not found")

            return True
        else:
            print(f"❌ Subtitle generation FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Subtitle generation error: {e}")
        return False

def test_subtitle_translation():
    """Test subtitle translation"""
    print("\n" + "="*60)
    print("TESTING: Subtitle Translation")
    print("="*60)

    # First generate subtitles if not exists
    video_path = "test_samples/sample_30s_en.mp4"
    generator = SubtitleGenerator(model_size="tiny")
    result = generator.process_file(video_path, "srt")

    if not result['success']:
        print("❌ Cannot test translation - subtitle generation failed")
        return False

    try:
        # Initialize translator
        translator = SubtitleTranslator()
        print("✓ SubtitleTranslator initialized")

        # Get SRT path
        srt_path = result['output_files']['srt']
        output_path = "test_translated_subtitles.srt"

        # Translate subtitles
        start_time = time.time()
        translate_result = translator.translate_subtitles(srt_path, output_path, "en", "es")
        end_time = time.time()

        print(".1f"        print(f"  Success: {translate_result['success']}")

        if translate_result['success']:
            print("✅ Subtitle translation PASSED")

            if os.path.exists(output_path):
                print(f"✅ Translated SRT file created: {output_path}")

                # Read content
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("✅ Translated content preview:")
                    lines = content.split('\n')[:10]
                    for line in lines:
                        print(f"    {line}")

                # Cleanup
                os.remove(output_path)

            return True
        else:
            print(f"❌ Subtitle translation FAILED: {translate_result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Subtitle translation error: {e}")
        return False

def test_subtitle_to_audio():
    """Test subtitle to audio conversion"""
    print("\n" + "="*60)
    print("TESTING: Subtitle to Audio")
    print("="*60)

    # First generate subtitles
    video_path = "test_samples/sample_30s_en.mp4"
    generator = SubtitleGenerator(model_size="tiny")
    result = generator.process_file(video_path, "srt")

    if not result['success']:
        print("❌ Cannot test subtitle-to-audio - subtitle generation failed")
        return False

    try:
        # Initialize audio translator
        config = TranslationConfig(
            source_lang="en",
            target_lang="es",
            auto_detect=False
        )
        translator = AudioTranslator(config)
        print("✓ AudioTranslator initialized")

        # Get SRT path and create test SRT content
        srt_content = """1
00:00:01,000 --> 00:00:05,000
Hello world, this is a test of subtitle to audio conversion.

2
00:00:05,000 --> 00:00:10,000
The quick brown fox jumps over the lazy dog.

3
00:00:10,000 --> 00:00:15,000
Thank you for testing the Octavia system.
"""

        # Create temporary SRT file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write(srt_content)
            temp_srt_path = f.name

        try:
            # Parse SRT
            segments = translator.parse_srt(temp_srt_path)
            print(f"✓ Parsed {len(segments)} SRT segments")

            # Generate audio
            output_path = "test_subtitle_audio.mp3"
            start_time = time.time()

            success, timing_segments = translator.synthesize_speech_with_timing(
                "Hello world, this is a test. The quick brown fox jumps over the lazy dog. Thank you for testing.",
                segments,
                output_path
            )

            end_time = time.time()

            print(".1f"            print(f"  Success: {success}")

            if success and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Audio file created: {output_path} ({file_size} bytes)")
                print("✅ Subtitle-to-audio PASSED")

                # Cleanup
                os.remove(output_path)
                return True
            else:
                print("❌ Subtitle-to-audio FAILED - audio file not created")
                return False

        finally:
            os.unlink(temp_srt_path)

    except Exception as e:
        print(f"❌ Subtitle-to-audio error: {e}")
        return False

def test_video_translation():
    """Test complete video translation"""
    print("\n" + "="*60)
    print("TESTING: Video Translation")
    print("="*60)

    video_path = "test_samples/sample_30s_en.mp4"

    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return False

    try:
        # Initialize pipeline
        config = PipelineConfig(
            chunk_size=10,  # Smaller chunks for faster testing
            max_workers=1,
            use_gpu=False,
            generate_subtitles=False,  # Disable for speed
            parallel_processing=False
        )

        pipeline = VideoTranslationPipeline(config)
        print("✓ VideoTranslationPipeline initialized")

        # Load models
        model_loaded = pipeline.load_models("en", "es")
        if model_loaded:
            print("✓ Models loaded successfully")
        else:
            print("⚠ Models not loaded (may still work with fallbacks)")

        # Process video
        start_time = time.time()
        result = pipeline.process_video_fast(video_path, "es")
        end_time = time.time()

        print(".1f"        print(f"  Success: {result['success']}")

        if result['success']:
            print("✅ Video translation PASSED")

            # Check output
            output_video = result.get('output_video')
            if output_video and os.path.exists(output_video):
                file_size = os.path.getsize(output_video)
                print(f"✅ Translated video created: {output_video} ({file_size} bytes)")
            else:
                print("⚠ No output video file found")

            print(f"  Target language: {result.get('target_language', 'es')}")
            print(f"  Chunks processed: {result.get('chunks_processed', 0)}")
            print(f"  Total chunks: {result.get('total_chunks', 0)}")

            # Check subtitles
            if result.get('subtitle_files'):
                print(f"  Subtitle files: {list(result['subtitle_files'].keys())}")

            return True
        else:
            print(f"❌ Video translation FAILED: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"❌ Video translation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 OCTAVIA TRANSLATION FUNCTION TESTS")
    print("="*60)
    print("Testing all core functions using sample_30s_en.mp4")
    print("="*60)

    results = []

    # Test 1: Subtitle Generation
    results.append(("Subtitle Generation", test_subtitle_generation()))

    # Test 2: Subtitle Translation
    results.append(("Subtitle Translation", test_subtitle_translation()))

    # Test 3: Subtitle to Audio
    results.append(("Subtitle to Audio", test_subtitle_to_audio()))

    # Test 4: Video Translation
    results.append(("Video Translation", test_video_translation()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Octavia translation functions are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
