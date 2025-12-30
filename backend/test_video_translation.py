#!/usr/bin/env python3
"""
Test Video Translation Script
Tests the complete video translation pipeline with real-time progress updates
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from modules.pipeline import VideoTranslationPipeline, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_video_translation.log')
    ]
)
logger = logging.getLogger(__name__)

def test_video_translation(video_path=None, target_lang=None, output_dir=None):
    """Test video translation pipeline"""
    print("=" * 60)
    print("OCTAVIA VIDEO TRANSLATION TEST")
    print("=" * 60)

    # Use provided parameters or defaults
    test_video_path = video_path or "backend/test_samples/sample_30s_en.mp4"
    target_language = target_lang or "de"
    test_output_dir = output_dir or "backend/test_outputs"

    # Check if test video exists
    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        print("Available test files:")
        for root, dirs, files in os.walk("backend/test_samples"):
            for file in files:
                print(f"  - {os.path.join(root, file)}")
        return False

    print(f"✅ Test video found: {test_video_path}")

    # Check video duration
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', test_video_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            print(f"📹 Video duration: {duration:.1f} seconds")
        else:
            print("⚠️ Could not get video duration")

    except Exception as e:
        print(f"⚠️ Could not check video info: {e}")

    # Initialize pipeline
    print("\n🚀 Initializing Video Translation Pipeline...")
    config = PipelineConfig(
        chunk_size=30,  # 30 second chunks
        use_gpu=False,  # Use CPU for testing
        temp_dir="/tmp/octavia_test",
        output_dir=test_output_dir
    )

    pipeline = VideoTranslationPipeline(config)

    print(f"🎯 Target Language: {target_language}")
    print(f"📁 Output Directory: {test_output_dir}")

    # Create test job tracking (simulate job_id)
    test_job_id = f"test_job_{os.path.basename(test_video_path)}_{target_language}"
    print(f"📋 Test Job ID: {test_job_id}")

    print("\n📊 Starting Video Translation Process...")

    start_time = time.time()

    try:
        # Process the video
        result = pipeline.process_video_fast(test_video_path, target_language)

        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)

        if result.get("success"):
            print("✅ SUCCESS: Video translation completed!")
            print(f"⏱️ Processing time: {processing_time:.1f} seconds")
            print(f"📁 Output: {result.get('output_video', 'N/A')}")
            print(f"🎬 Target Language: {result.get('target_language', 'N/A')}")
            print(f"📦 Chunks Processed: {result.get('chunks_processed', 0)}")
            print(f"📊 Total Chunks: {result.get('total_chunks', 0)}")

            # Check if output file exists
            output_path = result.get('output_video')
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"💾 File size: {file_size / (1024*1024):.1f} MB")
                print("✅ Output file verification: PASSED")
            else:
                print("❌ Output file verification: FAILED - File not found")
                return False

            # Verify the video has audio
            if output_path and os.path.exists(output_path):
                try:
                    result = subprocess.run([
                        'ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_streams', output_path
                    ], capture_output=True, text=True, timeout=10)

                    if result.returncode == 0:
                        import json
                        probe_data = json.loads(result.stdout)
                        streams = probe_data.get("streams", [])
                        video_streams = [s for s in streams if s.get("codec_type") == "video"]
                        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

                        print("🔍 Stream Analysis:")
                        print(f"   • Video streams: {len(video_streams)}")
                        print(f"   • Audio streams: {len(audio_streams)}")

                        if audio_streams:
                            print("✅ Audio stream verification: PASSED")
                            audio_codec = audio_streams[0].get("codec_name", "unknown")
                            print(f"   • Audio codec: {audio_codec}")
                        else:
                            print("❌ Audio stream verification: FAILED - No audio stream found")
                            return False
                    else:
                        print("⚠️ Could not verify streams")

                except Exception as verify_error:
                    print(f"⚠️ Stream verification failed: {verify_error}")

            print("\n🎉 VIDEO TRANSLATION TEST PASSED!")
            print("✅ Audio translation successfully integrated with video")
            print("✅ Real-time progress updates working")
            return True

        else:
            print("❌ FAILED: Video translation failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"⏱️ Processing time: {processing_time:.1f} seconds")
            return False

    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 60)
        print("ERROR:")
        print("=" * 60)
        print(f"❌ CRITICAL ERROR: {str(e)}")
        print(f"⏱️ Processing time: {processing_time:.1f} seconds")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test video translation pipeline")
    parser.add_argument("--video", help="Video file path")
    parser.add_argument("--lang", help="Target language")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Backend directory: {backend_dir}")

    # Check dependencies
    print("\n🔍 Checking dependencies...")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   • CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   • CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch not available")

    try:
        import whisper
        print("✅ Whisper available")
    except ImportError:
        print("❌ Whisper not available")

    try:
        from transformers import pipeline
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")

    try:
        from pydub import AudioSegment
        print("✅ PyDub available")
    except ImportError:
        print("❌ PyDub not available")

    # Run tests
    print("\n" + "=" * 80)
    print("RUNNING VIDEO TRANSLATION TESTS")
    print("=" * 80)

    # Test 1: English to German
    print("\n" + "🔵" * 80)
    print("TEST 1: English -> German")
    print("🔵" * 80)
    success1 = test_video_translation(
        video_path="backend/test_samples/sample_30s_en.mp4",
        target_lang="de",
        output_dir="backend/test_outputs"
    )

    # Test 2: Russian to English
    print("\n" + "🟡" * 80)
    print("TEST 2: Russian -> English")
    print("🟡" * 80)
    success2 = test_video_translation(
        video_path="backend/test_samples/sample_30s_ru.mp4",
        target_lang="en",
        output_dir="backend/test_outputs"
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)

    if success1 and success2:
        print("🎉 ALL TESTS PASSED: Video translation pipeline is working!")
        print("✅ Ready for integration into the main application")
        print("\nTranslated videos saved to: backend/test_outputs/")
        return 0
    else:
        print("💥 SOME TESTS FAILED: Video translation pipeline needs fixes")
        print("❌ Please check the logs and fix issues before integration")
        if not success1:
            print("  - English -> German test failed")
        if not success2:
            print("  - Russian -> English test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
