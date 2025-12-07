#!/usr/bin/env python3
"""
Command Line Interface for Octavia Video Translator
"""
import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime

from modules.pipeline import VideoTranslationPipeline, PipelineConfig
from modules.instrumentation import MetricsCollector
from modules.subtitle_generator import SubtitleGenerator
from modules.audio_translator import AudioTranslator

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def translate_video(args):
    """Translate a video file"""
    print(f"Starting video translation: {args.input}")
    
    # Load configuration
    config_dict = load_config(args.config)
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        chunk_size=config_dict.get('processing', {}).get('default_chunk_size', 30),
        max_chunk_size=config_dict.get('processing', {}).get('max_chunk_size', 120),
        condensation_ratio=config_dict.get('processing', {}).get('max_condensation_ratio', 1.2),
        timing_tolerance_ms=config_dict.get('processing', {}).get('max_duration_diff_ms', 200)
    )
    
    # Initialize pipeline
    pipeline = VideoTranslationPipeline(pipeline_config)
    
    try:
        # Process video
        result = pipeline.process_video(args.input, args.target_lang)
        
        # Output result
        print(f"\n Translation completed!")
        print(f"   Output: {result['output_path']}")
        print(f"   Duration match: {result['duration_match_within_tolerance']}")
        print(f"   Successful chunks: {result['successful_chunks']}/{result['total_chunks']}")
        print(f"   Total time: {result['total_time_seconds']:.1f}s")
        
        # Save result summary
        summary_path = f"translation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "input": args.input,
                "output": result["output_path"],
                "target_language": args.target_lang,
                "success": result["success"],
                "metrics": {
                    "total_chunks": result["total_chunks"],
                    "successful_chunks": result["successful_chunks"],
                    "avg_duration_diff_ms": result["avg_duration_diff_ms"],
                    "avg_condensation_ratio": result["avg_condensation_ratio"]
                },
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        print(f"   Summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        print(f" Translation failed: {str(e)}")
        return 1

def generate_subtitles(args):
    """Generate subtitles from video/audio"""
    print(f" Generating subtitles: {args.input}")
    
    generator = SubtitleGenerator()
    
    try:
        result = generator.process_file(
            args.input,
            args.format,
            args.language
        )
        
        print(f"\n Subtitles generated!")
        print(f"   Format: {args.format.upper()}")
        print(f"   Segments: {result['segment_count']}")
        print(f"   Language: {result['language']}")
        print(f"   Output: {result['output_file']}")
        
        return 0
        
    except Exception as e:
        print(f" Subtitle generation failed: {str(e)}")
        return 1

def translate_audio(args):
    """Translate audio file"""
    print(f" Translating audio: {args.input}")
    
    translator = AudioTranslator(args.source_lang, args.target_lang)
    
    try:
        result = translator.process_audio(args.input)
        
        print(f"\ Audio translation completed!")
        print(f"   Source: {result['source_lang']} → Target: {result['target_lang']}")
        print(f"   Duration match: {result['duration_match_percent']:.1f}%")
        print(f"   Speed adjustment: {result['speed_adjustment']:.2f}x")
        print(f"   Output: {result['output_path']}")
        
        return 0
        
    except Exception as e:
        print(f" Audio translation failed: {str(e)}")
        return 1

def test_integration(args):
    """Run integration test with 20-30s sample"""
    print(" Running integration test...")
    
    # Check for test sample
    test_video = "test_samples/sample_30s_en.mp4"
    if not os.path.exists(test_video):
        print(f" Test sample not found: {test_video}")
        print("   Please create a 20-30s test video in test_samples/")
        return 1
    
    # Run pipeline
    pipeline = VideoTranslationPipeline()
    
    try:
        result = pipeline.process_video(test_video, "es")
        
        # Validate results
        original_duration = 30.0  # Should match test sample
        output_duration = result.get('total_time_seconds', 0)
        duration_diff = abs(output_duration - original_duration)
        
        print(f"\n Integration Test Results:")
        print(f"   Input: {test_video}")
        print(f"   Output: {result.get('output_path', 'N/A')}")
        print(f"   Original duration: {original_duration:.1f}s")
        print(f"   Processing time: {output_duration:.1f}s")
        print(f"   Duration difference: {duration_diff:.3f}s")
        
        # Check technical requirements
        requirements_met = []
        
        # AT-1: Duration match within 1 frame
        if duration_diff <= 0.1:  # 100ms tolerance
            requirements_met.append(" Duration match within tolerance")
        else:
            requirements_met.append(" Duration match FAILED")
        
        # AT-2: Check condensation ratio
        avg_condensation = result.get('avg_condensation_ratio', 1.0)
        if avg_condensation <= 1.2:
            requirements_met.append(" Condensation within 1.2x limit")
        else:
            requirements_met.append(" Condensation EXCEEDS limit")
        
        # AT-4: Preview generation
        preview_path = "artifacts/preview.mp4"
        if os.path.exists(preview_path):
            requirements_met.append(" Preview generated")
        else:
            requirements_met.append(" Preview NOT generated")
        
        print(f"\n Technical Requirements:")
        for req in requirements_met:
            print(f"   {req}")
        
        # Save test report
        report = {
            "test_name": "integration_test",
            "timestamp": datetime.utcnow().isoformat(),
            "input_file": test_video,
            "requirements": requirements_met,
            "metrics": {
                "duration_diff_seconds": duration_diff,
                "avg_condensation_ratio": avg_condensation,
                "successful_chunks": result.get('successful_chunks', 0),
                "total_chunks": result.get('total_chunks', 0)
            },
            "passed": all("✅" in req for req in requirements_met)
        }
        
        report_path = "artifacts/integration_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n Test report saved to: {report_path}")
        
        return 0 if report["passed"] else 1
        
    except Exception as e:
        print(f" Integration test failed: {str(e)}")
        return 1

def show_metrics(args):
    """Show metrics from logs"""
    metrics_file = "artifacts/logs.jsonl"
    
    if not os.path.exists(metrics_file):
        print(f" Metrics file not found: {metrics_file}")
        return 1
    
    # Collect metrics
    collector = MetricsCollector()
    report = collector.generate_summary_report()
    
    print("\n Processing Metrics Summary:")
    print(f"   Hardware: {report['hardware']['cpu_count']} CPUs, "
          f"{report['hardware']['total_ram_gb']}GB RAM")
    
    if report['hardware']['gpu_available']:
        print(f"   GPU: {report['hardware']['gpu_name']} "
              f"({report['hardware']['gpu_memory_gb']}GB)")
    
    print(f"\n   Processing Stages: {report['processing_summary']['total_stages']}")
    print(f"   Success Rate: {report['processing_summary']['success_rate']}%")
    
    print(f"\n   Average Durations (ms):")
    for stage, duration in report['processing_summary']['average_durations_ms'].items():
        print(f"     {stage}: {duration}ms")
    
    print(f"\n   Quality Metrics:")
    print(f"     Condensation Events: {report['quality_metrics']['condensation_events']}")
    print(f"     Avg Condensation Ratio: {report['quality_metrics']['avg_condensation_ratio']}")
    print(f"     Within Tolerance: {report['quality_metrics']['within_tolerance_percentage']}%")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Octavia Video Translator - Technical Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video --input sample.mp4 --target es
  %(prog)s subtitles --input video.mp4 --format srt
  %(prog)s audio --input audio.wav --source en --target es
  %(prog)s test-integration
  %(prog)s metrics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Video translation command
    video_parser = subparsers.add_parser('video', help='Translate video file')
    video_parser.add_argument('--input', '-i', required=True, help='Input video file')
    video_parser.add_argument('--target-lang', '-t', default='es', help='Target language (default: es)')
    video_parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file')
    video_parser.add_argument('--output', '-o', help='Output file (default: auto-generated)')
    
    # Subtitle generation command
    subtitle_parser = subparsers.add_parser('subtitles', help='Generate subtitles')
    subtitle_parser.add_argument('--input', '-i', required=True, help='Input video/audio file')
    subtitle_parser.add_argument('--format', '-f', default='srt', choices=['srt', 'vtt'], 
                                help='Subtitle format')
    subtitle_parser.add_argument('--language', '-l', default='auto', help='Audio language')
    
    # Audio translation command
    audio_parser = subparsers.add_parser('audio', help='Translate audio file')
    audio_parser.add_argument('--input', '-i', required=True, help='Input audio file')
    audio_parser.add_argument('--source-lang', '-s', default='en', help='Source language')
    audio_parser.add_argument('--target-lang', '-t', default='es', help='Target language')
    
    # Integration test command
    test_parser = subparsers.add_parser('test-integration', help='Run integration test')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show processing metrics')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'video':
        return translate_video(args)
    elif args.command == 'subtitles':
        return generate_subtitles(args)
    elif args.command == 'audio':
        return translate_audio(args)
    elif args.command == 'test-integration':
        return test_integration(args)
    elif args.command == 'metrics':
        return show_metrics(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
