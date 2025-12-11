import os
import pytest
from modules.subtitle_generator import SubtitleGenerator

def test_subtitle_translation_basic():
    # Use a sample video or audio file for testing
    sample_file = os.path.join('test_samples', 'sample_30s_en.mp4')
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"
    generator = SubtitleGenerator()
    result = generator.process_file(sample_file, output_format='srt', language='es')
    assert result.get('success'), f"Subtitle translation failed: {result.get('error')}"
    assert 'output_files' in result and result['output_files'].get('srt'), "No SRT output file generated"
    srt_file = result['output_files']['srt']
    assert os.path.exists(srt_file), f"SRT file not found: {srt_file}"
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    assert len(content.strip()) > 0, "SRT file is empty"
    print("Subtitle translation test passed. Output file:", srt_file)

if __name__ == "__main__":
    test_subtitle_translation_basic()
