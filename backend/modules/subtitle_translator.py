import os
from typing import Dict, Any, Optional
import pysrt
import tempfile
from deep_translator import GoogleTranslator
from pathlib import Path

class SubtitleTranslator:
    def __init__(self):
        self.supported_formats = ['.srt', '.vtt', '.ass']
        self.supported_languages = {
            # Language names to codes
            'english': 'en', 'spanish': 'es', 'french': 'fr',
            'german': 'de', 'italian': 'it', 'portuguese': 'pt',
            'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
            'chinese': 'zh-cn', 'arabic': 'ar', 'hindi': 'hi',
            # Also support direct language codes
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
            'pt': 'pt', 'ru': 'ru', 'ja': 'ja', 'ko': 'ko', 'zh': 'zh-cn',
            'ar': 'ar', 'hi': 'hi'
        }
    
    def translate_subtitles(
        self,
        file_path: str,
        source_lang: str = 'auto',
        target_lang: str = 'en'
    ) -> Dict[str, Any]:
        """
        Translate subtitle file to target language
        """
        # Validate file format
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported: {self.supported_formats}")

        # Convert language names to codes
        source_code = self.supported_languages.get(source_lang.lower(), 'auto')
        target_code = self.supported_languages.get(target_lang.lower(), 'en')

        # Load subtitles
        if file_ext == '.srt':
            subs = pysrt.open(file_path)
        elif file_ext == '.vtt':
            # Convert VTT to SRT or handle differently
            subs = self._load_vtt(file_path)
        else:
            # For ASS/SSA, extract text content
            subs = self._load_ass(file_path)

        # Translate each subtitle
        translator = GoogleTranslator(source=source_code, target=target_code)

        translated_subs = []
        for sub in subs:
            try:
                translated_text = translator.translate(sub.text)
                sub.text = translated_text
                translated_subs.append(sub)
            except Exception as e:
                print(f"Error translating line: {e}")
                translated_subs.append(sub)  # Keep original if translation fails

        # Save translated subtitles
        output_path = self._save_translated(file_path, translated_subs, target_lang)

        return {
            "output_path": output_path,
            "segment_count": len(translated_subs),
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    def _load_vtt(self, file_path: str):
        """Load VTT file and convert to SRT format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple VTT to SRT conversion
        content = content.replace('WEBVTT', '').strip()
        lines = content.split('\n')
        
        subs = []
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                timecode = lines[i]
                text = lines[i + 1]
                subs.append(pysrt.SubRipItem(
                    index=len(subs) + 1,
                    start=pysrt.SubRipTime.from_string(timecode.split(' --> ')[0].replace('.', ',')),
                    end=pysrt.SubRipTime.from_string(timecode.split(' --> ')[1].replace('.', ',')),
                    text=text
                ))
        
        return subs
    
    def _load_ass(self, file_path: str):
        """Extract dialogue from ASS/SSA file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        subs = []
        in_events = False
        for line in lines:
            line = line.strip()
            if line == '[Events]':
                in_events = True
                continue
            if in_events and line.startswith('Dialogue:'):
                parts = line.split(',', maxsplit=9)
                if len(parts) >= 10:
                    start_time = parts[1]
                    end_time = parts[2]
                    text = parts[9]
                    
                    subs.append(pysrt.SubRipItem(
                        index=len(subs) + 1,
                        start=pysrt.SubRipTime.from_string(start_time.replace('.', ',')),
                        end=pysrt.SubRipTime.from_string(end_time.replace('.', ',')),
                        text=text
                    ))
        
        return subs
    
    def _save_translated(self, original_path: str, subtitles, target_lang: str) -> str:
        """Save translated subtitles to a new file"""
        original_name = Path(original_path).stem
        output_dir = tempfile.gettempdir()
        output_path = os.path.join(
            output_dir, 
            f"{original_name}_{target_lang}.srt"
        )
        
        # Create new SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in subtitles:
                f.write(f"{sub.index}\n")
                f.write(f"{str(sub.start).replace('.', ',')} --> {str(sub.end).replace('.', ',')}\n")
                f.write(f"{sub.text}\n\n")
        
        return output_path

# FastAPI endpoint example
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse

router = APIRouter()
translator = SubtitleTranslator()

@router.post("/translate/subtitles")
async def translate_subtitles(
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en")
):
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    input_path = os.path.join(temp_dir, file.filename)
    
    with open(input_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    # Translate
    try:
        output_path = translator.translate_subtitles(
            input_path, 
            source_language, 
            target_language
        )
        
        # Clean up input file
        os.remove(input_path)
        
        return FileResponse(
            output_path,
            filename=f"translated_{file.filename}",
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise e
