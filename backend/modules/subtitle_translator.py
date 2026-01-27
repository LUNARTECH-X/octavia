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
        target_lang: str = 'en',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Translate subtitle file to target language
        """
        # Validate file format
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported: {self.supported_formats}")

        # Convert language names to codes and validate
        source_lang_lower = source_lang.lower()
        target_lang_lower = target_lang.lower()

        # Validate target language (must be supported)
        if target_lang_lower not in self.supported_languages:
            supported_list = ', '.join(sorted(set([k for k in self.supported_languages.keys() if len(k) <= 10])))
            raise ValueError(f"Unsupported target language: '{target_lang}'. Supported languages: {supported_list}")

        # Get language codes
        source_code = self.supported_languages.get(source_lang_lower, 'auto')
        target_code = self.supported_languages[target_lang_lower]

        # Load subtitles with better error handling
        print(f"DEBUG: Loading subtitle file: {file_path}, format: {file_ext}")
        if progress_callback:
            progress_callback(5, f"Loading subtitle file...")
        
        if file_ext == '.srt':
            # Try different encodings
            subs = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    subs = pysrt.open(file_path, encoding=encoding)
                    if len(subs) > 0:
                        print(f"DEBUG: Successfully loaded {len(subs)} subtitles with {encoding} encoding")
                        break
                except Exception as e:
                    print(f"DEBUG: Failed to load with {encoding}: {e}")
                    continue
            
            if subs is None or len(subs) == 0:
                # Try manual parsing as fallback
                print("DEBUG: pysrt failed, trying manual SRT parsing")
                subs = self._manual_parse_srt(file_path)
                
        elif file_ext == '.vtt':
            # Convert VTT to SRT or handle differently
            subs = self._load_vtt(file_path)
        else:
            # For ASS/SSA, extract text content
            subs = self._load_ass(file_path)

        print(f"DEBUG: Loaded {len(subs) if subs else 0} subtitle segments")
        if progress_callback:
            progress_callback(10, f"Loaded {len(subs) if subs else 0} segments. Starting translation...")
        
        if not subs or len(subs) == 0:
            # Read file content for debugging
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                print(f"DEBUG: File content preview: {content[:200]}")
            except:
                pass
            raise ValueError(f"No subtitle segments found in file. Please check the file format.")

        # Translate each subtitle
        translator = GoogleTranslator(source=source_code, target=target_code)

        translated_subs = []
        total_segments = len(subs)
        for i, sub in enumerate(subs):
            try:
                if sub.text and sub.text.strip():
                    translated_text = translator.translate(sub.text)
                    sub.text = translated_text
                translated_subs.append(sub)
                
                # Update progress every 5 segments or at the end
                if i % 5 == 0 or i == total_segments - 1:
                    print(f"DEBUG: Translated {i+1}/{total_segments} segments")
                    if progress_callback:
                        # Map 10% to 90% range for translation
                        pct = 10 + int((i + 1) / total_segments * 80)
                        progress_callback(pct, f"Translated {i+1}/{total_segments} segments...")
            except Exception as e:
                print(f"Error translating line {i}: {e}")
                translated_subs.append(sub)  # Keep original if translation fails

        # Save translated subtitles
        output_path = self._save_translated(file_path, translated_subs, target_lang)
        print(f"DEBUG: Saved translated subtitles to {output_path}")
        if os.path.exists(output_path):
            print(f"DEBUG: Verified output file exists and has size {os.path.getsize(output_path)} bytes")
        else:
            print(f"DEBUG: ERROR - Output file does not exist at {output_path}")

        return {
            "output_path": output_path,
            "segment_count": len(translated_subs),
            "source_language": source_lang,
            "target_language": target_lang
        }
    
    def _manual_parse_srt(self, file_path: str):
        """Manually parse SRT file when pysrt fails"""
        subs = []
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except:
                    continue
            
            if not content:
                print("DEBUG: Could not read file with any encoding")
                return subs
            
            # Split into blocks (separated by empty lines)
            blocks = content.strip().split('\n\n')
            print(f"DEBUG: Found {len(blocks)} blocks in SRT file")
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    try:
                        # First line is index
                        index = int(lines[0].strip())
                        # Second line is timestamp
                        timestamp_line = lines[1].strip()
                        if ' --> ' in timestamp_line:
                            start_str, end_str = timestamp_line.split(' --> ')
                            # Third line onwards is text
                            text = '\n'.join(lines[2:])
                            
                            subs.append(pysrt.SubRipItem(
                                index=index,
                                start=pysrt.SubRipTime.from_string(start_str.strip()),
                                end=pysrt.SubRipTime.from_string(end_str.strip()),
                                text=text
                            ))
                    except Exception as e:
                        print(f"DEBUG: Failed to parse block: {e}")
                        continue
            
            print(f"DEBUG: Manual parsing found {len(subs)} subtitles")
        except Exception as e:
            print(f"DEBUG: Manual SRT parsing failed: {e}")
        
        return subs
    
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
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in subtitles:
                    f.write(f"{sub.index}\n")
                    # Use formatted time to be safe
                    start_time = str(sub.start).replace('.', ',')
                    end_time = str(sub.end).replace('.', ',')
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{sub.text}\n\n")
            print(f"DEBUG: _save_translated successfully wrote {len(subtitles)} segments to {output_path}")
        except Exception as e:
            print(f"DEBUG: ERROR in _save_translated: {e}")
            raise
        
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
