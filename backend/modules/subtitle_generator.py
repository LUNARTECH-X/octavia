"""
Enhanced Subtitle Generator with bilingual support
Module for generating subtitles from video/audio using Whisper
"""
import os
import json
import whisper
import ffmpeg
from datetime import timedelta
import webvtt
from typing import Dict, List, Optional

class SubtitleGenerator:
    # Class-level model cache to avoid reloading
    _model_cache = {}

    def __init__(self, model_size="tiny"):  # Changed default to tiny for speed
        self.model_size = model_size
        self.supported_formats = ["srt", "vtt", "ass", "ssa"]

        # Load model with caching
        if model_size not in self._model_cache:
            self._model_cache[model_size] = whisper.load_model(model_size)
        self.model = self._model_cache[model_size]
    
    def extract_audio(self, video_path, audio_path):
        """Extract audio from video for processing"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Use ffmpeg to extract high-quality audio
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                audio_path, 
                acodec='pcm_s16le', 
                ac=1, 
                ar='16000',
                loglevel='error'
            )
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            return audio_path
        except Exception as e:
            raise Exception(f"Audio extraction failed: {str(e)}")
    
    def transcribe_with_timestamps(self, audio_path, language=None):
        """Transcribe audio with precise timestamps and confidence scores"""
        try:
            result = self.model.transcribe(
                audio_path,
                word_timestamps=False,  # Disabled for speed - we don't need word-level timestamps
                language=language,
                task="transcribe",
                temperature=0.0,
                best_of=1,  # Reduced from 5 for speed
                beam_size=1,  # Reduced from 5 for speed
                condition_on_previous_text=True,  # Enabled for better coherence
                no_speech_threshold=0.6,  # More aggressive silence detection
                logprob_threshold=-1.0,  # Allow lower confidence segments
                compression_ratio_threshold=2.4  # Allow more compression
            )
            
            # Process segments for better accuracy
            processed_segments = []
            for segment in result["segments"]:
                if segment["text"].strip():
                    # Calculate confidence
                    avg_no_speech_prob = segment.get("no_speech_prob", 0.5)
                    confidence = 1.0 - avg_no_speech_prob if avg_no_speech_prob else 0.8
                    
                    processed_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "confidence": confidence,
                        "words": segment.get("words", [])
                    })
            
            return {
                "text": result["text"].strip(),
                "segments": processed_segments,
                "language": result.get("language", language or "en"),
                "success": True
            }
            
        except Exception as e:
            return {
                "text": "",
                "segments": [],
                "language": language or "en",
                "success": False,
                "error": str(e)
            }
    
    def format_to_srt(self, segments, include_confidence=False):
        """Convert whisper segments to SRT format"""
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            # Format timestamps
            start_time = timedelta(seconds=segment["start"])
            end_time = timedelta(seconds=segment["end"])
            
            start_str = self._format_timestamp(start_time)
            end_str = self._format_timestamp(end_time)
            
            # Format text
            text = segment["text"].strip()
            
            # Add confidence if requested
            if include_confidence and "confidence" in segment:
                text = f"{text} [{segment['confidence']:.1%}]"
            
            # Add to SRT
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries
        
        return "\n".join(srt_lines)
    
    def format_to_vtt(self, segments):
        """Convert whisper segments to VTT format"""
        vtt_lines = ["WEBVTT", ""]  # VTT header
        
        for i, segment in enumerate(segments, 1):
            # Format timestamps
            start_time = timedelta(seconds=segment["start"])
            end_time = timedelta(seconds=segment["end"])
            
            start_str = self._format_timestamp_vtt(start_time)
            end_str = self._format_timestamp_vtt(end_time)
            
            # Format text
            text = segment["text"].strip()
            
            # Add to VTT
            vtt_lines.append(f"{start_str} --> {end_str}")
            vtt_lines.append(text)
            vtt_lines.append("")  # Empty line
        
        return "\n".join(vtt_lines)
    
    def format_to_bilingual_srt(self, original_segments, translated_segments, original_lang="en", target_lang="de"):
        """Create bilingual subtitles with original and translated text"""
        srt_lines = []
        
        # Determine which segments to use (use the longer list)
        segments_to_use = original_segments if len(original_segments) >= len(translated_segments) else translated_segments
        
        for i, segment in enumerate(segments_to_use, 1):
            # Get timestamps
            start_time = timedelta(seconds=segment.get("start", segment.get("original_start", 0)))
            end_time = timedelta(seconds=segment.get("end", segment.get("original_end", segment.get("start", 0) + 5)))
            
            start_str = self._format_timestamp(start_time)
            end_str = self._format_timestamp(end_time)
            
            # Get texts
            original_text = ""
            translated_text = ""
            
            if i - 1 < len(original_segments):
                original_text = original_segments[i - 1].get("text", "").strip()
            
            if i - 1 < len(translated_segments):
                translated_text = translated_segments[i - 1].get("translated_text", "").strip()
                if not translated_text:
                    translated_text = translated_segments[i - 1].get("text", "").strip()
            
            # Format bilingual text
            if original_text and translated_text:
                bilingual_text = f"[{original_lang.upper()}] {original_text}\n[{target_lang.upper()}] {translated_text}"
            elif original_text:
                bilingual_text = f"[{original_lang.upper()}] {original_text}"
            elif translated_text:
                bilingual_text = f"[{target_lang.upper()}] {translated_text}"
            else:
                continue
            
            # Add to SRT
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(bilingual_text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def _format_timestamp(self, td):
        """Format timedelta to SRT timestamp (HH:MM:SS,mmm)"""
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _format_timestamp_vtt(self, td):
        """Format timedelta to VTT timestamp (HH:MM:SS.mmm)"""
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def generate_subtitles_for_translation(self, video_path, original_segments, translated_segments, 
                                          original_lang, target_lang, output_base):
        """Generate bilingual subtitles for translated video"""
        try:
            # Generate bilingual SRT
            bilingual_srt = self.format_to_bilingual_srt(
                original_segments, translated_segments, original_lang, target_lang
            )
            
            # Save bilingual subtitles
            bilingual_path = f"{output_base}_bilingual.srt"
            with open(bilingual_path, 'w', encoding='utf-8') as f:
                f.write(bilingual_srt)
            
            # Generate separate subtitles
            original_srt = self.format_to_srt(original_segments)
            original_path = f"{output_base}_{original_lang}.srt"
            with open(original_path, 'w', encoding='utf-8') as f:
                f.write(original_srt)
            
            translated_srt = self.format_to_srt(translated_segments)
            translated_path = f"{output_base}_{target_lang}.srt"
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_srt)
            
            # Generate VTT versions (optional)
            vtt_path = f"{output_base}_{target_lang}.vtt"
            vtt_content = self.format_to_vtt(translated_segments)
            with open(vtt_path, 'w', encoding='utf-8') as f:
                f.write(vtt_content)
            
            return {
                "bilingual": bilingual_path,
                "original": original_path,
                "translated": translated_path,
                "vtt": vtt_path,
                "success": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_file(self, file_path, output_format="srt", language=None, generate_all=False):
        """Main processing function with enhanced options"""
        # Check if file is video or audio
        is_video = any(file_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
        
        temp_audio = None
        try:
            if is_video:
                # Extract audio first
                temp_dir = os.path.dirname(file_path) or "."
                audio_path = os.path.join(temp_dir, f"temp_audio_{os.path.basename(file_path)}.wav")
                self.extract_audio(file_path, audio_path)
                media_path = audio_path
                temp_audio = audio_path
            else:
                media_path = file_path
            
            # Transcribe
            transcription = self.transcribe_with_timestamps(media_path, language)
            
            if not transcription["success"]:
                raise ValueError(f"Transcription failed: {transcription.get('error', 'Unknown error')}")
            
            # Generate output based on format
            output_files = {}
            
            if output_format.lower() == "srt" or generate_all:
                srt_content = self.format_to_srt(transcription["segments"])
                srt_file = "subtitles.srt"
                with open(srt_file, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                output_files["srt"] = srt_file
            
            if output_format.lower() == "vtt" or generate_all:
                vtt_content = self.format_to_vtt(transcription["segments"])
                vtt_file = "subtitles.vtt"
                with open(vtt_file, "w", encoding="utf-8") as f:
                    f.write(vtt_content)
                output_files["vtt"] = vtt_file
            
            # Cleanup temp audio if created
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return {
                "success": True,
                "output_files": output_files,
                "text": transcription["text"],
                "language": transcription["language"],
                "segment_count": len(transcription["segments"]),
                "segments": transcription["segments"]
            }
            
        except Exception as e:
            # Cleanup on error
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except:
                    pass
            
            return {
                "success": False,
                "error": str(e),
                "output_files": {},
                "text": "",
                "language": language or "en",
                "segment_count": 0,
                "segments": []
            }

# Example usage
if __name__ == "__main__":
    generator = SubtitleGenerator()
    
    # Test with a file
    result = generator.process_file("test_video.mp4", output_format="srt")
    
    if result["success"]:
        print(f"Generated {result['segment_count']} subtitle segments")
        print(f"Language: {result['language']}")
        for format_name, file_path in result["output_files"].items():
            print(f"{format_name.upper()}: {file_path}")
    else:
        print(f"Error: {result['error']}")
