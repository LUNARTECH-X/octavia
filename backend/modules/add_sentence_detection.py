with open('audio_translator.py', 'r', encoding='utf-8') as f:
    content = f.read()

marker = '''    def _translate_with_marian(self, text: str) -> Optional[str]:
        """Attempt translation using MarianMT pipeline with sentence splitting for CJK languages"""'''

new_methods = '''    def _detect_sentences(self, text: str) -> List[str]:
        """Detect sentence boundaries in text (supports CJK and English)"""
        import re
        
        source_lang = getattr(self.config, 'source_lang', 'en')
        is_cjk = source_lang in ['zh', 'ja', 'ko']
        
        if is_cjk:
            sentences = re.split(r'(?<=[.!?。！？])\\s*', text)
        else:
            sentences = re.split(r'(?<=[.!?])\\s+', text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _translate_by_sentences(self, text: str) -> str:
        """Translate text sentence by sentence for better quality and audio continuity"""
        try:
            if not self.translation_pipeline:
                return None
            
            sentences = self._detect_sentences(text)
            
            if len(sentences) <= 1:
                result = self.translation_pipeline(text, max_length=1024, num_beams=1)
                return result[0]['translation_text']
            
            logger.info(f"Translating {len(sentences)} sentences individually")
            
            translated_sentences = []
            for i, sent in enumerate(sentences):
                if len(sent) < 2:
                    translated_sentences.append('')
                    continue
                    
                try:
                    result = self.translation_pipeline(sent, max_length=512, num_beams=1)
                    translated = result[0]['translation_text'].strip()
                    translated_sentences.append(translated)
                except Exception as e:
                    logger.warning(f"Failed to translate sentence {i}: {e}")
                    translated_sentences.append(sent)
            
            translated_text = ' '.join(translated_sentences)
            translated_text = re.sub(r'\\s+', ' ', translated_text).strip()
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Sentence-by-sentence translation failed: {e}")
            return None
    
    def _translate_with_marian(self, text: str) -> Optional[str]:
        """Attempt translation using MarianMT pipeline with sentence splitting for CJK languages"""'''

if marker in content:
    content = content.replace(marker, new_methods)
    with open('audio_translator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS: Added sentence detection methods')
else:
    print('ERROR: Marker not found')
