import logging
import re

class SecretMasker(logging.Filter):
    """
    Logging filter that masks sensitive information like API keys and tokens.
    """
    def __init__(self, patterns=None):
        super().__init__()
        # Common patterns for secrets
        self.patterns = patterns or [
            r'sk-[a-zA-Z0-9]{32,}',           # OpenAI/OpenRouter-like keys
            r'bearer\s+[a-zA-Z0-9._-]+',      # Bearer tokens
            r'api[-_]?key\s*[:=]\s*["\']?[a-zA-Z0-9-_]{20,}', # API Keys
            r'token\s*[:=]\s*["\']?[a-zA-Z0-9-_]{20,}',       # Tokens
            r'password\s*[:=]\s*["\']?[^"\', ]+["\']?',       # Passwords
        ]

    def filter(self, record):
        if isinstance(record.msg, str):
            for pattern in self.patterns:
                # Mask the secret while preserving some context if possible
                # But for safety, we'll replace the whole match
                record.msg = re.sub(pattern, lambda m: self._mask_match(m.group(0)), record.msg, flags=re.IGNORECASE)
        return True

    def _mask_match(self, match_str):
        """Helper to mask sensitive parts of a match while keeping structure if helpful"""
        if match_str.lower().startswith("bearer"):
            return "Bearer ********"
        if ":" in match_str:
            key, val = match_str.split(":", 1)
            return f"{key}: ********"
        if "=" in match_str:
            key, val = match_str.split("=", 1)
            return f"{key}=********"
        return "********"
