import logging
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from services.logging_utils import SecretMasker

def test_masking():
    logger = logging.getLogger("test_masker")
    logger.setLevel(logging.INFO)
    
    # Add a file handler
    test_log = "test_sanitization.log"
    handler = logging.FileHandler(test_log)
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add SecretMasker
    masker = SecretMasker()
    handler.addFilter(masker)
    logger.addHandler(handler)
    
    # Log some secrets
    secrets_to_test = [
        "OpenRouter Key: sk-or-v1-abcdef1234567890abcdef1234567890",
        "OpenAI Key: sk-1234567890abcdef1234567890abcdef",
        "Authorization: Bearer my-secret-token-123",
        "db_password: 'super_secret_pwd_123'",
        "api_key=xyz123abc456def789ghi",
    ]
    
    print("Logging secrets...")
    for secret in secrets_to_test:
        logger.info(secret)
    
    # Read back and verify
    print("\nVerifying logs:")
    with open(test_log, "r") as f:
        content = f.read()
        print(content)
        
        # Check if secrets are masked
        if "sk-" in content and "********" in content:
            print("\n✅ Verification SUCCESS: Secrets are masked.")
        else:
            print("\n❌ Verification FAILED: Secrets might still be visible.")
            
    # Cleanup
    # os.remove(test_log)

if __name__ == "__main__":
    test_masking()
