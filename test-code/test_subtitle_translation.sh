#!/bin/bash

# Test curl command for subtitle translation
# This should work with the current backend configuration

echo "Testing subtitle translation endpoint..."
echo "====================================="

curl -X POST "http://localhost:8000/api/translate/subtitle-file" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN_HERE" \
  -F "file=@test_subtitle.srt" \
  -F "sourceLanguage=en" \
  -F "targetLanguage=es" \
  -v

echo ""
echo "====================================="
echo "If this fails with 422, check the backend logs for detailed error information."
