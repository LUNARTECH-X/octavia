#!/usr/bin/env python3
"""
Simple test for demo video translation with authentication
"""

import requests
import time
import jwt
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

# Create a demo JWT token for testing
def create_demo_token():
    """Create a JWT token for demo user"""
    secret = "123456789abcdef123456789abcdef123456789abcdef123456789abcdef1234"  # From .env file
    payload = {
        "sub": "550e8400-e29b-41d4-a716-446655440000",  # Valid UUID format for demo user
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token

print("Testing backend server...")
response = requests.get(f"{API_BASE}/docs")
print(f"Backend status: {response.status_code}")

print("\nCreating demo authentication token...")
demo_token = create_demo_token()
print(f"Demo token created: {demo_token[:50]}...")

headers = {"Authorization": f"Bearer {demo_token}"}

print("\nTesting video translation API with authentication...")
test_video_path = "backend/test_samples/sample_30s_en.mp4"

# Test with authentication
with open(test_video_path, "rb") as f:
    files = {"file": ("sample_30s_en.mp4", f, "video/mp4")}
    data = {"target_language": "es"}

    response = requests.post(f"{API_BASE}/api/translate/video", files=files, data=data, headers=headers, timeout=30)
    print(f"Translation start response: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")

        if result.get("success"):
            job_id = result["job_id"]
            print(f"✅ Job started: {job_id}")

            # Check status a few times
            for i in range(10):  # Check more times for video processing
                time.sleep(5)
                status_response = requests.get(f"{API_BASE}/api/jobs/{job_id}/status", headers=headers)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data.get("success") and "data" in status_data:
                        job_info = status_data["data"]
                        status = job_info.get("status", "unknown")
                        progress = job_info.get("progress", 0)
                        message = job_info.get("status_message", "")
                        print(f"📊 Status check {i+1}: {status} | Progress: {progress}% | {message}")

                        if status == "completed":
                            print("🎉 Translation completed!")
                            download_url = job_info.get("download_url")
                            if download_url:
                                print(f"⬇️ Download URL: {download_url}")

                                # Try to download
                                download_response = requests.get(f"{API_BASE}{download_url}", headers=headers)
                                if download_response.status_code == 200:
                                    print(f"✅ Download successful! File size: {len(download_response.content)} bytes")
                                else:
                                    print(f"❌ Download failed: {download_response.status_code}")
                            break
                        elif status == "failed":
                            print(f"❌ Translation failed: {job_info.get('error', 'Unknown error')}")
                            break
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
        else:
            print("❌ Translation failed to start")
    else:
        print(f"❌ Request failed: {response.status_code} - {response.text}")

print("\nTest completed!")
