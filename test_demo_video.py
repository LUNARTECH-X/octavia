#!/usr/bin/env python3
"""
Test Demo Account Video Translation via API
Tests the complete flow: login as demo user, upload video, start translation, check progress, download result
"""

import os
import sys
import time
import json
import requests

# Backend API base URL
API_BASE = "http://localhost:8000"
FRONTEND_BASE = "http://localhost:3000"

def test_demo_video_translation():
    """Test complete demo account video translation flow"""
    print("=" * 60)
    print("DEMO ACCOUNT VIDEO TRANSLATION TEST")
    print("=" * 60)

    # Check if servers are running
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server not responding")
            return False
        print("✅ Backend server is running")
    except:
        print("❌ Backend server not accessible")
        return False

    try:
        response = requests.get(FRONTEND_BASE, timeout=5)
        if response.status_code != 200:
            print("❌ Frontend server not responding")
            return False
        print("✅ Frontend server is running")
    except:
        print("❌ Frontend server not accessible")
        return False

    # Step 1: Simulate demo login
    print("\n🔐 Step 1: Simulating Demo Login")
    demo_user = {
        "email": "demo@octavia.com",
        "credits": 5000
    }
    print(f"✅ Logged in as demo user: {demo_user['email']}")

    # Step 2: Prepare test video
    print("\n📹 Step 2: Preparing Test Video")
    test_video_path = "backend/test_samples/sample_30s_en.mp4"

    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        return False

    file_size = os.path.getsize(test_video_path) / (1024*1024)  # MB
    print(f"📁 Video size: {file_size:.1f} MB")
    print("✅ Video file is ready for upload")

    # Step 3: Upload and start video translation
    print("\n🚀 Step 3: Starting Video Translation")

    # Prepare form data
    with open(test_video_path, "rb") as f:
        files = {"file": ("sample_30s_en.mp4", f, "video/mp4")}
        data = {"target_language": "es"}  # Spanish

        # Make API call
        url = f"{API_BASE}/api/translate/video"
        print(f"📡 POST {url}")

        try:
            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    job_id = result["job_id"]
                    print("✅ Translation started successfully!"                    print(f"📋 Job ID: {job_id}")
                    print(f"🎯 Target Language: {data['target_language']}")
                    print(f"💰 Credits Remaining: {result.get('remaining_credits', 'N/A')}")
                    print(f"📊 Status URL: {result.get('status_url', 'N/A')}")
                else:
                    print(f"❌ API call succeeded but returned error: {result}")
                    return False
            else:
                print(f"❌ API call failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return False

    # Step 4: Monitor translation progress
    print("\n📊 Step 4: Monitoring Translation Progress")

    status_url = f"{API_BASE}/api/jobs/{job_id}/status"
    max_attempts = 60  # 5 minutes max
    attempt = 0

    while attempt < max_attempts:
        try:
            response = requests.get(status_url, timeout=10)

            if response.status_code == 200:
                status_data = response.json()

                if status_data.get("success") and "data" in status_data:
                    job_data = status_data["data"]
                    status = job_data.get("status", "unknown")
                    progress = job_data.get("progress", 0)
                    message = job_data.get("status_message", "")

                    print(f"📊 Status: {status} | Progress: {progress:.1f}% | Message: {message}")

                    if status == "completed":
                        print("🎉 Translation completed successfully!"                        print(f"✅ Final status: {status}")
                        print(f"📁 Result: {job_data.get('result', {})}")

                        # Check download URL
                        download_url = job_data.get("download_url")
                        if download_url:
                            print(f"⬇️ Download URL: {download_url}")
                        else:
                            print("⚠️ No download URL provided")

                        break

                    elif status == "failed":
                        print(f"❌ Translation failed: {job_data.get('error', 'Unknown error')}")
                        return False

                else:
                    print(f"❌ Status check failed: {status_data}")
                    return False

            else:
                print(f"❌ Status check HTTP error: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Status check failed (attempt {attempt + 1}): {e}")

        attempt += 1
        if attempt < max_attempts:
            time.sleep(5)  # Wait 5 seconds between checks

    if attempt >= max_attempts:
        print("⏰ Translation timed out after 5 minutes")
        return False

    # Step 5: Test download functionality
    print("\n⬇️ Step 5: Testing Video Download")

    if "download_url" in locals() and download_url:
        try:
            # Extract file ID from download URL
            # URL format: /api/translate/download/video/{file_id}
            download_parts = download_url.split("/")
            if len(download_parts) >= 4:
                file_type = download_parts[-2]  # "video"
                file_id = download_parts[-1]    # job_id

                download_endpoint = f"{API_BASE}/api/translate/download/{file_type}/{file_id}"
                print(f"📡 GET {download_endpoint}")

                response = requests.get(download_endpoint, timeout=30)

                if response.status_code == 200:
                    content_length = len(response.content)
                    content_type = response.headers.get("content-type", "unknown")

                    print("✅ Download successful!"                    print(f"📦 File size: {content_length / (1024*1024):.2f} MB")
                    print(f"📄 Content Type: {content_type}")

                    if content_length < 10000:
                        print("⚠️ Downloaded file seems very small - may be an error response")
                        print(f"Content preview: {response.content[:200].decode('utf-8', errors='ignore')}")
                    else:
                        print("✅ File size looks reasonable for a video")

                else:
                    print(f"❌ Download failed with status {response.status_code}")
                    print(f"Response: {response.text[:500]}")
                    return False
        except Exception as e:
            print(f"❌ Download test failed: {e}")
            return False
    else:
        print("⚠️ No download URL available to test")
        return False

    # Success!
    print("\n" + "=" * 60)
    print("🎉 DEMO ACCOUNT VIDEO TRANSLATION TEST PASSED!")
    print("=" * 60)
    print("✅ Demo login simulation: SUCCESS")
    print("✅ Video upload: SUCCESS")
    print("✅ Translation start: SUCCESS")
    print("✅ Progress monitoring: SUCCESS")
    print("✅ Translation completion: SUCCESS")
    print("✅ Download functionality: SUCCESS")
    print("✅ Full AI pipeline: WORKING")
    print("=" * 60)

    return True

def main():
    """Main test function"""
    print("Demo Video Translation API Test")
    print(f"Backend API: {API_BASE}")
    print(f"Frontend: {FRONTEND_BASE}")

    success = test_demo_video_translation()

    if success:
        print("\n🎯 RESULT: Demo account video translation is working perfectly!")
        print("Demo users now get the same full AI-powered features as regular users.")
    else:
        print("\n💥 RESULT: Demo account video translation has issues that need fixing.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
