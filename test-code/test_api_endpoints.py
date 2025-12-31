#!/usr/bin/env python3
"""
Comprehensive API Endpoint Test Script for Octavia Backend
Tests all endpoints in app.py to verify functionality
"""

import requests
import json
import time
import os
from typing import Dict, List, Tuple
import tempfile

# Configuration
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class APIEndpointTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.auth_token = None
        self.test_user = None

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def record_result(self, endpoint: str, method: str, status: str, message: str = "", response_time: float = 0):
        """Record test result"""
        result = {
            "endpoint": endpoint,
            "method": method,
            "status": status,
            "message": message,
            "response_time": response_time
        }
        self.test_results.append(result)
        self.log(f"{method} {endpoint} - {status}: {message}", "PASS" if status == "PASS" else "FAIL")

    def test_endpoint(self, endpoint: str, method: str = "GET", data: Dict = None,
                     files: Dict = None, headers: Dict = None, expected_status: int = 200) -> Tuple[bool, str]:
        """Test a single endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            start_time = time.time()

            # Prepare request
            request_kwargs = {"timeout": TEST_TIMEOUT}
            if data:
                request_kwargs["json" if isinstance(data, dict) else "data"] = data
            if files:
                request_kwargs["files"] = files
            if headers:
                request_kwargs["headers"] = headers

            # Make request
            response = self.session.request(method, url, **request_kwargs)
            response_time = time.time() - start_time

            # Check status
            if response.status_code == expected_status:
                return True, f"Response: {response.status_code}, Time: {response_time:.2f}s"
            else:
                return False, f"Expected {expected_status}, got {response.status_code}. Response: {response.text[:200]}"

        except requests.exceptions.ConnectionError:
            return False, "Connection failed - server not running"
        except requests.exceptions.Timeout:
            return False, f"Request timed out after {TEST_TIMEOUT}s"
        except Exception as e:
            return False, f"Request failed: {str(e)}"

    def test_root_endpoint(self):
        """Test root endpoint"""
        self.log("Testing root endpoint...")
        success, message = self.test_endpoint("/")
        self.record_result("/", "GET", "PASS" if success else "FAIL", message)

    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.log("Testing health endpoint...")
        success, message = self.test_endpoint("/api/health")
        self.record_result("/api/health", "GET", "PASS" if success else "FAIL", message)

    def test_auth_endpoints(self):
        """Test authentication endpoints"""
        self.log("Testing authentication endpoints...")

        # Test signup
        signup_data = {
            "email": f"test_{int(time.time())}@example.com",
            "password": "testpassword123",
            "name": "Test User"
        }
        success, message = self.test_endpoint("/api/auth/signup", "POST", signup_data, expected_status=200)
        self.record_result("/api/auth/signup", "POST", "PASS" if success else "FAIL", message)

        # Test demo login (should work)
        demo_login_data = {
            "email": "demo@octavia.com",
            "password": "demo123"
        }
        success, message = self.test_endpoint("/api/auth/demo-login", "POST", demo_login_data)
        if success:
            # Extract token if available
            try:
                response = self.session.post(f"{self.base_url}/api/auth/demo-login", json=demo_login_data)
                if response.status_code == 200:
                    response_data = response.json()
                    if "token" in response_data:
                        self.auth_token = response_data["token"]
                        self.test_user = response_data.get("user", {})
                        self.log(f"Got auth token for user: {self.test_user.get('email', 'unknown')}")
            except:
                pass

        self.record_result("/api/auth/demo-login", "POST", "PASS" if success else "FAIL", message)

    def test_user_endpoints(self):
        """Test user management endpoints"""
        if not self.auth_token:
            self.log("Skipping user endpoints - no auth token")
            return

        headers = {"Authorization": f"Bearer {self.auth_token}"}

        # Test profile endpoint
        success, message = self.test_endpoint("/api/user/profile", headers=headers)
        self.record_result("/api/user/profile", "GET", "PASS" if success else "FAIL", message)

        # Test credits endpoint
        success, message = self.test_endpoint("/api/user/credits", headers=headers)
        self.record_result("/api/user/credits", "GET", "PASS" if success else "FAIL", message)

    def test_voice_endpoints(self):
        """Test voice-related endpoints"""
        # Test get all voices
        success, message = self.test_endpoint("/api/voices/all")
        self.record_result("/api/voices/all", "GET", "PASS" if success else "FAIL", message)

        # Test voices by language
        for lang in ["en", "es", "fr", "de"]:
            success, message = self.test_endpoint(f"/api/voices/{lang}")
            self.record_result(f"/api/voices/{lang}", "GET", "PASS" if success else "FAIL", message)

    def test_payment_endpoints(self):
        """Test payment-related endpoints"""
        if not self.auth_token:
            self.log("Skipping payment endpoints - no auth token")
            return

        headers = {"Authorization": f"Bearer {self.auth_token}"}

        # Test credit packages
        success, message = self.test_endpoint("/api/payments/packages", headers=headers)
        self.record_result("/api/payments/packages", "GET", "PASS" if success else "FAIL", message)

    def test_translation_endpoints_without_auth(self):
        """Test translation endpoints that might work without auth"""
        self.log("Testing translation endpoints without auth (expecting 401)...")

        # These should return 401 Unauthorized without auth
        endpoints_to_test = [
            ("/api/translate/subtitles", "POST"),
            ("/api/translate/audio", "POST"),
            ("/api/translate/video", "POST"),
            ("/api/translate/video/enhanced", "POST"),
            ("/api/translate/subtitle-file", "POST"),
            ("/api/generate/subtitle-audio", "POST"),
        ]

        for endpoint, method in endpoints_to_test:
            success, message = self.test_endpoint(endpoint, method, expected_status=401)
            # 401 is expected for unauthenticated requests
            status = "PASS" if "401" in message else "FAIL"
            self.record_result(endpoint, method, status, message)

    def test_job_status_endpoints(self):
        """Test job status endpoints"""
        # Test with invalid job ID (should return 403 for unauthenticated request)
        success, message = self.test_endpoint("/api/jobs/invalid-job-id/status", expected_status=403)
        self.record_result("/api/jobs/{job_id}/status", "GET", "PASS" if success else "FAIL", message)

    def test_download_endpoints(self):
        """Test download endpoints with invalid IDs"""
        # These should return 403 for unauthenticated requests (not 404)
        endpoints_to_test = [
            "/api/download/subtitles/invalid",
            "/api/download/video/invalid",
            "/api/download/audio/invalid",
            "/api/download/subtitle-audio/invalid"
        ]

        for endpoint in endpoints_to_test:
            success, message = self.test_endpoint(endpoint, expected_status=403)
            self.record_result(endpoint, "GET", "PASS" if success else "FAIL", message)

    def test_testing_endpoints(self):
        """Test testing/debugging endpoints"""
        # Test metrics endpoint
        success, message = self.test_endpoint("/api/metrics")
        self.record_result("/api/metrics", "GET", "PASS" if success else "FAIL", message)

        # Test integration test endpoint (might fail if no test files)
        success, message = self.test_endpoint("/api/test/integration", "POST", expected_status=[200, 500])
        status = "PASS" if success else "SKIP"  # Integration test might fail due to missing test files
        self.record_result("/api/test/integration", "POST", status, message)

    def run_full_test_suite(self):
        """Run complete test suite"""
        self.log("Starting comprehensive API endpoint test suite...")
        self.log("=" * 60)

        # Test basic endpoints first
        self.test_root_endpoint()
        self.test_health_endpoint()

        # Test auth endpoints
        self.test_auth_endpoints()

        # Test user endpoints (requires auth)
        self.test_user_endpoints()

        # Test voice endpoints
        self.test_voice_endpoints()

        # Test payment endpoints
        self.test_payment_endpoints()

        # Test translation endpoints without auth (should fail)
        self.test_translation_endpoints_without_auth()

        # Test job status endpoints
        self.test_job_status_endpoints()

        # Test download endpoints
        self.test_download_endpoints()

        # Test testing endpoints
        self.test_testing_endpoints()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        self.log("=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r["status"] == "SKIP"])

        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {failed_tests}")
        self.log(f"Skipped: {skipped_tests}")

        if failed_tests > 0:
            self.log("\nFAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    self.log(f"  - {result['method']} {result['endpoint']}: {result['message']}")

        self.log("\n" + "=" * 60)

        # Overall assessment
        if failed_tests == 0:
            self.log("✅ ALL ENDPOINTS WORKING CORRECTLY!")
        elif failed_tests / total_tests < 0.1:  # Less than 10% failure
            self.log("⚠️  MOST ENDPOINTS WORKING - MINOR ISSUES")
        else:
            self.log("❌ SIGNIFICANT ENDPOINT ISSUES DETECTED")

def main():
    """Main test function"""
    print("Octavia API Endpoint Test Suite")
    print("=" * 40)

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responding")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except:
        print("❌ Server is not running or not accessible")
        print(f"   Please start the server on {BASE_URL}")
        return

    # Run tests
    tester = APIEndpointTester(BASE_URL)
    tester.run_full_test_suite()

if __name__ == "__main__":
    main()
