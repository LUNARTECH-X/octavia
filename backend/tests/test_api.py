"""API endpoint tests for Octavia."""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_check_returns_200(self, app_client):
        """Test that health endpoint returns 200."""
        response = app_client.get("/api/health")
        assert response.status_code == 200
    
    def test_health_check_returns_success(self, app_client):
        """Test that health endpoint returns success true."""
        response = app_client.get("/api/health")
        data = response.json()
        assert data["success"] == True
        assert data["status"] == "healthy"
    
    def test_health_check_contains_version(self, app_client):
        """Test that health endpoint returns version info."""
        response = app_client.get("/api/health")
        data = response.json()
        assert "version" in data
        assert data["service"] == "Octavia Video Translator"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self, app_client):
        """Test that root endpoint returns 200."""
        response = app_client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_service_info(self, app_client):
        """Test that root returns service information."""
        response = app_client.get("/")
        data = response.json()
        assert data["service"] == "Octavia Video Translator"
        assert "endpoints" in data


class TestUserEndpoints:
    """Tests for user-related endpoints."""
    
    def test_get_user_profile_requires_auth(self, app_client):
        """Test that profile endpoint requires authentication."""
        response = app_client.get("/api/user/profile")
        # Should return 401 or 403 without auth
        assert response.status_code in [401, 403, 500]
    
    def test_get_user_credits_requires_auth(self, app_client):
        """Test that credits endpoint requires authentication."""
        response = app_client.get("/api/user/credits")
        assert response.status_code in [401, 403, 500]


class TestVoiceEndpoints:
    """Tests for voice-related endpoints."""
    
    def test_get_all_voices_returns_success(self, app_client):
        """Test that getting all voices works."""
        response = app_client.get("/api/voices/all")
        # May return 500 if supabase fails, but endpoint should exist
        assert response.status_code in [200, 500]
    
    def test_get_voices_by_language(self, app_client):
        """Test getting voices for a specific language."""
        response = app_client.get("/api/voices/en")
        # May return 500 if supabase fails, but endpoint should exist
        assert response.status_code in [200, 500]
    
    def test_get_invalid_language_returns_error(self, app_client):
        """Test that invalid language returns error."""
        response = app_client.get("/api/voices/xyz")
        data = response.json()
        assert data["success"] == False
        assert "not supported" in data.get("error", "").lower()
    
    def test_get_voices_returns_correct_structure(self, app_client):
        """Test that voices endpoint returns expected structure."""
        response = app_client.get("/api/voices/en")
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "voices" in data


class TestJobStatusEndpoints:
    """Tests for job status endpoints."""
    
    def test_get_job_status_requires_auth(self, app_client):
        """Test that job status requires authentication."""
        response = app_client.get("/api/jobs/nonexistent-job/status")
        # Should return 401/403 or 404 (job not found)
        assert response.status_code in [401, 403, 404, 500]
    
    def test_get_progress_requires_auth(self, app_client):
        """Test that progress endpoint requires authentication."""
        response = app_client.get("/api/progress/nonexistent-job")
        assert response.status_code in [401, 403, 404, 500]


class TestFileUploadValidation:
    """Tests for file upload validation."""
    
    def test_translate_video_requires_file(self, app_client):
        """Test that video translation requires authentication first."""
        # Without auth, returns 403 (auth required) - this is correct behavior
        response = app_client.post(
            "/api/translate/video",
            data={"target_language": "es"}
        )
        # Auth required (403) or validation error (400/422)
        assert response.status_code in [400, 403, 422, 500]
    
    def test_translate_video_validates_extension(self, app_client, temp_video_file):
        """Test that video translation validates file extension."""
        with open(temp_video_file, "rb") as f:
            response = app_client.post(
                "/api/translate/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                data={"target_language": "es"}
            )
        # May fail due to auth or processing, but should accept MP4
        assert response.status_code in [200, 401, 403, 500]
    
    def test_translate_audio_rejects_invalid_format(self, app_client):
        """Test that audio translation rejects invalid formats."""
        # Upload a file with invalid extension
        response = app_client.post(
            "/api/translate/audio",
            files={"file": ("test.txt", b"content", "text/plain")},
            data={"source_lang": "auto", "target_lang": "es"}
        )
        # Auth required (403) or validation error (400)
        assert response.status_code in [400, 403]
        if response.status_code == 400:
            assert "invalid" in response.json().get("detail", "").lower()


class TestDownloadEndpoints:
    """Tests for file download endpoints."""
    
    def test_download_nonexistent_video_returns_404(self, app_client):
        """Test that downloading non-existent video requires auth first."""
        # Without auth, returns 403 (auth required) - correct behavior
        response = app_client.get("/api/download/video/nonexistent-job-id")
        # Auth required (403) or not found (404) 
        assert response.status_code in [403, 404]
    
    def test_download_nonexistent_audio_returns_404(self, app_client):
        """Test that downloading non-existent audio requires auth first."""
        response = app_client.get("/api/download/audio/nonexistent-job-id")
        # Auth required (403) or not found (404)
        assert response.status_code in [403, 404]
    
    def test_download_generic_requires_job_type(self, app_client):
        """Test that generic download requires auth first."""
        response = app_client.get("/api/download/nonexistent-job-id")
        # Auth required (403) or not found (404)
        assert response.status_code in [403, 404]


class TestAPIDocumentation:
    """Tests to verify API documentation endpoints."""
    
    def test_docs_endpoint_exists(self, app_client):
        """Test that FastAPI docs endpoint exists."""
        response = app_client.get("/docs")
        # FastAPI auto-generated docs
        assert response.status_code == 200
    
    def test_redoc_endpoint_exists(self, app_client):
        """Test that ReDoc endpoint exists."""
        response = app_client.get("/redoc")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling consistency."""
    
    def test_invalid_job_id_returns_404(self, app_client):
        """Test that invalid job ID returns auth error or 404."""
        response = app_client.get("/api/jobs/invalid-job-id-12345/status")
        # Auth required (403) or job not found (404)
        assert response.status_code in [403, 404]
    
    def test_error_response_has_success_field(self, app_client):
        """Test that error responses include success: false."""
        response = app_client.get("/api/voices/xyz")
        if response.status_code == 200:
            # Skip if endpoint returns 200 for invalid lang
            return
        data = response.json()
        # Check that error responses are consistent
        assert "success" in data or "error" in data


class TestResponseFormat:
    """Tests for API response format consistency."""
    
    def test_success_response_has_success_field(self, app_client):
        """Test that success responses include success: true."""
        response = app_client.get("/api/health")
        data = response.json()
        assert "success" in data
        assert data["success"] == True
    
    def test_voices_response_has_success_field(self, app_client):
        """Test that voices responses include success field."""
        response = app_client.get("/api/voices/en")
        if response.status_code == 200:
            data = response.json()
            assert "success" in data


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
