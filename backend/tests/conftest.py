"""Pytest configuration and fixtures for Octavia tests."""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set demo mode for tests
os.environ["DEMO_MODE"] = "true"
os.environ["ENABLE_TEST_MODE"] = "true"


@pytest.fixture
def mock_supabase():
    """Mock Supabase client for testing."""
    mock = MagicMock()
    
    # Mock user table
    mock_user = MagicMock()
    mock_user.id = "test-user-id"
    mock_user.email = "demo@octavia.com"
    mock_user.name = "Test User"
    mock_user.credits = 5000
    mock_user.is_verified = True
    mock_user.created_at = MagicMock()
    mock_user.created_at.isoformat = lambda: "2024-01-01T00:00:00"
    
    mock.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {
            "id": "test-user-id",
            "email": "demo@octavia.com",
            "name": "Test User",
            "credits": 5000,
            "is_verified": True
        }
    ]
    
    return mock


@pytest.fixture
def mock_user():
    """Mock user object for testing."""
    user = MagicMock()
    user.id = "test-user-id"
    user.email = "demo@octavia.com"
    user.name = "Test User"
    user.credits = 5000
    user.is_verified = True
    return user


@pytest.fixture
def app_client():
    """Create a test client for the FastAPI app."""
    from app import app
    from fastapi.testclient import TestClient
    
    return TestClient(app)


@pytest.fixture
def auth_headers(mock_user):
    """Create authentication headers for requests."""
    # This would normally include a JWT token
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "id": "test-job-id",
        "type": "video",
        "status": "processing",
        "progress": 50,
        "target_language": "es",
        "user_id": "test-user-id",
        "message": "Processing video..."
    }


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary video file for testing."""
    video_file = tmp_path / "test_video.mp4"
    # Write minimal valid MP4 header (not a real video, but enough for tests)
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    return str(video_file)
