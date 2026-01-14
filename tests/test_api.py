"""
Tests for API Module
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Tests for API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        assert "model_loaded" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
