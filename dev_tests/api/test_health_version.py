"""
Tests for the `version` field exposed by the /health endpoint.

Verifies that the health payload includes a structured `version` object
populated from module-level BUILD_VERSION_INFO, with the expected shape
and value types. Does not assert on specific git-derived values, since
commit and dirty counts vary between runs.
"""

import datetime

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# Default headers to simulate internal IP for bypassing IP filter middleware.
INTERNAL_IP_HEADERS = {"X-Real-IP": "127.0.0.1"}


class InternalTestClient:
    """TestClient wrapper that injects internal-IP headers on every call."""

    def __init__(self, client: TestClient):
        self._client = client

    def _merge_headers(self, headers):
        merged = dict(INTERNAL_IP_HEADERS)
        if headers:
            merged.update(headers)
        return merged

    def get(self, url, **kwargs):
        kwargs["headers"] = self._merge_headers(kwargs.get("headers"))
        return self._client.get(url, **kwargs)


@pytest.fixture
def client():
    """Create a test client with mocked services and bypassed IP filter."""
    with patch('core.security.is_ip_allowed', return_value=True), \
         patch('core.app_state.get_ai_service') as mock_get_ai_service, \
         patch('core.generation_routes.ai_service') as _mock_ai_service:

        mock_service = MagicMock()
        mock_service.generate_content = AsyncMock(return_value="Generated content")
        mock_service.generate_content_stream = AsyncMock()
        mock_get_ai_service.return_value = mock_service

        from core.app_state import app
        base_client = TestClient(app)
        yield InternalTestClient(base_client)


class TestHealthVersionField:
    """Tests for the /health endpoint `version` field."""

    def test_health_contains_version_key(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Response is 200 and JSON body contains a `version` key.
        """
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    def test_version_has_required_shape(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: The `version` object exposes the 4 expected keys.
        """
        response = client.get("/health")
        assert response.status_code == 200
        version = response.json()["version"]
        assert isinstance(version, dict)
        for key in ("commit", "label", "dirty_files_at_startup", "started_at"):
            assert key in version, f"missing key in version payload: {key}"

    def test_version_field_types(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Each `version` field has the expected type and `started_at`
              is parseable as an ISO 8601 datetime.
        """
        response = client.get("/health")
        assert response.status_code == 200
        version = response.json()["version"]

        # commit: str OR None
        assert version["commit"] is None or isinstance(version["commit"], str)

        # label: str (may be "unknown" when git is unavailable)
        assert isinstance(version["label"], str)
        assert len(version["label"]) > 0

        # dirty_files_at_startup: int OR None
        dirty = version["dirty_files_at_startup"]
        assert dirty is None or isinstance(dirty, int)
        # bool is a subclass of int; reject it explicitly to catch regressions.
        assert not isinstance(dirty, bool)

        # started_at: str parseable via datetime.fromisoformat
        started_at = version["started_at"]
        assert isinstance(started_at, str)
        parsed = datetime.datetime.fromisoformat(started_at)
        assert isinstance(parsed, datetime.datetime)
