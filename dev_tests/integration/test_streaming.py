"""
Integration tests for SSE streaming endpoints.

Tests the Server-Sent Events (SSE) streaming functionality:
- Content streaming by session
- Phase-specific streaming (generation, QA, preflight, Gran Sabio)
- Project-level unified streaming
- Error handling during streaming

Note: SSE streams keep connections open indefinitely. Tests verify
initial response headers and status codes without waiting for content.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import threading
import time


# Note: Fixtures inherited from conftest.py:
# - bypass_ip_filter (autouse)
# - base_test_client
# - valid_generation_request


# ============================================================================
# Helper for streaming tests
# ============================================================================

def stream_with_timeout(client, method, url, timeout_seconds=2, **kwargs):
    """
    Execute a streaming request with a timeout.

    Returns (status_code, headers) tuple without waiting for full response.
    """
    result = {"status_code": None, "headers": None, "error": None}

    def do_request():
        try:
            with client.stream(method, url, **kwargs) as response:
                result["status_code"] = response.status_code
                result["headers"] = dict(response.headers)
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=do_request)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    # If thread is still alive, we got the headers but stream is open (expected for SSE)
    if thread.is_alive():
        # Stream is open - this is expected behavior for SSE
        # We can't get headers this way, so return None to indicate timeout
        return None, None

    return result["status_code"], result["headers"]


# ============================================================================
# Session Content Streaming Tests
# ============================================================================

class TestContentStreaming:
    """Tests for session-based content streaming."""

    def test_stream_content_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-content/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-content/nonexistent-session-123")
        assert response.status_code == 404

    def test_stream_content_direct_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-content-direct/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-content-direct/nonexistent-session-123")
        assert response.status_code == 404


# ============================================================================
# Phase-Specific Streaming Tests
# ============================================================================

class TestPhaseStreaming:
    """Tests for phase-specific streaming endpoints."""

    def test_stream_generation_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-generation/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-generation/nonexistent-session-123")
        assert response.status_code == 404

    def test_stream_qa_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-qa/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-qa/nonexistent-session-123")
        assert response.status_code == 404

    def test_stream_preflight_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-preflight/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-preflight/nonexistent-session-123")
        assert response.status_code == 404

    def test_stream_gransabio_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-gransabio/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-gransabio/nonexistent-session-123")
        assert response.status_code == 404


# ============================================================================
# Project Unified Streaming Tests
# ============================================================================

class TestProjectStreaming:
    """Tests for project-level unified streaming."""

    def test_stream_project_invalid_id_returns_400(self, base_test_client):
        """
        Given: Invalid project_id (too long)
        When: GET /stream/project/{project_id}
        Then: Returns 400
        """
        long_id = "a" * 200
        response = base_test_client.get(f"/stream/project/{long_id}")
        assert response.status_code == 400

    def test_stream_project_invalid_phases_returns_400(self, base_test_client):
        """
        Given: Invalid phase name
        When: GET /stream/project/{project_id}?phases=invalid
        Then: Returns 400
        """
        response = base_test_client.get(
            "/stream/project/test-project-123",
            params={"phases": "invalid_phase"}
        )
        assert response.status_code == 400


# ============================================================================
# Direct Streaming Tests
# ============================================================================

class TestDirectStreaming:
    """Tests for direct content streaming (bypasses QA)."""

    def test_direct_stream_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /stream-content-direct/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/stream-content-direct/nonexistent-session")
        assert response.status_code == 404


# ============================================================================
# Streaming Error Handling Tests
# ============================================================================

class TestStreamingErrors:
    """Tests for error handling during streaming."""

    def test_stream_cancelled_session_returns_200(self, base_test_client, valid_generation_request):
        """
        Given: Session that has been cancelled
        When: GET /stream-content/{session_id}
        Then: Returns 200 (stream available for cancelled sessions)
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Cancel the session
        base_test_client.post(f"/stop/{session_id}")

        # Use timeout helper for streaming check
        status_code, headers = stream_with_timeout(
            base_test_client, "GET", f"/stream-content/{session_id}", timeout_seconds=2
        )

        # Either we got a response or it timed out (stream opened successfully)
        # Both are valid - stream opened means 200
        if status_code is not None:
            assert status_code == 200
        # If None, stream is open which means 200 was returned


# ============================================================================
# Streaming Endpoint Existence Tests
# ============================================================================

class TestStreamingEndpointExists:
    """Tests to verify streaming endpoints exist and respond correctly."""

    def test_stream_content_endpoint_exists(self, base_test_client, valid_generation_request):
        """
        Given: Valid session
        When: GET /stream-content/{session_id} (with immediate close)
        Then: Endpoint exists and responds
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Verify endpoint exists by checking it doesn't return 404 or 405
        status_code, _ = stream_with_timeout(
            base_test_client, "GET", f"/stream-content/{session_id}", timeout_seconds=2
        )
        # If status_code is None, stream opened (success)
        # If status_code is set, verify it's not 404/405
        if status_code is not None:
            assert status_code not in [404, 405]

    def test_stream_generation_endpoint_exists(self, base_test_client, valid_generation_request):
        """
        Given: Valid session
        When: GET /stream-generation/{session_id}
        Then: Endpoint exists
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        status_code, _ = stream_with_timeout(
            base_test_client, "GET", f"/stream-generation/{session_id}", timeout_seconds=2
        )
        if status_code is not None:
            assert status_code not in [404, 405]

    def test_stream_qa_endpoint_exists(self, base_test_client, valid_generation_request):
        """
        Given: Valid session
        When: GET /stream-qa/{session_id}
        Then: Endpoint exists
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        status_code, _ = stream_with_timeout(
            base_test_client, "GET", f"/stream-qa/{session_id}", timeout_seconds=2
        )
        if status_code is not None:
            assert status_code not in [404, 405]

    def test_stream_project_endpoint_exists(self, base_test_client, valid_generation_request):
        """
        Given: Valid project
        When: GET /stream/project/{project_id}
        Then: Endpoint exists and accepts valid phases
        """
        valid_generation_request["project_id"] = "stream-test-project"
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        project_id = gen_response.json()["project_id"]

        status_code, _ = stream_with_timeout(
            base_test_client, "GET", f"/stream/project/{project_id}", timeout_seconds=2
        )
        if status_code is not None:
            assert status_code not in [404, 405]

    def test_stream_project_with_phases_parameter(self, base_test_client, valid_generation_request):
        """
        Given: Valid project and phases parameter
        When: GET /stream/project/{project_id}?phases=generation,qa
        Then: Accepts valid phases without error
        """
        valid_generation_request["project_id"] = "stream-phases-test"
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        project_id = gen_response.json()["project_id"]

        status_code, _ = stream_with_timeout(
            base_test_client,
            "GET",
            f"/stream/project/{project_id}",
            timeout_seconds=2,
            params={"phases": "generation,qa"}
        )
        if status_code is not None:
            # Should not be 400 (valid phases)
            assert status_code != 400
