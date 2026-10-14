"""
Tests for client/sync_client.py - Synchronous Gran Sabio Client

Tests cover:
- Client initialization (base_url, api_key, timeout)
- Header building
- Request handling and error mapping
- Health & Info methods
- Project Management
- Content Generation
- Text Analysis
- Attachments
- Convenience methods
"""

import json
import os

# Import after mocking requests
import sys
import time
from unittest.mock import Mock, patch

import pytest
from requests import Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_requests():
    """Mock the requests module."""
    with patch('client.sync_client.requests') as mock_req:
        yield mock_req


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {}
    response.text = ""
    return response


@pytest.fixture
def http_response():
    """Build complete HTTP bodies with the real requests JSON decoder."""
    def build(status, body):
        response = Response()
        response.status_code = status
        response.headers["Content-Type"] = "application/json"
        response.encoding = "utf-8"
        response._content = body.encode("utf-8")
        return response

    return build


@pytest.fixture
def client():
    """Create a GranSabioClient with default settings."""
    from client import GranSabioClient
    return GranSabioClient(base_url="http://test-server:8000", api_key="test-key")


@pytest.fixture
def client_no_key():
    """Create a GranSabioClient without API key."""
    from client import GranSabioClient
    return GranSabioClient(base_url="http://test-server:8000")


# ==============================================================================
# Initialization Tests
# ==============================================================================

def test_system_one_sends_the_explicit_model(client, mock_requests, mock_response):
    mock_response.json.return_value = {"model": "jev-1.13.0", "answers": {"flag": {"noul": 0.9}}}
    mock_requests.request.return_value = mock_response
    result = client.evaluate_system_one("state", {"flag": {"type": "noul"}}, model="jev-latest")
    assert result["answers"]["flag"]["noul"] == 0.9
    assert mock_requests.request.call_args.kwargs["json"]["model"] == "jev-latest"
    assert mock_requests.request.call_args.args[:2] == ("POST", "http://test-server:8000/systemone")


class TestClientInitialization:
    """Tests for GranSabioClient initialization."""

    def test_default_base_url(self):
        """Given: No base_url provided, Then: Uses default localhost."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GRANSABIO_BASE_URL", None)
            from client import GranSabioClient
            client = GranSabioClient()
            assert client.base_url == "http://localhost:8000"

    def test_custom_base_url(self):
        """Given: Custom base_url, Then: Uses provided URL."""
        from client import GranSabioClient
        client = GranSabioClient(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"

    def test_base_url_strips_trailing_slash(self):
        """Given: base_url with trailing slash, Then: Strips trailing slash."""
        from client import GranSabioClient
        client = GranSabioClient(base_url="http://test:8000/")
        assert client.base_url == "http://test:8000"

    def test_base_url_from_env(self):
        """Given: GRANSABIO_BASE_URL env var, Then: Uses env value."""
        with patch.dict(os.environ, {"GRANSABIO_BASE_URL": "http://env-server:8000"}):
            from client import GranSabioClient
            client = GranSabioClient()
            assert client.base_url == "http://env-server:8000"

    def test_api_key_from_init(self):
        """Given: api_key in init, Then: Uses provided key."""
        from client import GranSabioClient
        client = GranSabioClient(api_key="init-key")
        assert client.api_key == "init-key"

    def test_api_key_from_env(self):
        """Given: GRANSABIO_API_KEY env var, Then: Uses env value."""
        with patch.dict(os.environ, {"GRANSABIO_API_KEY": "env-key"}):
            from client import GranSabioClient
            client = GranSabioClient()
            assert client.api_key == "env-key"

    def test_custom_timeout(self):
        """Given: Custom timeout, Then: Uses provided timeout."""
        from client import GranSabioClient
        client = GranSabioClient(timeout=(10, 300))
        assert client.timeout == (10, 300)

    def test_default_timeout(self):
        """Given: No timeout, Then: Uses the centralized long-running default."""
        from client import GranSabioClient
        client = GranSabioClient()
        assert client.timeout == (30, 12000)


# ==============================================================================
# Header Building Tests
# ==============================================================================

class TestHeaders:
    """Tests for _headers() method."""

    def test_headers_with_api_key(self, client):
        """Given: Client with API key, Then: Includes Authorization header."""
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"

    def test_headers_without_api_key(self, client_no_key):
        """Given: Client without API key, Then: No Authorization header."""
        headers = client_no_key._headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_provider_keys_are_not_loaded_from_environment(self):
        """Provider secrets in the process environment must not be forwarded."""
        from client import GranSabioClient

        provider_env = {
            "OPENAI_API_KEY": "openai-secret",
            "GOOGLE_API_KEY": "google-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
        }
        with patch.dict(os.environ, provider_env, clear=True):
            client = GranSabioClient(base_url="https://untrusted.example")

        headers = client._headers()
        assert client.provider_keys == {}
        assert "X-OpenAI-Key" not in headers
        assert "X-Gemini-Key" not in headers
        assert "X-Anthropic-Key" not in headers

    def test_explicit_provider_keys_are_preserved(self):
        """Callers can still opt in by passing the existing explicit argument."""
        from client import GranSabioClient

        client = GranSabioClient(
            base_url="https://trusted.example",
            provider_keys={"X-OpenAI-Key": "explicit-secret"},
        )

        assert client._headers()["X-OpenAI-Key"] == "explicit-secret"


# ==============================================================================
# Request Handling Tests
# ==============================================================================

class TestRequest:
    """Tests for _request() method."""

    def test_successful_request(self, client, mock_requests, mock_response):
        """Given: Successful request, Then: Returns response."""
        mock_requests.request.return_value = mock_response

        response = client._request("GET", "/test")

        mock_requests.request.assert_called_once()
        assert response == mock_response

    def test_connection_error_raises_client_error(self, client, mock_requests):
        """Given: ConnectionError, Then: Raises GranSabioClientError."""
        from requests import exceptions as req_exc

        from client import GranSabioClientError

        mock_requests.request.side_effect = req_exc.ConnectionError("Connection refused")

        with pytest.raises(GranSabioClientError) as exc:
            client._request("GET", "/test")

        assert "Cannot connect" in str(exc.value)

    def test_timeout_error_raises_client_error(self, client, mock_requests):
        """Given: Timeout, Then: Raises GranSabioClientError."""
        from requests import exceptions as req_exc

        from client import GranSabioClientError

        mock_requests.request.side_effect = req_exc.Timeout("Request timeout")

        with pytest.raises(GranSabioClientError) as exc:
            client._request("GET", "/test")

        assert "timed out" in str(exc.value)

    def test_request_exception_raises_client_error(self, client, mock_requests):
        """Given: Generic RequestException, Then: Raises GranSabioClientError."""
        from requests import exceptions as req_exc

        from client import GranSabioClientError

        mock_requests.request.side_effect = req_exc.RequestException("Network error")

        with pytest.raises(GranSabioClientError) as exc:
            client._request("GET", "/test")

        assert "Request failed" in str(exc.value)


# ==============================================================================
# Health & Info Tests
# ==============================================================================

class TestHealthAndInfo:
    """Tests for health and info methods."""

    def test_health_check_success(self, client, mock_requests, mock_response):
        """Given: Healthy server, Then: Returns health data."""
        mock_response.json.return_value = {
            "status": "healthy",
            "active_sessions": 5,
            "timestamp": "2025-01-07T12:00:00Z"
        }
        mock_requests.request.return_value = mock_response

        result = client.health_check()

        assert result["status"] == "healthy"
        assert result["active_sessions"] == 5

    def test_health_check_failure(self, client, mock_requests, mock_response):
        """Given: Health check fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status_code = 500
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.health_check()

        assert exc.value.status_code == 500

    def test_get_models_success(self, client, mock_requests, mock_response):
        """Given: Models endpoint succeeds, Then: Returns model dict."""
        mock_response.json.return_value = {
            "openai": [{"model_id": "gpt-4o"}],
            "anthropic": [{"model_id": "claude-sonnet-4-20250514"}]
        }
        mock_requests.request.return_value = mock_response

        result = client.get_models()

        assert "openai" in result
        assert "anthropic" in result

    def test_is_available_when_healthy(self, client, mock_requests, mock_response):
        """Given: Server is healthy, Then: Returns True."""
        mock_response.json.return_value = {"status": "healthy"}
        mock_requests.request.return_value = mock_response

        assert client.is_available() is True

    def test_is_available_when_unhealthy(self, client, mock_requests, mock_response):
        """Given: Server returns unhealthy, Then: Returns False."""
        mock_response.json.return_value = {"status": "unhealthy"}
        mock_requests.request.return_value = mock_response

        assert client.is_available() is False

    def test_is_available_when_unreachable(self, client, mock_requests):
        """Given: Server unreachable, Then: Returns False."""
        from requests import exceptions as req_exc
        mock_requests.request.side_effect = req_exc.ConnectionError()

        assert client.is_available() is False

    def test_get_info_returns_structured_data(self, client, mock_requests, mock_response):
        """Given: Health check succeeds, Then: get_info returns enriched data."""
        mock_response.json.return_value = {
            "status": "healthy",
            "active_sessions": 3,
            "timestamp": "2025-01-07T12:00:00Z"
        }
        mock_requests.request.return_value = mock_response

        result = client.get_info()

        assert result["service"] == "Gran Sabio LLM Engine"
        assert result["version"] == "1.0.0"
        assert result["status"] == "healthy"
        assert result["active_sessions"] == 3


# ==============================================================================
# Project Management Tests
# ==============================================================================

class TestProjectManagement:
    """Tests for project management methods."""

    def test_reserve_project_auto_id(self, client, mock_requests, mock_response):
        """Given: No project_id, Then: Server generates ID."""
        mock_response.json.return_value = {"project_id": "generated-uuid"}
        mock_requests.request.return_value = mock_response

        result = client.reserve_project()

        assert result == "generated-uuid"

    def test_reserve_project_custom_id(self, client, mock_requests, mock_response):
        """Given: Custom project_id, Then: Uses provided ID."""
        mock_response.json.return_value = {"project_id": "my-project"}
        mock_requests.request.return_value = mock_response

        result = client.reserve_project(project_id="my-project")

        assert result == "my-project"

    def test_reserve_project_failure(self, client, mock_requests, mock_response):
        """Given: Reserve fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status_code = 400
        mock_response.text = "Invalid project ID"
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError):
            client.reserve_project()

    def test_start_project(self, client, mock_requests, mock_response):
        """Given: Valid project_id, Then: Starts project."""
        mock_response.json.return_value = {"status": "started", "project_id": "proj-1"}
        mock_requests.request.return_value = mock_response

        result = client.start_project("proj-1")

        assert result["status"] == "started"

    def test_stop_project(self, client, mock_requests, mock_response):
        """Given: Active project, Then: Stops project."""
        mock_response.json.return_value = {"status": "stopped", "sessions_cancelled": 2}
        mock_requests.request.return_value = mock_response

        result = client.stop_project("proj-1")

        assert result["status"] == "stopped"


# ==============================================================================
# Content Generation Tests
# ==============================================================================

class TestGenerate:
    """Tests for generate() and related methods."""

    def test_generate_returns_immediately_when_not_waiting(self, client, mock_requests, mock_response):
        """Given: wait_for_completion=False, Then: Returns init response."""
        mock_response.json.return_value = {
            "session_id": "sess-123",
            "status": "started"
        }
        mock_requests.request.return_value = mock_response

        result = client.generate(
            prompt="Test prompt",
            qa_layers=[],
            wait_for_completion=False
        )

        assert result["session_id"] == "sess-123"

    def test_generate_raises_on_preflight_rejection(self, client, mock_requests, mock_response):
        """Given: Preflight rejects, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {
            "status": "preflight_rejected",
            "session_id": None,
            "project_id": "project-preflight",
            "request_name": "editorial-sufficiency",
            "preflight_feedback": {
                "user_feedback": "Contradictory requirements"
            }
        }
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.generate(prompt="Test", qa_layers=[])

        assert "rejected by preflight" in str(exc.value)
        assert exc.value.provider_outcome == "rejected"
        assert exc.value.details["response"] == mock_response.json.return_value
        assert exc.value.details["preflight_feedback"] == mock_response.json.return_value["preflight_feedback"]

    def test_generate_raises_on_auto_qa_rejection(self, client, mock_requests, mock_response):
        """Given: Auto-QA rejects, Then: Raises GranSabioClientError with feedback."""
        from client import GranSabioClientError

        mock_response.json.return_value = {
            "status": "auto_qa_rejected",
            "session_id": None,
            "project_id": "project-auto-qa",
            "request_name": "editorial-sufficiency",
            "preflight_feedback": {"decision": "proceed"},
            "auto_qa_feedback": {
                "message": "Planner could not build a non-contradictory QA contract"
            },
        }
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.generate(prompt="Test", qa_layers=[])

        assert "rejected by Auto-QA" in str(exc.value)
        assert exc.value.provider_outcome == "rejected"
        assert exc.value.details["response"] == mock_response.json.return_value
        assert exc.value.details["auto_qa_feedback"] == mock_response.json.return_value["auto_qa_feedback"]

    def test_generate_raises_on_missing_session_id(self, client, mock_requests, mock_response):
        """Given: No session_id in response, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {"status": "unknown"}
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.generate(prompt="Test", qa_layers=[])

        assert "No session_id" in str(exc.value)
        assert exc.value.provider_outcome is None

    def test_generate_includes_optional_params(self, client, mock_requests, mock_response):
        """Given: Optional params provided, Then: Included in payload."""
        mock_response.json.return_value = {"session_id": "sess-1", "status": "started"}
        mock_requests.request.return_value = mock_response

        client.generate(
            prompt="Test",
            min_words=500,
            max_words=1000,
            qa_layers=[],
            json_output=True,
            reasoning_effort="high",
            wait_for_completion=False
        )

        # Verify request was made with expected payload
        call_args = mock_requests.request.call_args
        payload = call_args[1]["json"]
        assert payload["min_words"] == 500
        assert payload["max_words"] == 1000
        assert payload["json_output"] is True
        assert payload["reasoning_effort"] == "high"
        assert "max_tokens" not in payload

    def test_generate_includes_explicit_max_tokens(self, client, mock_requests, mock_response):
        """Given: max_tokens provided, Then: Included in payload."""
        mock_response.json.return_value = {"session_id": "sess-1", "status": "started"}
        mock_requests.request.return_value = mock_response

        client.generate(
            prompt="Test",
            max_tokens=12000,
            qa_layers=[],
            wait_for_completion=False
        )

        payload = mock_requests.request.call_args[1]["json"]
        assert payload["max_tokens"] == 12000

    def test_get_status_success(self, client, mock_requests, mock_response):
        """Given: Valid session_id, Then: Returns status."""
        mock_response.json.return_value = {
            "session_id": "sess-1",
            "status": "generating",
            "progress": 50
        }
        mock_requests.request.return_value = mock_response

        result = client.get_status("sess-1")

        assert result["status"] == "generating"
        assert result["progress"] == 50

    def test_get_result_completed(self, client, mock_requests, mock_response):
        """Given: Completed session, Then: Returns result."""
        mock_response.json.return_value = {
            "session_id": "sess-1",
            "status": "completed",
            "content": "Generated content here"
        }
        mock_requests.request.return_value = mock_response

        result = client.get_result("sess-1")

        assert result["content"] == "Generated content here"

    def test_get_result_in_progress(self, client, mock_requests, mock_response):
        """Given: In-progress session, Then: Returns status indicator."""
        mock_response.status_code = 202
        mock_requests.request.return_value = mock_response

        result = client.get_result("sess-1")

        assert result["status"] == "in_progress"

    def test_stop_session(self, client, mock_requests, mock_response):
        """Given: Active session, Then: Cancels it."""
        mock_response.json.return_value = {"status": "cancelled"}
        mock_requests.request.return_value = mock_response

        result = client.stop_session("sess-1")

        assert result["status"] == "cancelled"


# ==============================================================================
# Wait for Result Tests
# ==============================================================================

class TestWaitForResult:
    """Tests for wait_for_result() method."""

    def test_wait_for_result_completes(self, client, mock_requests):
        """Given: Session completes, Then: Returns result."""
        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "completed"}

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {"content": "Final content"}

        mock_requests.request.side_effect = [status_response, result_response]

        result = client.wait_for_result("sess-1", poll_interval=0.01)

        assert result["content"] == "Final content"

    def test_wait_for_result_handles_failure(self, client, mock_requests):
        """Given: Session fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {
            "status": "failed",
            "failure_reason": "Generation error"
        }

        result_response = Mock()
        result_response.status_code = 200
        provider_error = {
            "kind": "quota_exhausted",
            "provider": "openai",
            "model": "gpt-5.1",
            "provider_error_code": "insufficient_quota",
        }
        result_response.json.return_value = {
            "status": "failed",
            "failure_reason": "Generation error",
            "error_type": "provider_error",
            "error_code": "quota_exhausted",
            "provider_error": provider_error,
        }

        mock_requests.request.side_effect = [status_response, result_response]

        with pytest.raises(GranSabioClientError) as exc:
            client.wait_for_result("sess-1", poll_interval=0.01)

        assert "status 'failed'" in str(exc.value)
        assert "Generation error" in str(exc.value)
        assert exc.value.details["error_code"] == "quota_exhausted"
        assert exc.value.details["provider_error"] == provider_error

    def test_wait_for_result_handles_semantic_rejection(self, client, mock_requests):
        """Given: Session is semantically rejected, Then: Raises GranSabioGenerationRejected."""
        from client import GranSabioGenerationRejected

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "rejected"}

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {
            "status": "rejected",
            "approved": False,
            "failure_reason": "QA said no",
        }

        mock_requests.request.side_effect = [status_response, result_response]

        with pytest.raises(GranSabioGenerationRejected) as exc:
            client.wait_for_result("sess-1", poll_interval=0.01)

        assert "QA said no" in str(exc.value)

    def test_wait_for_result_handles_cancellation(self, client, mock_requests):
        """Given: Session cancelled, Then: Raises GranSabioGenerationCancelled."""
        from client import GranSabioGenerationCancelled

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "cancelled"}
        mock_requests.request.return_value = status_response

        with pytest.raises(GranSabioGenerationCancelled) as exc:
            client.wait_for_result("sess-1", poll_interval=0.01)

        assert "cancelled" in str(exc.value)

    def test_wait_for_result_timeout(self, client, mock_requests):
        """Given: Session never completes, Then: Raises timeout error."""
        from client import GranSabioClientError

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "generating"}
        mock_requests.request.return_value = status_response

        with pytest.raises(GranSabioClientError) as exc:
            client.wait_for_result("sess-1", poll_interval=0.01, max_wait=0.05)

        assert "Timed out" in str(exc.value)

    def test_wait_for_result_calls_on_status_callback(self, client, mock_requests):
        """Given: on_status callback, Then: Called for each poll."""
        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "completed"}

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {"content": "Done"}

        mock_requests.request.side_effect = [status_response, result_response]

        callback = Mock()
        client.wait_for_result("sess-1", poll_interval=0.01, on_status=callback)

        callback.assert_called_once()


# ==============================================================================
# Text Analysis Tests
# ==============================================================================

class TestTextAnalysis:
    """Tests for text analysis methods."""

    def test_analyze_lexical_diversity(self, client, mock_requests, mock_response):
        """Given: Valid text, Then: Returns diversity analysis."""
        mock_response.json.return_value = {
            "decision": "PASS",
            "metrics": {"mtld": 85.5},
            "grade": "GREEN"
        }
        mock_requests.request.return_value = mock_response

        result = client.analyze_lexical_diversity("Sample text for analysis")

        assert result["decision"] == "PASS"
        assert "metrics" in result

    def test_analyze_lexical_diversity_with_options(self, client, mock_requests, mock_response):
        """Given: Custom options, Then: Included in request."""
        mock_response.json.return_value = {"decision": "PASS"}
        mock_requests.request.return_value = mock_response

        client.analyze_lexical_diversity(
            "Text",
            metrics="all",
            top_words=50,
            analyze_windows=True,
            language="es"
        )

        call_args = mock_requests.request.call_args
        payload = call_args[1]["json"]
        assert payload["metrics"] == "all"
        assert payload["top_words"] == 50
        assert payload["analyze_windows"] is True
        assert payload["language"] == "es"

    def test_analyze_repetition(self, client, mock_requests, mock_response):
        """Given: Valid text, Then: Returns repetition analysis."""
        mock_response.json.return_value = {
            "total_ngrams": 100,
            "repeated_ngrams": 5,
            "patterns": []
        }
        mock_requests.request.return_value = mock_response

        result = client.analyze_repetition("Sample text with some repetition")

        assert "total_ngrams" in result

    def test_analyze_repetition_with_options(self, client, mock_requests, mock_response):
        """Given: Custom n-gram options, Then: Included in request."""
        mock_response.json.return_value = {"total_ngrams": 50}
        mock_requests.request.return_value = mock_response

        client.analyze_repetition(
            "Text",
            min_n=3,
            max_n=7,
            min_count=3,
            diagnostics="full"
        )

        call_args = mock_requests.request.call_args
        payload = call_args[1]["json"]
        assert payload["min_n"] == 3
        assert payload["max_n"] == 7
        assert payload["min_count"] == 3
        assert payload["diagnostics"] == "full"


# ==============================================================================
# Attachment Tests
# ==============================================================================

class TestAttachments:
    """Tests for attachment methods."""

    def test_upload_attachment(self, client):
        """Given: File content, Then: Uploads as multipart."""
        with patch('client.sync_client.requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"upload_id": "att-123"}
            mock_request.return_value = mock_response

            result = client.upload_attachment(
                username="test-user",
                content=b"file content",
                filename="test.txt",
                content_type="text/plain"
            )

            assert result["upload_id"] == "att-123"
            mock_request.assert_called_once()

    def test_upload_attachment_failure(self, client):
        """Given: Upload fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        with patch('client.sync_client.requests.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid file"
            mock_request.return_value = mock_response

            with pytest.raises(GranSabioClientError):
                client.upload_attachment("user", b"content", "file.txt")

    def test_upload_attachment_base64(self, client, mock_requests, mock_response):
        """Given: Base64 content, Then: Uploads via JSON."""
        mock_response.json.return_value = {"upload_id": "att-456"}
        mock_requests.request.return_value = mock_response

        result = client.upload_attachment_base64(
            username="test-user",
            content_base64="SGVsbG8gV29ybGQ=",
            filename="test.txt",
            content_type="text/plain"
        )

        assert result["upload_id"] == "att-456"


# ==============================================================================
# Convenience Method Tests
# ==============================================================================

class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_generate_json_parses_content(self, client, mock_requests):
        """Given: JSON output, Then: Parses and includes parsed_content."""
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"session_id": "sess-1", "status": "started"}

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "completed"}

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {
            "content": '{"name": "test", "value": 42}'
        }

        mock_requests.request.side_effect = [init_response, status_response, result_response]

        result = client.generate_json(
            prompt="Generate JSON",
            schema={"type": "object"},
            poll_interval=0.01
        )

        assert result["parsed_content"] == {"name": "test", "value": 42}

    def test_generate_json_handles_parse_error(self, client, mock_requests):
        """Given: Invalid JSON output, Then: Sets parse_error flag."""
        init_response = Mock()
        init_response.status_code = 200
        init_response.json.return_value = {"session_id": "sess-1"}

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "completed"}

        result_response = Mock()
        result_response.status_code = 200
        result_response.json.return_value = {"content": "not valid json"}

        mock_requests.request.side_effect = [init_response, status_response, result_response]

        result = client.generate_json(
            prompt="Generate JSON",
            schema={"type": "object"},
            poll_interval=0.01
        )

        assert result["parsed_content"] is None
        assert result["parse_error"] is True

    def test_generate_fast_bypasses_qa(self, client, mock_requests, mock_response):
        """Given: generate_fast called, Then: Uses empty qa_layers."""
        mock_response.json.return_value = {"session_id": "sess-1"}
        mock_requests.request.return_value = mock_response

        client.generate_fast(prompt="Quick test", wait_for_completion=False)

        call_args = mock_requests.request.call_args
        payload = call_args[1]["json"]
        assert payload["qa_layers"] == []
        assert payload["max_iterations"] == 1


# ==============================================================================
# Body-read transient-error handling (retry coverage for get_status / get_result)
# ==============================================================================

class TestSyncBodyReadTransient:
    """get_status / get_result must convert mid-read requests errors to
    TransientGranSabioError so `_call_with_retry` can retry idempotent GET polls.
    """

    def test_get_status_converts_chunked_encoding_error(self, client):
        from requests import exceptions as req_exc

        from client import TransientGranSabioError

        response = Mock()
        response.status_code = 200
        response.json.side_effect = req_exc.ChunkedEncodingError("truncated chunked body")
        client._request = Mock(return_value=response)

        with pytest.raises(TransientGranSabioError) as exc_info:
            client.get_status("sess-123")

        assert "transient protocol error" in str(exc_info.value).lower()
        assert "/status/sess-123" in str(exc_info.value)

    def test_get_result_converts_content_decoding_error(self, client):
        from requests import exceptions as req_exc

        from client import TransientGranSabioError

        response = Mock()
        response.status_code = 200
        response.json.side_effect = req_exc.ContentDecodingError("truncated gzip")
        client._request = Mock(return_value=response)

        with pytest.raises(TransientGranSabioError):
            client.get_result("sess-xyz")

    def test_get_status_converts_json_decode_error(self, client):
        from client import TransientGranSabioError

        response = Mock()
        response.status_code = 200
        response.json.side_effect = ValueError("unterminated JSON")
        client._request = Mock(return_value=response)

        with pytest.raises(TransientGranSabioError):
            client.get_status("sess-json")

    def test_get_status_error_path_text_read_fails_fast(self, client):
        """HTTP error responses must not become retryable when their body is unreadable."""
        from requests import exceptions as req_exc

        from client import GranSabioClientError

        class BrokenTextResponse:
            status_code = 500

            @property
            def text(self):
                raise req_exc.ChunkedEncodingError("body truncated")

        client._request = Mock(return_value=BrokenTextResponse())

        with pytest.raises(GranSabioClientError) as exc_info:
            client.get_status("sess-timeout")

        assert exc_info.value.status_code == 500
        assert "<body unreadable>" in str(exc_info.value)

    def test_call_with_retry_retries_body_read_failure(self, client):
        """End-to-end: _call_with_retry must retry a body-read ChunkedEncodingError."""
        from requests import exceptions as req_exc

        response_fail = Mock()
        response_fail.status_code = 200
        response_fail.json.side_effect = req_exc.ChunkedEncodingError("truncated")

        response_ok = Mock()
        response_ok.status_code = 200
        response_ok.json.return_value = {"status": "completed"}

        client._request = Mock(side_effect=[response_fail, response_ok])

        with patch("time.sleep"):
            result = client._call_with_retry(
                op_name="get_status",
                resource_id="sess-retry",
                func=lambda: client.get_status("sess-retry"),
                start_time=time.monotonic(),
                max_wait=60.0,
            )

        assert result == {"status": "completed"}
        assert client._request.call_count == 2

    @pytest.mark.parametrize("outcome", ["recovered", "exhausted", "deadline"])
    def test_fetch_result_polling_retries_complete_invalid_json(
        self, client, mock_requests, http_response, outcome
    ):
        """A complete non-JSON 200 stays inside the bounded GET retry operation."""
        from client import GranSabioClientError, TransientGranSabioError
        from client._common import POLL_RETRY_BACKOFF_SECONDS

        invalid = http_response(200, "<html>Temporary proxy response</html>")
        final = {"status": "completed", "content": "Final content"}
        if outcome == "recovered":
            mock_requests.request.side_effect = [invalid, http_response(200, json.dumps(final))]
        else:
            mock_requests.request.return_value = invalid

        with patch("client.sync_client.time.sleep") as sleep:
            if outcome == "recovered":
                assert client._fetch_result_polling("sess-1", timeout_seconds=12000) == final
                attempts = 2
            else:
                timeout = 1 if outcome == "deadline" else 12000
                message = "Timed out" if outcome == "deadline" else "exhausted retries"
                with pytest.raises(GranSabioClientError, match=message) as error:
                    client._fetch_result_polling("sess-1", timeout_seconds=timeout)
                assert isinstance(error.value.__cause__, TransientGranSabioError)
                attempts = 1 if outcome == "deadline" else len(POLL_RETRY_BACKOFF_SECONDS) + 1
            assert sleep.call_count == attempts - 1

        assert [call.args[:2] for call in mock_requests.request.call_args_list] == [
            ("GET", "http://test-server:8000/result/sess-1")
        ] * attempts

    @pytest.mark.parametrize("entry", ["wait_for_result", "_fetch_result_polling"])
    def test_result_polling_handles_http_states_consistently(
        self, client, mock_requests, http_response, entry
    ):
        """202 waits; 425/429/503 back off without interpreting the HTTP body."""
        final = {"status": "completed", "content": "Final content"}
        responses = iter([
            http_response(202, '{"detail": "Generación en marcha"}'),
            http_response(425, "not JSON"),
            http_response(429, "<html>Rate limited</html>"),
            http_response(503, '{"detail": "Busy"}'),
            http_response(200, json.dumps(final)),
        ])

        def respond(method, url, **kwargs):
            assert method == "GET"
            if url.endswith("/status/sess-1"):
                return http_response(200, '{"status": "completed"}')
            assert url.endswith("/result/sess-1")
            return next(responses)

        mock_requests.request.side_effect = respond
        with patch("client.sync_client.time.sleep"):
            if entry == "wait_for_result":
                result = client.wait_for_result("sess-1", poll_interval=0, max_wait=120)
            else:
                result = client._fetch_result_polling("sess-1", poll_interval=0, timeout_seconds=120)

        assert result == final
        urls = [call.args[1] for call in mock_requests.request.call_args_list]
        assert urls.count("http://test-server:8000/result/sess-1") == 5
        assert urls.count("http://test-server:8000/status/sess-1") == (2 if entry == "wait_for_result" else 0)

    @pytest.mark.parametrize("entry", ["wait_for_result", "_fetch_result_polling"])
    def test_pending_result_respects_wait_deadline(
        self, client, mock_requests, http_response, entry
    ):
        """A terminal status followed by repeated 202 cannot wait indefinitely."""
        from client import GranSabioClientError

        def respond(method, url, **kwargs):
            assert method == "GET"
            if url.endswith("/status/sess-1"):
                return http_response(200, '{"status": "completed"}')
            return http_response(202, "")

        mock_requests.request.side_effect = respond
        with pytest.raises(GranSabioClientError, match="Timed out"):
            if entry == "wait_for_result":
                client.wait_for_result("sess-1", poll_interval=1, max_wait=0.05)
            else:
                client._fetch_result_polling("sess-1", poll_interval=1, timeout_seconds=0.05)

    def test_fetch_result_polling_preserves_active_window_during_backoff(
        self, client, mock_requests, http_response, monkeypatch
    ):
        """Recent stream activity extends an expired window before a transient GET."""
        from client._common import ActivityMonitor

        clock = Mock()
        clock.monotonic.return_value = 0.0

        def advance(seconds):
            clock.monotonic.return_value += seconds

        clock.sleep.side_effect = advance
        monkeypatch.setattr("client.sync_client.time", clock)
        monkeypatch.setattr("client._common.time", clock)
        monitor = ActivityMonitor(inactivity_window=10)
        final = {"status": "completed", "content": "Final content"}

        def respond(method, url, **kwargs):
            if mock_requests.request.call_count == 1:
                clock.monotonic.return_value = 10.0
                monitor.mark(iteration=2)
                return http_response(202, "")
            if mock_requests.request.call_count == 2:
                return http_response(503, "")
            return http_response(200, json.dumps(final))

        mock_requests.request.side_effect = respond
        result = client._fetch_result_polling(
            "sess-1", timeout_seconds=10, poll_interval=0, activity_monitor=monitor
        )

        assert result == final
        assert mock_requests.request.call_count == 3
        assert clock.monotonic.return_value == 15.0

    def test_fetch_result_polling_bad_request_fails_immediately(
        self, client, mock_requests, http_response
    ):
        """400 is an HTTP failure even when its detail resembles old pending text."""
        from client import GranSabioClientError, TransientGranSabioError

        mock_requests.request.return_value = http_response(
            400, '{"detail": "Generation still in progress"}'
        )
        with pytest.raises(GranSabioClientError) as error:
            client._fetch_result_polling("sess-1", timeout_seconds=1)

        assert error.value.status_code == 400
        assert not isinstance(error.value, TransientGranSabioError)
        assert mock_requests.request.call_count == 1

    def test_generate_does_not_retry_complete_invalid_json(
        self, client, mock_requests, http_response
    ):
        """An unreadable creation receipt leaves exactly one POST on the transport."""
        from client import GranSabioClientError, TransientGranSabioError

        mock_requests.request.return_value = http_response(200, "<html>Receipt unavailable</html>")
        with pytest.raises(GranSabioClientError, match="may have been accepted") as error:
            client.generate(prompt="Test", qa_layers=[], wait_for_completion=False)

        assert not isinstance(error.value, TransientGranSabioError)
        assert [call.args[:2] for call in mock_requests.request.call_args_list] == [
            ("POST", "http://test-server:8000/generate")
        ]

    def test_fetch_result_polling_non_retryable_status_body_read_fails_fast(self, client):
        """A truncated 500 body must not become a retryable polling failure."""
        from requests import exceptions as req_exc

        from client import GranSabioClientError

        response = Mock()
        response.status_code = 500
        response.json.side_effect = req_exc.ChunkedEncodingError("truncated")
        response.text = "server error"
        client._request = Mock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc_info:
            client._fetch_result_polling("sess-500", timeout_seconds=1, poll_interval=0)

        assert exc_info.value.status_code == 500
        assert client._request.call_count == 1


# ==============================================================================
# Module-Level Helpers
# ==============================================================================

class TestCreateClientHelper:
    """Tests for the module-level create_client() convenience function."""

    def test_create_client_function(self):
        """Given: create_client(), Then: Returns GranSabioClient."""
        from client.sync_client import create_client
        client = create_client(base_url="http://test:8000")
        from client import GranSabioClient
        assert isinstance(client, GranSabioClient)


# ==============================================================================
# GranSabioClientError Tests
# ==============================================================================

class TestGranSabioClientError:
    """Tests for GranSabioClientError exception."""

    def test_error_with_message_only(self):
        """Given: Message only, Then: Creates exception with defaults."""
        from client import GranSabioClientError
        error = GranSabioClientError("Test error")

        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.details == {}
        assert error.provider_outcome is None

    def test_error_with_status_code(self):
        """Given: Status code, Then: Stores status_code."""
        from client import GranSabioClientError
        error = GranSabioClientError("Not found", status_code=404)

        assert error.status_code == 404

    def test_error_with_details(self):
        """Given: Details dict, Then: Stores details."""
        from client import GranSabioClientError
        details = {"field": "value", "extra": 123}
        error = GranSabioClientError("Error", details=details)

        assert error.details == details


# ==============================================================================
# Idempotent Submission Tests
# ==============================================================================

class TestSyncIdempotentSubmission:
    """Request IDs on POST /generate, replay waits, and reconciliation reads."""

    @staticmethod
    def _response(status, body, headers=None):
        response = Response()
        response.status_code = status
        response.headers["Content-Type"] = "application/json"
        response.headers.update(headers or {})
        response.encoding = "utf-8"
        response._content = body.encode("utf-8")
        return response

    def test_request_id_travels_only_on_its_own_post(self, client, mock_requests):
        from client._common import SAAS_REQUEST_ID_HEADER

        mock_requests.request.return_value = self._response(
            200, json.dumps({"session_id": "sess-1", "status": "initialized"}),
        )
        client.generate(prompt="Test", qa_layers=[], wait_for_completion=False, request_id="axv2:abc")
        client.generate(prompt="Test", qa_layers=[], wait_for_completion=False)

        first, second = mock_requests.request.call_args_list
        assert first.kwargs["headers"][SAAS_REQUEST_ID_HEADER] == "axv2:abc"
        assert first.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert SAAS_REQUEST_ID_HEADER not in second.kwargs["headers"]
        assert SAAS_REQUEST_ID_HEADER not in client._headers()
        assert "request_id" not in first.kwargs["json"]

    @pytest.mark.parametrize("request_id", ["", "x" * 129, "has space", "slash/no"])
    def test_invalid_request_id_fails_before_sending(self, client, mock_requests, request_id):
        with pytest.raises(ValueError, match="request_id"):
            client.generate(prompt="Test", qa_layers=[], request_id=request_id)
        assert mock_requests.request.call_count == 0

    def test_pending_replay_maps_to_transient_with_retry_after(self, client, mock_requests):
        from client import TransientGranSabioError

        mock_requests.request.return_value = self._response(
            202, json.dumps({"detail": "still in progress", "request_id": "axv2:1"}),
            headers={"Retry-After": "5"},
        )
        with pytest.raises(TransientGranSabioError) as error:
            client.generate(prompt="Test", qa_layers=[], request_id="axv2:1")
        assert error.value.status_code == 202
        assert error.value.details["retry_after_seconds"] == 5.0
        assert error.value.details["request_id"] == "axv2:1"
        assert mock_requests.request.call_count == 1

    def test_pending_status_without_request_id_keeps_plain_error(self, client, mock_requests):
        from client import GranSabioClientError, TransientGranSabioError

        mock_requests.request.return_value = self._response(
            202, json.dumps({"detail": "still in progress"}), headers={"Retry-After": "5"},
        )
        with pytest.raises(GranSabioClientError) as error:
            client.generate(prompt="Test", qa_layers=[])
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == 202
        assert "request_id" not in error.value.details
        assert mock_requests.request.call_count == 1

    @pytest.mark.parametrize(("status", "body", "engine"), [
        (422, json.dumps({"detail": "invalid"}), True),
        (502, "<html>Bad gateway</html>", False),
    ])
    def test_http_errors_report_engine_origin(self, client, mock_requests, status, body, engine):
        from client import GranSabioClientError, TransientGranSabioError

        mock_requests.request.return_value = self._response(status, body)
        with pytest.raises(GranSabioClientError) as error:
            client.generate(prompt="Test", qa_layers=[])
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == status
        assert error.value.details["json_error_detail"] is engine
        assert "request_id" not in error.value.details

    def test_get_request_status_reads_encoded_snapshot_with_retry(self, client, mock_requests):
        snapshot = {"found": True, "request_id": "axv2:a", "state": "error", "status": "error",
                    "session_id": None, "project_id": None, "init_response": None,
                    "error": {"status_code": 422, "detail": "invalid"}}
        mock_requests.request.side_effect = [self._response(503, "busy"),
                                             self._response(200, json.dumps(snapshot))]
        with patch("client.sync_client.time.sleep") as sleep:
            assert client.get_request_status("axv2:a") == snapshot
        assert sleep.call_count == 1
        assert mock_requests.request.call_args.args[:2] == (
            "GET", "http://test-server:8000/api/saas/requests/axv2%3Aa",
        )

    @pytest.mark.parametrize("status", [400, 404])
    def test_get_request_status_http_errors_fail_without_retry(self, client, mock_requests, status):
        from client import GranSabioClientError, TransientGranSabioError

        mock_requests.request.return_value = self._response(status, '{"detail": "private"}')
        with pytest.raises(GranSabioClientError) as error:
            client.get_request_status("axv2:4")
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == status
        assert mock_requests.request.call_count == 1

    def test_list_sessions_passes_paging_and_returns_body(self, client, mock_requests):
        body = {"sessions": [], "project_id": None}
        mock_requests.request.return_value = self._response(200, json.dumps(body))
        assert client.list_sessions(limit=5) == body
        call = mock_requests.request.call_args
        assert call.args[:2] == ("GET", "http://test-server:8000/debugger/sessions")
        assert call.kwargs["params"] == {"limit": 5, "offset": 0}
        with pytest.raises(ValueError):
            client.list_sessions(limit=0)

    @pytest.mark.parametrize("status", [200, 403, 503])
    def test_session_details_are_an_encoded_read_without_generation(self, client, mock_requests, status):
        from client import GranSabioClientError, TransientGranSabioError

        payload = {"session_id": "session?one", "request": {"prompt": "Private source"}}
        mock_requests.request.return_value = self._response(status, json.dumps(payload))
        if status == 200:
            assert client.get_session_details("session?one") == payload
        else:
            expected = TransientGranSabioError if status == 503 else GranSabioClientError
            with pytest.raises(expected) as error:
                client.get_session_details("session?one")
            assert error.value.status_code == status
            assert "Private source" not in str(error.value)
        assert mock_requests.request.call_count == 1
        assert mock_requests.request.call_args.args[:2] == (
            "GET", "http://test-server:8000/debugger/sessions/session%3Fone",
        )
