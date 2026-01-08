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

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

# Import after mocking requests
import sys
import os
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
        """Given: No timeout, Then: Uses default (30, 600)."""
        from client import GranSabioClient
        client = GranSabioClient()
        assert client.timeout == (30, 600)


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
            "status": "rejected",
            "preflight_feedback": {
                "user_feedback": "Contradictory requirements"
            }
        }
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.generate(prompt="Test", qa_layers=[])

        assert "rejected by preflight" in str(exc.value)

    def test_generate_raises_on_missing_session_id(self, client, mock_requests, mock_response):
        """Given: No session_id in response, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {"status": "unknown"}
        mock_requests.request.return_value = mock_response

        with pytest.raises(GranSabioClientError) as exc:
            client.generate(prompt="Test", qa_layers=[])

        assert "No session_id" in str(exc.value)

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
            "error": "Generation error"
        }
        mock_requests.request.return_value = status_response

        with pytest.raises(GranSabioClientError) as exc:
            client.wait_for_result("sess-1", poll_interval=0.01)

        assert "Generation failed" in str(exc.value)

    def test_wait_for_result_handles_cancellation(self, client, mock_requests):
        """Given: Session cancelled, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        status_response = Mock()
        status_response.status_code = 200
        status_response.json.return_value = {"status": "cancelled"}
        mock_requests.request.return_value = status_response

        with pytest.raises(GranSabioClientError) as exc:
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
        with patch('client.sync_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"upload_id": "att-123"}
            mock_post.return_value = mock_response

            result = client.upload_attachment(
                username="test-user",
                content=b"file content",
                filename="test.txt",
                content_type="text/plain"
            )

            assert result["upload_id"] == "att-123"
            mock_post.assert_called_once()

    def test_upload_attachment_failure(self, client):
        """Given: Upload fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        with patch('client.sync_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Invalid file"
            mock_post.return_value = mock_response

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
# Backward Compatibility Tests
# ==============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_bioai_client_alias(self):
        """Given: BioAIClient import, Then: Same as GranSabioClient."""
        from client import GranSabioClient, BioAIClient
        assert BioAIClient is GranSabioClient

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

    def test_bioai_client_error_alias(self):
        """Given: BioAIClientError import, Then: Same as GranSabioClientError."""
        from client import GranSabioClientError, BioAIClientError
        assert BioAIClientError is GranSabioClientError
