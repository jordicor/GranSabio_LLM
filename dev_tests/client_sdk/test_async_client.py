"""
Tests for client/async_client.py - Asynchronous Gran Sabio Client

Tests cover:
- Client initialization and session management
- Async context manager
- Header building
- Request handling and error mapping
- Health & Info methods
- Project Management
- Content Generation
- Text Analysis
- Attachments
- Convenience methods (generate_json, generate_fast, generate_parallel)
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    return session


@pytest.fixture
def mock_response():
    """Create a mock aiohttp response."""
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={})
    response.text = AsyncMock(return_value="")
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    return response


@pytest.fixture
def http_response():
    """Supply complete HTTP bodies at the session boundary with real JSON parsing."""
    def build(status, body):
        response = AsyncMock(spec=aiohttp.ClientResponse)
        response.status = status
        response.json.side_effect = lambda: json.loads(body)
        response.text.return_value = body
        response.__aenter__.return_value = response
        response.__aexit__.return_value = None
        return response

    return build


@pytest.fixture
async def client():
    """Create an AsyncGranSabioClient for testing."""
    from client import AsyncGranSabioClient
    client = AsyncGranSabioClient(
        base_url="http://test-server:8000",
        api_key="test-key"
    )
    # Create mock session
    client._session = MagicMock(spec=aiohttp.ClientSession)
    client._session.closed = False
    return client


@pytest.fixture
def client_not_connected():
    """Create an AsyncGranSabioClient without connection."""
    from client import AsyncGranSabioClient
    return AsyncGranSabioClient(
        base_url="http://test-server:8000",
        api_key="test-key"
    )


# ==============================================================================
# Initialization Tests
# ==============================================================================

async def test_system_one_sends_the_explicit_model(client, mock_response):
    mock_response.json.return_value = {"model": "jev-1.13.0", "answers": {"flag": {"noul": 0.9}}}
    client._session.request = AsyncMock(return_value=mock_response)
    result = await client.evaluate_system_one("state", {"flag": {"type": "noul"}}, model="jev-latest")
    assert result["answers"]["flag"]["noul"] == 0.9
    assert client._session.request.call_args.kwargs["json"]["model"] == "jev-latest"
    assert client._session.request.call_args.args[:2] == ("POST", "http://test-server:8000/systemone")


class TestAsyncClientInitialization:
    """Tests for AsyncGranSabioClient initialization."""

    def test_default_base_url(self):
        """Given: No base_url provided, Then: Uses default localhost."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GRANSABIO_BASE_URL", None)
            from client import AsyncGranSabioClient
            client = AsyncGranSabioClient()
            assert client.base_url == "http://localhost:8000"

    def test_custom_base_url(self):
        """Given: Custom base_url, Then: Uses provided URL."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient(base_url="http://custom:9000")
        assert client.base_url == "http://custom:9000"

    def test_base_url_strips_trailing_slash(self):
        """Given: base_url with trailing slash, Then: Strips trailing slash."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient(base_url="http://test:8000/")
        assert client.base_url == "http://test:8000"

    def test_api_key_from_init(self):
        """Given: api_key in init, Then: Uses provided key."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient(api_key="init-key")
        assert client.api_key == "init-key"

    def test_api_key_from_env(self):
        """Given: GRANSABIO_API_KEY env var, Then: Uses env value."""
        with patch.dict(os.environ, {"GRANSABIO_API_KEY": "env-key"}):
            from client import AsyncGranSabioClient
            client = AsyncGranSabioClient()
            assert client.api_key == "env-key"

    def test_default_timeout(self):
        """Given: No timeout, Then: Uses default ClientTimeout."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient()
        assert client.timeout.total == 12000
        assert client.timeout.connect == 30

    def test_custom_timeout(self):
        """Given: Custom timeout, Then: Uses provided timeout."""
        from client import AsyncGranSabioClient
        custom_timeout = aiohttp.ClientTimeout(total=300, connect=10)
        client = AsyncGranSabioClient(timeout=custom_timeout)
        assert client.timeout.total == 300
        assert client.timeout.connect == 10


# ==============================================================================
# Session Management Tests
# ==============================================================================

class TestSessionManagement:
    """Tests for session management methods."""

    @pytest.mark.asyncio
    async def test_connect_creates_session(self, client_not_connected):
        """Given: Not connected, Then: connect() creates session."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            await client_not_connected.connect()

            assert client_not_connected._session is not None
            mock_session_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        """Given: Connected client, Then: close() closes session."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient()
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        client._session = mock_session
        client._owns_session = True

        await client.close()

        mock_session.close.assert_called_once()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self):
        """Given: async with context, Then: Connects on enter, closes on exit."""
        from client import AsyncGranSabioClient

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.close = AsyncMock()
            mock_session_class.return_value = mock_session

            async with AsyncGranSabioClient() as client:
                assert client._session is not None

            mock_session.close.assert_called_once()

    def test_session_property_raises_when_not_connected(self, client_not_connected):
        """Given: Not connected, Then: session property raises RuntimeError."""
        with pytest.raises(RuntimeError) as exc:
            _ = client_not_connected.session

        assert "not connected" in str(exc.value).lower()


# ==============================================================================
# Header Building Tests
# ==============================================================================

class TestAsyncHeaders:
    """Tests for _headers() method."""

    @pytest.mark.asyncio
    async def test_headers_with_api_key(self, client):
        """Given: Client with API key, Then: Includes Authorization header."""
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_headers_without_api_key(self):
        """Given: Client without API key, Then: No Authorization header."""
        from client import AsyncGranSabioClient
        client = AsyncGranSabioClient(base_url="http://test:8000")

        headers = client._headers()
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_provider_keys_are_not_loaded_from_environment(self):
        """Provider secrets in the process environment must not be forwarded."""
        from client import AsyncGranSabioClient

        provider_env = {
            "OPENAI_API_KEY": "openai-secret",
            "GOOGLE_API_KEY": "google-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
        }
        with patch.dict(os.environ, provider_env, clear=True):
            client = AsyncGranSabioClient(base_url="https://untrusted.example")

        headers = client._headers()
        assert client.provider_keys == {}
        assert "X-OpenAI-Key" not in headers
        assert "X-Gemini-Key" not in headers
        assert "X-Anthropic-Key" not in headers

    def test_explicit_provider_keys_are_preserved(self):
        """Callers can still opt in by passing the existing explicit argument."""
        from client import AsyncGranSabioClient

        client = AsyncGranSabioClient(
            base_url="https://trusted.example",
            provider_keys={"X-Gemini-Key": "explicit-secret"},
        )

        assert client._headers()["X-Gemini-Key"] == "explicit-secret"


# ==============================================================================
# Request Handling Tests
# ==============================================================================

class TestAsyncRequest:
    """Tests for _request() method."""

    @pytest.mark.asyncio
    async def test_successful_request(self, client, mock_response):
        """Given: Successful request, Then: Returns response."""
        client._session.request = AsyncMock(return_value=mock_response)

        response = await client._request("GET", "/test")

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_connection_error_raises_client_error(self, client):
        """Given: ClientConnectorError, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        client._session.request = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError("Connection refused")
            )
        )

        with pytest.raises(GranSabioClientError) as exc:
            await client._request("GET", "/test")

        assert "Cannot connect" in str(exc.value)

    @pytest.mark.asyncio
    async def test_timeout_error_raises_client_error(self, client):
        """Given: TimeoutError, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        client._session.request = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(GranSabioClientError) as exc:
            await client._request("GET", "/test")

        assert "timed out" in str(exc.value)

    @pytest.mark.asyncio
    async def test_client_error_raises_client_error(self, client):
        """Given: Generic ClientError, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        client._session.request = AsyncMock(
            side_effect=aiohttp.ClientError("Network error")
        )

        with pytest.raises(GranSabioClientError) as exc:
            await client._request("GET", "/test")

        assert "Request failed" in str(exc.value)


# ==============================================================================
# Health & Info Tests
# ==============================================================================

class TestAsyncHealthAndInfo:
    """Tests for async health and info methods."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_response):
        """Given: Healthy server, Then: Returns health data."""
        mock_response.json.return_value = {
            "status": "healthy",
            "active_sessions": 5
        }
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["active_sessions"] == 5

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client, mock_response):
        """Given: Health check fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status = 500
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.health_check()

        assert exc.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_models_success(self, client, mock_response):
        """Given: Models endpoint succeeds, Then: Returns model dict."""
        mock_response.json.return_value = {
            "openai": [{"model_id": "gpt-4o"}],
            "anthropic": [{"model_id": "claude-sonnet-4-20250514"}]
        }
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.get_models()

        assert "openai" in result
        assert "anthropic" in result

    @pytest.mark.asyncio
    async def test_is_available_when_healthy(self, client, mock_response):
        """Given: Server is healthy, Then: Returns True."""
        mock_response.json.return_value = {"status": "healthy"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_when_unreachable(self, client):
        """Given: Server unreachable, Then: Returns False."""
        client._session.request = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(),
                os_error=OSError()
            )
        )

        result = await client.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_get_info_returns_structured_data(self, client, mock_response):
        """Given: Health check succeeds, Then: get_info returns enriched data."""
        mock_response.json.return_value = {
            "status": "healthy",
            "active_sessions": 3,
            "timestamp": "2025-01-07T12:00:00Z"
        }
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.get_info()

        assert result["service"] == "Gran Sabio LLM Engine"
        assert result["version"] == "1.0.0"
        assert result["status"] == "healthy"


# ==============================================================================
# Project Management Tests
# ==============================================================================

class TestAsyncProjectManagement:
    """Tests for async project management methods."""

    @pytest.mark.asyncio
    async def test_reserve_project_auto_id(self, client, mock_response):
        """Given: No project_id, Then: Server generates ID."""
        mock_response.json.return_value = {"project_id": "generated-uuid"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.reserve_project()

        assert result == "generated-uuid"

    @pytest.mark.asyncio
    async def test_reserve_project_custom_id(self, client, mock_response):
        """Given: Custom project_id, Then: Uses provided ID."""
        mock_response.json.return_value = {"project_id": "my-project"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.reserve_project(project_id="my-project")

        assert result == "my-project"

    @pytest.mark.asyncio
    async def test_reserve_project_failure(self, client, mock_response):
        """Given: Reserve fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status = 400
        mock_response.text.return_value = "Invalid ID"
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError):
            await client.reserve_project()

    @pytest.mark.asyncio
    async def test_reserve_project_success_body_read_failure_raises_client_error(self, client):
        """Given: Reserve succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.reserve_project()

        assert "Failed to read JSON response from /project/new" in str(exc.value)
        assert client._session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_start_project(self, client, mock_response):
        """Given: Valid project_id, Then: Starts project."""
        mock_response.json.return_value = {"status": "started"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.start_project("proj-1")

        assert result["status"] == "started"

    @pytest.mark.asyncio
    async def test_start_project_success_body_read_failure_raises_client_error(self, client):
        """Given: Project start succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.start_project("proj-1")

        assert "Failed to read JSON response from /project/start/proj-1" in str(exc.value)
        assert client._session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_project(self, client, mock_response):
        """Given: Active project, Then: Stops project."""
        mock_response.json.return_value = {"status": "stopped"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.stop_project("proj-1")

        assert result["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_project_success_body_read_failure_raises_client_error(self, client):
        """Given: Project stop succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.stop_project("proj-1")

        assert "Failed to read JSON response from /project/stop/proj-1" in str(exc.value)
        assert client._session.request.call_count == 1


# ==============================================================================
# Content Generation Tests
# ==============================================================================

class TestAsyncGenerate:
    """Tests for async generate() and related methods."""

    @pytest.mark.asyncio
    async def test_generate_returns_immediately_when_not_waiting(self, client, mock_response):
        """Given: wait_for_completion=False, Then: Returns init response."""
        mock_response.json.return_value = {"session_id": "sess-123", "status": "started"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.generate(
            prompt="Test prompt",
            qa_layers=[],
            wait_for_completion=False
        )

        assert result["session_id"] == "sess-123"

    @pytest.mark.asyncio
    async def test_generate_raises_on_preflight_rejection(self, client, mock_response):
        """Given: Preflight rejects, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {
            "status": "preflight_rejected",
            "session_id": None,
            "project_id": "project-preflight",
            "request_name": "editorial-sufficiency",
            "preflight_feedback": {"user_feedback": "Contradictory requirements"}
        }
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "rejected by preflight" in str(exc.value)
        assert exc.value.provider_outcome == "rejected"
        assert exc.value.details["response"] == mock_response.json.return_value
        assert exc.value.details["preflight_feedback"] == mock_response.json.return_value["preflight_feedback"]

    @pytest.mark.asyncio
    async def test_generate_raises_on_auto_qa_rejection(self, client, mock_response):
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
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "rejected by Auto-QA" in str(exc.value)
        assert exc.value.provider_outcome == "rejected"
        assert exc.value.details["response"] == mock_response.json.return_value
        assert exc.value.details["auto_qa_feedback"] == mock_response.json.return_value["auto_qa_feedback"]

    @pytest.mark.asyncio
    async def test_generate_raises_on_missing_session_id(self, client, mock_response):
        """Given: No session_id in response, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {"status": "unknown"}
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "No session_id" in str(exc.value)
        assert exc.value.provider_outcome is None

    @pytest.mark.asyncio
    async def test_generate_success_body_read_failure_raises_client_error(self, client):
        """Given: POST /generate succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[], wait_for_completion=False)

        assert "Failed to read JSON response from /generate" in str(exc.value)
        assert client._session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_omits_max_tokens_when_not_provided(self, client, mock_response):
        """Given: max_tokens omitted, Then: Payload lets server choose model default."""
        mock_response.json.return_value = {"session_id": "sess-1", "status": "started"}
        client._session.request = AsyncMock(return_value=mock_response)

        await client.generate(
            prompt="Test",
            qa_layers=[],
            wait_for_completion=False
        )

        payload = client._session.request.call_args[1]["json"]
        assert "max_tokens" not in payload

    @pytest.mark.asyncio
    async def test_generate_includes_explicit_max_tokens(self, client, mock_response):
        """Given: max_tokens provided, Then: Included in payload."""
        mock_response.json.return_value = {"session_id": "sess-1", "status": "started"}
        client._session.request = AsyncMock(return_value=mock_response)

        await client.generate(
            prompt="Test",
            max_tokens=12000,
            qa_layers=[],
            wait_for_completion=False
        )

        payload = client._session.request.call_args[1]["json"]
        assert payload["max_tokens"] == 12000

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "read_error"),
        [
            (500, aiohttp.ClientPayloadError("truncated response body")),
            (422, asyncio.TimeoutError()),
        ],
    )
    async def test_generate_error_body_uses_read_error_text(self, client, status_code, read_error):
        """Given: POST /generate error body read fails, Then: Degrades without leaking aiohttp read errors."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = status_code
        response.json = AsyncMock(return_value={})
        response.text = AsyncMock(side_effect=read_error)
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "<body unreadable>" in str(exc.value)
        assert exc.value.status_code == status_code
        assert response.text.await_count == 1

    @pytest.mark.asyncio
    async def test_get_status_success(self, client, mock_response):
        """Given: Valid session_id, Then: Returns status."""
        mock_response.json.return_value = {
            "session_id": "sess-1",
            "status": "generating",
            "progress": 50
        }
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.get_status("sess-1")

        assert result["status"] == "generating"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [200, 403, 503])
    async def test_session_details_are_an_encoded_read_without_generation(self, client, mock_response, status):
        from client import GranSabioClientError, TransientGranSabioError

        mock_response.status = status
        payload = {"session_id": "session?one", "request": {"prompt": "Private source"}}
        mock_response.json.return_value = payload
        client._session.request = AsyncMock(return_value=mock_response)
        if status == 200:
            assert await client.get_session_details("session?one") == payload
        else:
            expected = TransientGranSabioError if status == 503 else GranSabioClientError
            with pytest.raises(expected) as error:
                await client.get_session_details("session?one")
            assert error.value.status_code == status
            assert "Private source" not in str(error.value)
        assert client._session.request.await_count == 1
        assert client._session.request.call_args.args[:2] == (
            "GET", "http://test-server:8000/debugger/sessions/session%3Fone",
        )

    @pytest.mark.asyncio
    async def test_get_result_completed(self, client, mock_response):
        """Given: Completed session, Then: Returns result."""
        mock_response.json.return_value = {"content": "Generated content"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.get_result("sess-1")

        assert result["content"] == "Generated content"

    @pytest.mark.asyncio
    async def test_get_result_in_progress(self, client, mock_response):
        """Given: In-progress session, Then: Returns status indicator."""
        mock_response.status = 202
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.get_result("sess-1")

        assert result["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_stop_session(self, client, mock_response):
        """Given: Active session, Then: Cancels it."""
        mock_response.json.return_value = {"status": "cancelled"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.stop_session("sess-1")

        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_session_success_body_read_failure_raises_client_error(self, client):
        """Given: Session stop succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.stop_session("sess-1")

        assert "Failed to read JSON response from /stop/sess-1" in str(exc.value)
        assert client._session.request.call_count == 1


# ==============================================================================
# Wait for Result Tests
# ==============================================================================

class TestAsyncWaitForResult:
    """Tests for async wait_for_result() method."""

    @pytest.mark.asyncio
    async def test_wait_for_result_completes(self, client):
        """Given: Session completes, Then: Returns result."""
        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "completed"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={"content": "Final content"})
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(side_effect=[status_response, result_response])

        result = await client.wait_for_result("sess-1", poll_interval=0.01)

        assert result["content"] == "Final content"

    @pytest.mark.asyncio
    async def test_wait_for_result_handles_failure(self, client):
        """Given: Session fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={
            "status": "failed",
            "failure_reason": "Generation error"
        })
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={
            "status": "failed",
            "failure_reason": "Generation error",
        })
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(side_effect=[status_response, result_response])

        with pytest.raises(GranSabioClientError) as exc:
            await client.wait_for_result("sess-1", poll_interval=0.01)

        assert "status 'failed'" in str(exc.value)
        assert "Generation error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_wait_for_result_handles_semantic_rejection(self, client):
        """Given: Session is semantically rejected, Then: Raises GranSabioGenerationRejected."""
        from client import GranSabioGenerationRejected

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "rejected"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={
            "status": "rejected",
            "approved": False,
            "failure_reason": "QA said no",
        })
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(side_effect=[status_response, result_response])

        with pytest.raises(GranSabioGenerationRejected) as exc:
            await client.wait_for_result("sess-1", poll_interval=0.01)

        assert "QA said no" in str(exc.value)

    @pytest.mark.asyncio
    async def test_wait_for_result_handles_cancellation(self, client):
        """Given: Session cancelled, Then: Raises GranSabioGenerationCancelled."""
        from client import GranSabioGenerationCancelled

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "cancelled"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(return_value=status_response)

        with pytest.raises(GranSabioGenerationCancelled) as exc:
            await client.wait_for_result("sess-1", poll_interval=0.01)

        assert "cancelled" in str(exc.value)

    @pytest.mark.asyncio
    async def test_wait_for_result_timeout(self, client):
        """Given: Session never completes, Then: Raises timeout error."""
        from client import GranSabioClientError

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "generating"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(return_value=status_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.wait_for_result("sess-1", poll_interval=0.01, max_wait=0.05)

        assert "Timed out" in str(exc.value)

    @pytest.mark.asyncio
    async def test_wait_for_completion_alias(self, client):
        """Given: wait_for_completion called, Then: Delegates to wait_for_result."""
        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "completed"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={"content": "Done"})
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(side_effect=[status_response, result_response])

        result = await client.wait_for_completion("sess-1", poll_interval=0.01)

        assert result["content"] == "Done"


# ==============================================================================
# Text Analysis Tests
# ==============================================================================

class TestAsyncTextAnalysis:
    """Tests for async text analysis methods."""

    @pytest.mark.asyncio
    async def test_analyze_lexical_diversity(self, client, mock_response):
        """Given: Valid text, Then: Returns diversity analysis."""
        mock_response.json.return_value = {
            "decision": "PASS",
            "metrics": {"mtld": 85.5}
        }
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.analyze_lexical_diversity("Sample text")

        assert result["decision"] == "PASS"

    @pytest.mark.asyncio
    async def test_analyze_lexical_diversity_failure(self, client, mock_response):
        """Given: Analysis fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status = 400
        mock_response.text.return_value = "Invalid request"
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError):
            await client.analyze_lexical_diversity("text")

    @pytest.mark.asyncio
    async def test_analyze_repetition(self, client, mock_response):
        """Given: Valid text, Then: Returns repetition analysis."""
        mock_response.json.return_value = {"total_ngrams": 100}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.analyze_repetition("Sample text")

        assert "total_ngrams" in result

    @pytest.mark.asyncio
    async def test_analyze_repetition_failure(self, client, mock_response):
        """Given: Analysis fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.status = 500
        mock_response.text.return_value = "Server error"
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError):
            await client.analyze_repetition("text")


# ==============================================================================
# Attachment Tests
# ==============================================================================

class TestAsyncAttachments:
    """Tests for async attachment methods."""

    @pytest.mark.asyncio
    async def test_upload_attachment(self, client):
        """Given: File content, Then: Uploads as multipart."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"upload_id": "att-123"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.upload_attachment(
            username="test-user",
            content=b"file content",
            filename="test.txt"
        )

        assert result["upload_id"] == "att-123"

    @pytest.mark.asyncio
    async def test_upload_attachment_failure(self, client):
        """Given: Upload fails, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Invalid file")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError):
            await client.upload_attachment("user", b"content", "file.txt")

    @pytest.mark.asyncio
    async def test_upload_attachment_success_body_read_failure_raises_client_error(self, client):
        """Given: Multipart upload succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        mock_response.text = AsyncMock(return_value="")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.upload_attachment("user", b"content", "file.txt")

        assert "Failed to read JSON response from /attachments" in str(exc.value)
        assert client._session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_upload_attachment_base64_success_body_read_failure_raises_client_error(self, client):
        """Given: Base64 upload succeeds but JSON body is truncated, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated response body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._session.request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.upload_attachment_base64(
                username="test-user",
                content_base64="SGVsbG8=",
                filename="test.txt",
            )

        assert "Failed to read JSON response from /attachments/base64" in str(exc.value)
        assert client._session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_upload_attachment_base64(self, client, mock_response):
        """Given: Base64 content, Then: Uploads via JSON."""
        mock_response.json.return_value = {"upload_id": "att-456"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.upload_attachment_base64(
            username="test-user",
            content_base64="SGVsbG8=",
            filename="test.txt"
        )

        assert result["upload_id"] == "att-456"


# ==============================================================================
# Convenience Method Tests
# ==============================================================================

class TestAsyncConvenienceMethods:
    """Tests for async convenience methods."""

    @pytest.mark.asyncio
    async def test_generate_json_parses_content(self, client):
        """Given: JSON output, Then: Parses and includes parsed_content."""
        init_response = AsyncMock()
        init_response.status = 200
        init_response.json = AsyncMock(return_value={"session_id": "sess-1"})
        init_response.__aenter__ = AsyncMock(return_value=init_response)
        init_response.__aexit__ = AsyncMock()

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "completed"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={
            "content": '{"name": "test", "value": 42}'
        })
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(
            side_effect=[init_response, status_response, result_response]
        )

        result = await client.generate_json(
            prompt="Generate JSON",
            schema={"type": "object"},
            poll_interval=0.01
        )

        assert result["parsed_content"] == {"name": "test", "value": 42}

    @pytest.mark.asyncio
    async def test_generate_json_handles_parse_error(self, client):
        """Given: Invalid JSON output, Then: Sets parse_error flag."""
        init_response = AsyncMock()
        init_response.status = 200
        init_response.json = AsyncMock(return_value={"session_id": "sess-1"})
        init_response.__aenter__ = AsyncMock(return_value=init_response)
        init_response.__aexit__ = AsyncMock()

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "completed"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        result_response = AsyncMock()
        result_response.status = 200
        result_response.json = AsyncMock(return_value={"content": "not valid json"})
        result_response.__aenter__ = AsyncMock(return_value=result_response)
        result_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(
            side_effect=[init_response, status_response, result_response]
        )

        result = await client.generate_json(
            prompt="Generate JSON",
            schema={"type": "object"},
            poll_interval=0.01
        )

        assert result["parsed_content"] is None
        assert result["parse_error"] is True

    @pytest.mark.asyncio
    async def test_generate_fast_bypasses_qa(self, client, mock_response):
        """Given: generate_fast called, Then: Uses empty qa_layers."""
        mock_response.json.return_value = {"session_id": "sess-1"}
        client._session.request = AsyncMock(return_value=mock_response)

        await client.generate_fast(prompt="Quick test", wait_for_completion=False)

        # Verify the call was made with qa_layers=[]
        call_args = client._session.request.call_args
        payload = call_args[1]["json"]
        assert payload["qa_layers"] == []
        assert payload["max_iterations"] == 1

    @pytest.mark.asyncio
    async def test_generate_parallel_runs_concurrently(self, client):
        """Given: Multiple prompts, Then: Generates in parallel."""
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={"session_id": "sess-1"})
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(return_value=response)

        results = await client.generate_parallel(
            prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
            qa_layers=[],
            wait_for_completion=False
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_generate_parallel_returns_exceptions(self, client):
        """Given: Some prompts fail, Then: Returns exceptions in results."""

        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={"session_id": "sess-1"})
        success_response.__aenter__ = AsyncMock(return_value=success_response)
        success_response.__aexit__ = AsyncMock()

        fail_response = AsyncMock()
        fail_response.status = 400
        fail_response.text = AsyncMock(return_value="Bad request")
        fail_response.__aenter__ = AsyncMock(return_value=fail_response)
        fail_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(side_effect=[success_response, fail_response])

        results = await client.generate_parallel(
            prompts=["Good", "Bad"],
            qa_layers=[],
            wait_for_completion=False
        )

        assert len(results) == 2
        # One success, one exception
        assert isinstance(results[0], dict) or isinstance(results[0], Exception)


# ==============================================================================
# Module-Level Helpers
# ==============================================================================

class TestAsyncCreateClientHelper:
    """Tests for the module-level async create_client() convenience function."""

    @pytest.mark.asyncio
    async def test_create_client_function(self):
        """Given: create_client(), Then: Returns connected AsyncGranSabioClient."""
        from client.async_client import create_client

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            client = await create_client(base_url="http://test:8000")

            from client import AsyncGranSabioClient
            assert isinstance(client, AsyncGranSabioClient)
            assert client._session is not None


# ==============================================================================
# Body-read transient-error handling (retry coverage for get_status / get_result)
# ==============================================================================

class TestAsyncBodyReadTransient:
    """get_status / get_result / stop_session must convert mid-read aiohttp
    payload / disconnect errors to TransientGranSabioError so `_call_with_retry`
    retries them instead of bubbling them up as raw aiohttp exceptions.
    """

    @pytest.mark.asyncio
    async def test_get_status_converts_client_payload_error(self, client):
        from client import TransientGranSabioError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated chunked body")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(return_value=response)

        with pytest.raises(TransientGranSabioError) as exc_info:
            await client.get_status("sess-123")
        assert "transient protocol error" in str(exc_info.value).lower()
        assert "/status/sess-123" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_result_converts_server_disconnected(self, client):
        from client import TransientGranSabioError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(
            side_effect=aiohttp.ServerDisconnectedError("peer closed connection")
        )
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(return_value=response)

        with pytest.raises(TransientGranSabioError):
            await client.get_result("sess-xyz")

    @pytest.mark.asyncio
    async def test_get_status_converts_json_decode_error(self, client):
        from client import TransientGranSabioError

        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(side_effect=ValueError("unterminated JSON"))
        response.text = AsyncMock(return_value="")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(return_value=response)

        with pytest.raises(TransientGranSabioError):
            await client.get_status("sess-json")

    @pytest.mark.asyncio
    async def test_get_status_error_path_text_read_fails_fast(self, client):
        """HTTP error responses must not become retryable when their body is unreadable."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 500  # triggers the error path that reads text
        response.json = AsyncMock(return_value={})
        response.text = AsyncMock(side_effect=asyncio.TimeoutError())
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc_info:
            await client.get_status("sess-timeout")
        assert exc_info.value.status_code == 500
        assert "<body unreadable>" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_with_retry_retries_body_read_failure(self, client):
        """End-to-end: _call_with_retry must retry a body-read ClientPayloadError."""

        # First response raises ClientPayloadError on json(); second succeeds.
        response_fail = AsyncMock()
        response_fail.status = 200
        response_fail.json = AsyncMock(
            side_effect=aiohttp.ClientPayloadError("truncated")
        )
        response_fail.text = AsyncMock(return_value="")
        response_fail.__aenter__ = AsyncMock(return_value=response_fail)
        response_fail.__aexit__ = AsyncMock(return_value=None)

        response_ok = AsyncMock()
        response_ok.status = 200
        response_ok.json = AsyncMock(return_value={"status": "completed"})
        response_ok.text = AsyncMock(return_value="")
        response_ok.__aenter__ = AsyncMock(return_value=response_ok)
        response_ok.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(side_effect=[response_fail, response_ok])

        # Patch the retry-backoff sleep so the test runs fast.
        import time
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client._call_with_retry(
                op_name="get_status",
                resource_id="sess-retry",
                func=lambda: client.get_status("sess-retry"),
                start_time=time.monotonic(),
                max_wait=60.0,
            )
        assert result == {"status": "completed"}
        assert client._request.call_count == 2

    @pytest.mark.parametrize("outcome", ["recovered", "exhausted", "deadline"])
    async def test_fetch_result_polling_retries_complete_invalid_json(
        self, client, http_response, outcome
    ):
        """A complete non-JSON 200 stays inside the bounded GET retry operation."""
        from client import GranSabioClientError, TransientGranSabioError
        from client._common import POLL_RETRY_BACKOFF_SECONDS

        invalid = http_response(200, "<html>Temporary proxy response</html>")
        final = {"status": "completed", "content": "Final content"}
        transport = AsyncMock(return_value=invalid)
        if outcome == "recovered":
            transport.side_effect = [invalid, http_response(200, json.dumps(final))]
        client._session.request = transport

        with patch("client.async_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            if outcome == "recovered":
                assert await client._fetch_result_polling("sess-1", timeout_seconds=12000) == final
                attempts = 2
            else:
                timeout = 1 if outcome == "deadline" else 12000
                message = "Timed out" if outcome == "deadline" else "exhausted retries"
                with pytest.raises(GranSabioClientError, match=message) as error:
                    await client._fetch_result_polling("sess-1", timeout_seconds=timeout)
                assert isinstance(error.value.__cause__, TransientGranSabioError)
                attempts = 1 if outcome == "deadline" else len(POLL_RETRY_BACKOFF_SECONDS) + 1
            assert sleep.await_count == attempts - 1

        assert [call.args[:2] for call in transport.call_args_list] == [
            ("GET", "http://test-server:8000/result/sess-1")
        ] * attempts

    @pytest.mark.parametrize("entry", ["wait_for_result", "_fetch_result_polling"])
    async def test_result_polling_handles_http_states_consistently(self, client, http_response, entry):
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

        transport = AsyncMock(side_effect=respond)
        client._session.request = transport
        with patch("client.async_client.asyncio.sleep", new_callable=AsyncMock):
            if entry == "wait_for_result":
                result = await client.wait_for_result("sess-1", poll_interval=0, max_wait=120)
            else:
                result = await client._fetch_result_polling("sess-1", poll_interval=0, timeout_seconds=120)

        assert result == final
        urls = [call.args[1] for call in transport.call_args_list]
        assert urls.count("http://test-server:8000/result/sess-1") == 5
        assert urls.count("http://test-server:8000/status/sess-1") == (2 if entry == "wait_for_result" else 0)

    @pytest.mark.parametrize("entry", ["wait_for_result", "_fetch_result_polling"])
    async def test_pending_result_respects_wait_deadline(self, client, http_response, entry):
        """A terminal status followed by repeated 202 cannot wait indefinitely."""
        from client import GranSabioClientError

        def respond(method, url, **kwargs):
            assert method == "GET"
            if url.endswith("/status/sess-1"):
                return http_response(200, '{"status": "completed"}')
            return http_response(202, "")

        client._session.request = AsyncMock(side_effect=respond)
        with pytest.raises(GranSabioClientError, match="Timed out"):
            if entry == "wait_for_result":
                await client.wait_for_result("sess-1", poll_interval=1, max_wait=0.05)
            else:
                await client._fetch_result_polling("sess-1", poll_interval=1, timeout_seconds=0.05)

    async def test_fetch_result_polling_preserves_active_window_during_backoff(
        self, client, http_response, monkeypatch
    ):
        """Recent stream activity extends an expired window before a transient GET."""
        from client._common import ActivityMonitor

        clock = MagicMock()
        clock.monotonic.return_value = 0.0

        def advance(seconds):
            clock.monotonic.return_value += seconds

        monkeypatch.setattr("client.async_client.time", clock)
        monkeypatch.setattr("client._common.time", clock)
        monitor = ActivityMonitor(inactivity_window=10)
        final = {"status": "completed", "content": "Final content"}

        def respond(method, url, **kwargs):
            if client._session.request.call_count == 1:
                clock.monotonic.return_value = 10.0
                monitor.mark(iteration=2)
                return http_response(202, "")
            if client._session.request.call_count == 2:
                return http_response(503, "")
            return http_response(200, json.dumps(final))

        client._session.request = AsyncMock(side_effect=respond)
        with patch("client.async_client.asyncio.sleep", new=AsyncMock(side_effect=advance)):
            result = await client._fetch_result_polling(
                "sess-1", timeout_seconds=10, poll_interval=0, activity_monitor=monitor
            )

        assert result == final
        assert client._session.request.call_count == 3
        assert clock.monotonic.return_value == 15.0

    async def test_fetch_result_polling_bad_request_fails_immediately(self, client, http_response):
        """400 is an HTTP failure even when its detail resembles old pending text."""
        from client import GranSabioClientError, TransientGranSabioError

        client._session.request = AsyncMock(return_value=http_response(
            400, '{"detail": "Generation still in progress"}'
        ))
        with pytest.raises(GranSabioClientError) as error:
            await client._fetch_result_polling("sess-1", timeout_seconds=1)

        assert error.value.status_code == 400
        assert not isinstance(error.value, TransientGranSabioError)
        assert client._session.request.call_count == 1

    async def test_generate_does_not_retry_complete_invalid_json(self, client, http_response):
        """An unreadable creation receipt leaves exactly one POST on the transport."""
        from client import GranSabioClientError, TransientGranSabioError

        client._session.request = AsyncMock(return_value=http_response(
            200, "<html>Receipt unavailable</html>"
        ))
        with pytest.raises(GranSabioClientError, match="may have been accepted") as error:
            await client.generate(prompt="Test", qa_layers=[], wait_for_completion=False)

        assert not isinstance(error.value, TransientGranSabioError)
        assert [call.args[:2] for call in client._session.request.call_args_list] == [
            ("POST", "http://test-server:8000/generate")
        ]

    @pytest.mark.asyncio
    async def test_fetch_result_polling_non_retryable_status_body_read_fails_fast(self, client):
        """A truncated 500 body must not become a retryable polling failure."""
        from client import GranSabioClientError

        response = AsyncMock()
        response.status = 500
        response.json = AsyncMock(side_effect=aiohttp.ClientPayloadError("truncated"))
        response.text = AsyncMock(return_value="server error")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)

        client._request = AsyncMock(return_value=response)

        with pytest.raises(GranSabioClientError) as exc_info:
            await client._fetch_result_polling(
                "sess-500",
                timeout_seconds=1,
                poll_interval=0,
            )

        assert exc_info.value.status_code == 500
        assert client._request.call_count == 1


# ==============================================================================
# Idempotent Submission Tests
# ==============================================================================

class TestAsyncIdempotentSubmission:
    """Request IDs on POST /generate, replay waits, and reconciliation reads."""

    @staticmethod
    def _response(status, body, headers=None):
        response = AsyncMock(spec=aiohttp.ClientResponse)
        response.status = status
        response.headers = dict(headers or {})
        response.json.side_effect = lambda: json.loads(body)
        response.text.return_value = body
        response.__aenter__.return_value = response
        response.__aexit__.return_value = None
        return response

    async def test_request_id_travels_only_on_its_own_post(self, client):
        from client._common import SAAS_REQUEST_ID_HEADER

        client._session.request = AsyncMock(return_value=self._response(
            200, json.dumps({"session_id": "sess-1", "status": "initialized"}),
        ))
        await client.generate(prompt="Test", qa_layers=[], wait_for_completion=False,
                              request_id="axv2:abc-1.2_3")
        await client.generate(prompt="Test", qa_layers=[], wait_for_completion=False)

        first, second = client._session.request.call_args_list
        assert first.kwargs["headers"][SAAS_REQUEST_ID_HEADER] == "axv2:abc-1.2_3"
        assert first.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert SAAS_REQUEST_ID_HEADER not in second.kwargs["headers"]
        assert SAAS_REQUEST_ID_HEADER not in client._headers()
        assert "request_id" not in first.kwargs["json"]

    @pytest.mark.parametrize("request_id", ["", "x" * 129, "has space", "accent-é", 7])
    async def test_invalid_request_id_fails_before_sending(self, client, request_id):
        client._session.request = AsyncMock()
        with pytest.raises(ValueError, match="request_id"):
            await client.generate(prompt="Test", qa_layers=[], request_id=request_id)
        assert client._session.request.await_count == 0

    async def test_request_id_is_keyword_only(self):
        import inspect

        from client import AsyncGranSabioClient
        parameter = inspect.signature(AsyncGranSabioClient.generate).parameters["request_id"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    async def test_pending_replay_maps_to_transient_with_retry_after(self, client):
        from client import TransientGranSabioError

        client._session.request = AsyncMock(return_value=self._response(
            202, json.dumps({"detail": "still in progress", "request_id": "axv2:1"}),
            headers={"Retry-After": "5"},
        ))
        with pytest.raises(TransientGranSabioError) as error:
            await client.generate(prompt="Private prompt", qa_layers=[], request_id="axv2:1")
        assert error.value.status_code == 202
        assert error.value.details["retry_after_seconds"] == 5.0
        assert error.value.details["request_id"] == "axv2:1"
        assert error.value.details["json_error_detail"] is True
        assert "Private prompt" not in str(error.value)
        assert client._session.request.await_count == 1

    async def test_pending_status_without_request_id_keeps_plain_error(self, client):
        from client import GranSabioClientError, TransientGranSabioError

        client._session.request = AsyncMock(return_value=self._response(
            202, json.dumps({"detail": "still in progress"}), headers={"Retry-After": "5"},
        ))
        with pytest.raises(GranSabioClientError) as error:
            await client.generate(prompt="Private prompt", qa_layers=[])
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == 202
        assert "request_id" not in error.value.details
        assert "Private prompt" not in str(error.value)
        assert client._session.request.await_count == 1

    @pytest.mark.parametrize(("status", "body", "headers", "engine", "retry_after"), [
        (422, json.dumps({"detail": [{"msg": "field required"}]}), {}, True, None),
        (502, "<html>Bad gateway</html>", {}, False, None),
        (503, json.dumps({"detail": "Generation capacity reached"}), {"Retry-After": "7"}, True, 7.0),
    ])
    async def test_http_errors_report_engine_origin_without_body(
        self, client, status, body, headers, engine, retry_after,
    ):
        from client import GranSabioClientError, TransientGranSabioError

        client._session.request = AsyncMock(return_value=self._response(status, body, headers))
        with pytest.raises(GranSabioClientError) as error:
            await client.generate(prompt="Test", qa_layers=[], request_id="axv2:2")
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == status
        assert error.value.details["json_error_detail"] is engine
        assert error.value.details.get("retry_after_seconds") == retry_after
        assert error.value.details["request_id"] == "axv2:2"
        assert set(error.value.details) <= {
            "payload_metadata", "json_error_detail", "retry_after_seconds", "request_id",
        }

    async def test_get_request_status_reads_encoded_snapshot(self, client):
        snapshot = {"found": True, "request_id": "axv2:a.b", "state": "completed",
                    "status": "initialized", "session_id": "sess-9", "project_id": None,
                    "init_response": {"session_id": "sess-9", "status": "initialized"}}
        client._session.request = AsyncMock(return_value=self._response(200, json.dumps(snapshot)))
        assert await client.get_request_status("axv2:a.b") == snapshot
        assert client._session.request.call_args.args[:2] == (
            "GET", "http://test-server:8000/api/saas/requests/axv2%3Aa.b",
        )
        assert client._session.request.call_args.kwargs["json"] is None

    async def test_get_request_status_retries_transient_status(self, client):
        responses = [self._response(503, "unavailable"),
                     self._response(200, json.dumps({"found": False, "request_id": "axv2:3"}))]
        client._session.request = AsyncMock(side_effect=responses)
        with patch("client.async_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
            assert await client.get_request_status("axv2:3") == {"found": False, "request_id": "axv2:3"}
        assert sleep.await_count == 1
        assert client._session.request.await_count == 2

    @pytest.mark.parametrize("status", [400, 404])
    async def test_get_request_status_http_errors_fail_without_retry(self, client, status):
        from client import GranSabioClientError, TransientGranSabioError

        client._session.request = AsyncMock(return_value=self._response(status, '{"detail": "private"}'))
        with pytest.raises(GranSabioClientError) as error:
            await client.get_request_status("axv2:4")
        assert not isinstance(error.value, TransientGranSabioError)
        assert error.value.status_code == status
        assert "private" not in str(error.value)
        assert client._session.request.await_count == 1

    async def test_get_request_status_validates_request_id(self, client):
        client._session.request = AsyncMock()
        with pytest.raises(ValueError):
            await client.get_request_status("not/valid")
        assert client._session.request.await_count == 0

    async def test_list_sessions_passes_paging_and_returns_body(self, client):
        body = {"sessions": [{"session_id": "sess-1", "request_name": "job",
                              "created_at": "2026-10-13 10:00:00", "updated_at": "2026-10-13 10:01:00",
                              "status": "completed", "phase": "done", "project_id": None}],
                "project_id": None}
        client._session.request = AsyncMock(return_value=self._response(200, json.dumps(body)))
        assert await client.list_sessions(limit=100, offset=200, project_id="proj-1") == body
        call = client._session.request.call_args
        assert call.args[:2] == ("GET", "http://test-server:8000/debugger/sessions")
        assert call.kwargs["params"] == {"limit": 100, "offset": 200, "project_id": "proj-1"}

    @pytest.mark.parametrize(("limit", "offset", "project_id"), [
        (0, 0, None), (True, 0, None), (5, -1, None), (5, 0, " "),
    ])
    async def test_list_sessions_rejects_invalid_paging_before_sending(self, client, limit, offset, project_id):
        client._session.request = AsyncMock()
        with pytest.raises(ValueError):
            await client.list_sessions(limit=limit, offset=offset, project_id=project_id)
        assert client._session.request.await_count == 0

    async def test_list_sessions_disabled_debugger_is_a_client_error(self, client):
        from client import GranSabioClientError

        client._session.request = AsyncMock(return_value=self._response(404, '{"detail": "disabled"}'))
        with pytest.raises(GranSabioClientError) as error:
            await client.list_sessions()
        assert error.value.status_code == 404
        assert client._session.request.call_args.kwargs["params"] == {"limit": 25, "offset": 0}
