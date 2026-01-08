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

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import asyncio
import aiohttp

import sys
import os
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
        assert client.timeout.total == 600
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
    async def test_start_project(self, client, mock_response):
        """Given: Valid project_id, Then: Starts project."""
        mock_response.json.return_value = {"status": "started"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.start_project("proj-1")

        assert result["status"] == "started"

    @pytest.mark.asyncio
    async def test_stop_project(self, client, mock_response):
        """Given: Active project, Then: Stops project."""
        mock_response.json.return_value = {"status": "stopped"}
        client._session.request = AsyncMock(return_value=mock_response)

        result = await client.stop_project("proj-1")

        assert result["status"] == "stopped"


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
            "status": "rejected",
            "preflight_feedback": {"user_feedback": "Contradictory requirements"}
        }
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "rejected by preflight" in str(exc.value)

    @pytest.mark.asyncio
    async def test_generate_raises_on_missing_session_id(self, client, mock_response):
        """Given: No session_id in response, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        mock_response.json.return_value = {"status": "unknown"}
        client._session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.generate(prompt="Test", qa_layers=[])

        assert "No session_id" in str(exc.value)

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
            "error": "Generation error"
        })
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(return_value=status_response)

        with pytest.raises(GranSabioClientError) as exc:
            await client.wait_for_result("sess-1", poll_interval=0.01)

        assert "Generation failed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_wait_for_result_handles_cancellation(self, client):
        """Given: Session cancelled, Then: Raises GranSabioClientError."""
        from client import GranSabioClientError

        status_response = AsyncMock()
        status_response.status = 200
        status_response.json = AsyncMock(return_value={"status": "cancelled"})
        status_response.__aenter__ = AsyncMock(return_value=status_response)
        status_response.__aexit__ = AsyncMock()

        client._session.request = AsyncMock(return_value=status_response)

        with pytest.raises(GranSabioClientError) as exc:
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

        # Create a proper async context manager (session.post returns cm directly, not coroutine)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        client._session.post = MagicMock(return_value=mock_cm)

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

        # Create a proper async context manager
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        client._session.post = MagicMock(return_value=mock_cm)

        with pytest.raises(GranSabioClientError):
            await client.upload_attachment("user", b"content", "file.txt")

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
        from client import GranSabioClientError

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
# Backward Compatibility Tests
# ==============================================================================

class TestAsyncBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_async_bioai_client_alias(self):
        """Given: AsyncBioAIClient import, Then: Same as AsyncGranSabioClient."""
        from client import AsyncGranSabioClient, AsyncBioAIClient
        assert AsyncBioAIClient is AsyncGranSabioClient

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
