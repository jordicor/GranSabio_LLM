"""
Tests for API Endpoints - Basic CRUD Operations

Sub-Phase 2.4: Tests for basic API endpoint functionality.
This module tests health, models, generate validation, and status/result endpoints.

Test Categories:
1. Health Endpoint (5 tests)
2. Models Endpoint (5 tests)
3. Generate Endpoint Validation (20 tests)
4. Status/Result Endpoints (10 tests)

Total: 40 tests
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import uuid


# Default headers to simulate internal IP for bypassing IP filter middleware
INTERNAL_IP_HEADERS = {"X-Real-IP": "127.0.0.1"}


# ============================================================================
# Fixtures
# ============================================================================

class InternalTestClient:
    """
    Wrapper around TestClient that automatically adds internal IP headers.
    This bypasses the IP filter middleware for testing purposes.
    """

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

    def post(self, url, **kwargs):
        kwargs["headers"] = self._merge_headers(kwargs.get("headers"))
        return self._client.post(url, **kwargs)

    def put(self, url, **kwargs):
        kwargs["headers"] = self._merge_headers(kwargs.get("headers"))
        return self._client.put(url, **kwargs)

    def delete(self, url, **kwargs):
        kwargs["headers"] = self._merge_headers(kwargs.get("headers"))
        return self._client.delete(url, **kwargs)

    def patch(self, url, **kwargs):
        kwargs["headers"] = self._merge_headers(kwargs.get("headers"))
        return self._client.patch(url, **kwargs)


@pytest.fixture
def client():
    """Create a test client with mocked services and bypassed IP filter."""
    # Patch security to bypass IP filter for testing
    with patch('core.security.is_ip_allowed', return_value=True), \
         patch('core.app_state.get_ai_service') as mock_get_ai_service, \
         patch('core.generation_routes.ai_service') as mock_ai_service:

        # Setup mock AI service
        mock_service = MagicMock()
        mock_service.generate_content = AsyncMock(return_value="Generated content")
        mock_service.generate_content_stream = AsyncMock()
        mock_get_ai_service.return_value = mock_service

        from core.app_state import app
        base_client = TestClient(app)
        yield InternalTestClient(base_client)


@pytest.fixture
def mock_preflight_proceed():
    """Mock preflight validation to always proceed."""
    from models import PreflightResult
    result = PreflightResult(
        decision="proceed",
        user_feedback="Validation passed",
        summary="Request validated",
        confidence=0.95,
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )
    with patch('core.generation_routes.run_preflight_validation',
               new_callable=AsyncMock, return_value=result):
        yield result


@pytest.fixture
def mock_preflight_reject():
    """Mock preflight validation to reject."""
    from models import PreflightResult
    result = PreflightResult(
        decision="reject",
        user_feedback="Request contains contradictions",
        summary="Validation failed",
        confidence=0.92,
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )
    with patch('core.generation_routes.run_preflight_validation',
               new_callable=AsyncMock, return_value=result):
        yield result


@pytest.fixture
def valid_generate_request():
    """Return a valid minimal generate request payload."""
    return {
        "prompt": "Write a comprehensive article about software testing best practices",
        "generator_model": "gpt-4o",
        "qa_layers": [],  # No QA to simplify tests
        "qa_models": [],
    }


@pytest.fixture
def valid_generate_request_with_qa():
    """Return a valid generate request with QA layers."""
    return {
        "prompt": "Write a comprehensive article about software testing best practices",
        "generator_model": "gpt-4o",
        "qa_layers": [
            {
                "name": "Quality Check",
                "description": "Check content quality",
                "criteria": "Content should be clear and accurate",
                "min_score": 7.0,
                "order": 1
            }
        ],
        "qa_models": ["gpt-4o"],
        "gran_sabio_model": "claude-sonnet-4-20250514"
    }


# ============================================================================
# Health Endpoint Tests (5 tests)
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Returns 200 OK
        """
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Returns status 'healthy'
        """
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_active_sessions_count(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Response includes active_sessions count
        """
        response = client.get("/health")
        data = response.json()
        assert "active_sessions" in data
        assert isinstance(data["active_sessions"], int)

    def test_health_includes_timestamp(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Response includes ISO timestamp
        """
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        # Validate ISO format (basic check)
        assert "T" in data["timestamp"]

    def test_health_returns_json_content_type(self, client):
        """
        Given: A running API server
        When: GET /health is called
        Then: Content-Type is application/json
        """
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ============================================================================
# API Info Endpoint Tests (part of Models tests)
# ============================================================================

class TestApiInfoEndpoint:
    """Tests for GET /api endpoint."""

    def test_api_info_returns_200(self, client):
        """
        Given: A running API server
        When: GET /api is called
        Then: Returns 200 OK
        """
        response = client.get("/api")
        assert response.status_code == 200

    def test_api_info_includes_service_name(self, client):
        """
        Given: A running API server
        When: GET /api is called
        Then: Response includes service name
        """
        response = client.get("/api")
        data = response.json()
        assert data["service"] == "Gran Sabio LLM Engine"

    def test_api_info_includes_status(self, client):
        """
        Given: A running API server
        When: GET /api is called
        Then: Response includes operational status
        """
        response = client.get("/api")
        data = response.json()
        assert data["status"] == "operational"


# ============================================================================
# Models Endpoint Tests (5 tests)
# ============================================================================

class TestModelsEndpoint:
    """Tests for GET /models and GET /models/{name} endpoints."""

    def test_models_list_returns_200(self, client):
        """
        Given: A running API server
        When: GET /models is called
        Then: Returns 200 OK
        """
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_list_returns_dict_by_provider(self, client):
        """
        Given: A running API server
        When: GET /models is called
        Then: Returns dict with provider keys
        """
        response = client.get("/models")
        data = response.json()
        assert isinstance(data, dict)
        # Should have at least one provider
        assert len(data) > 0

    def test_models_known_model_returns_200(self, client):
        """
        Given: A known model name
        When: GET /models/{name} is called
        Then: Returns 200 OK with model info
        """
        response = client.get("/models/gpt-4o")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data

    def test_models_unknown_model_raises_error(self, client):
        """
        Given: An unknown model name
        When: GET /models/{name} is called
        Then: Raises RuntimeError (unhandled by endpoint)

        Note: The current API implementation raises RuntimeError for unknown models
        which is not caught by the endpoint. Ideally this should return 404 HTTPException.
        This test documents the current behavior.
        """
        with pytest.raises(RuntimeError) as exc_info:
            client.get("/models/nonexistent-model-xyz")
        assert "Unknown model" in str(exc_info.value)

    def test_models_response_excludes_api_key(self, client):
        """
        Given: A known model name
        When: GET /models/{name} is called
        Then: Response does not include raw api_key
        """
        response = client.get("/models/gpt-4o")
        data = response.json()
        # Should have has_api_key flag, not raw key
        assert "api_key" not in data
        assert "has_api_key" in data


# ============================================================================
# Generate Endpoint Validation Tests (20 tests)
# ============================================================================

class TestGenerateEndpointValidation:
    """Tests for POST /generate endpoint input validation."""

    def test_generate_missing_prompt_returns_422(self, client):
        """
        Given: Request without prompt field
        When: POST /generate is called
        Then: Returns 422 Unprocessable Entity
        """
        response = client.post("/generate", json={
            "generator_model": "gpt-4o"
        })
        assert response.status_code == 422

    def test_generate_short_prompt_returns_422(self, client):
        """
        Given: Prompt shorter than 10 characters
        When: POST /generate is called
        Then: Returns 422 validation error
        """
        response = client.post("/generate", json={
            "prompt": "Hi",
            "generator_model": "gpt-4o"
        })
        assert response.status_code == 422

    def test_generate_missing_generator_model_returns_400(self, client):
        """
        Given: Request without generator_model
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about testing",
            "qa_layers": []
        })
        assert response.status_code == 400
        assert "generator_model" in response.json()["detail"].lower()

    def test_generate_invalid_generator_model_returns_400(self, client):
        """
        Given: Invalid generator_model name
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about testing",
            "generator_model": "invalid-model-xyz",
            "qa_layers": []
        })
        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    def test_generate_qa_layers_without_qa_models_returns_400(self, client):
        """
        Given: QA layers provided without qa_models
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Check quality",
                    "criteria": "Be accurate",
                    "min_score": 7.0,
                    "order": 1
                }
            ]
        })
        assert response.status_code == 400
        assert "qa_models" in response.json()["detail"].lower()

    def test_generate_qa_layers_with_empty_qa_models_returns_400(self, client):
        """
        Given: QA layers provided with empty qa_models list
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Check quality",
                    "criteria": "Be accurate",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": []
        })
        assert response.status_code == 400

    def test_generate_qa_layers_without_gran_sabio_model_returns_400(self, client):
        """
        Given: QA layers provided without gran_sabio_model
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Check quality",
                    "criteria": "Be accurate",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"]
        })
        assert response.status_code == 400
        assert "gran_sabio_model" in response.json()["detail"].lower()

    def test_generate_invalid_qa_model_returns_400(self, client):
        """
        Given: Invalid qa_model name
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Check quality",
                    "criteria": "Be accurate",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["invalid-model-xyz"],
            "gran_sabio_model": "claude-sonnet-4-20250514"
        })
        assert response.status_code == 400

    def test_generate_invalid_gran_sabio_model_returns_400(self, client):
        """
        Given: Invalid gran_sabio_model name
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Check quality",
                    "criteria": "Be accurate",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"],
            "gran_sabio_model": "invalid-model-xyz"
        })
        assert response.status_code == 400

    def test_generate_word_count_enforcement_without_limits_returns_400(self, client):
        """
        Given: word_count_enforcement enabled without min/max_words
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [],
            "word_count_enforcement": {
                "enabled": True,
                "flexibility_percent": 10,
                "direction": "both",
                "severity": "important"
            }
        })
        assert response.status_code == 400
        assert "word" in response.json()["detail"].lower()

    def test_generate_invalid_word_count_config_returns_422(self, client):
        """
        Given: Invalid word_count_enforcement configuration
        When: POST /generate is called
        Then: Returns 422 Unprocessable Entity (Pydantic validation error)
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [],
            "min_words": 500,
            "max_words": 1000,
            "word_count_enforcement": {
                "enabled": True,
                "flexibility_percent": 150,  # Invalid: > 100 (Pydantic le=100)
                "direction": "both",
                "severity": "important"
            }
        })
        # Pydantic validation catches this before endpoint logic
        assert response.status_code == 422

    def test_generate_project_id_too_long_returns_400(self, client):
        """
        Given: project_id longer than 128 characters
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        long_project_id = "x" * 129
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [],
            "project_id": long_project_id
        })
        assert response.status_code == 400
        assert "project_id" in response.json()["detail"].lower()

    def test_generate_context_documents_without_username_returns_400(self, client):
        """
        Given: context_documents provided without username
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [],
            "context_documents": [
                {"username": "testuser", "upload_id": "test-upload-123"}
            ]
        })
        assert response.status_code == 400
        assert "username" in response.json()["detail"].lower()

    def test_generate_images_without_username_returns_400(self, client):
        """
        Given: images provided without username
        When: POST /generate is called
        Then: Returns 400 Bad Request
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "generator_model": "gpt-4o",
            "qa_layers": [],
            "images": [
                {"username": "testuser", "upload_id": "test-image-123"}
            ]
        })
        assert response.status_code == 400
        assert "username" in response.json()["detail"].lower()

    def test_generate_valid_request_no_qa_returns_session_id(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: Valid request without QA layers
        When: POST /generate is called
        Then: Returns session_id and status initialized
        """
        response = client.post("/generate", json=valid_generate_request)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "initialized"

    def test_generate_valid_request_returns_project_id(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: Valid request
        When: POST /generate is called
        Then: Returns project_id (equals session_id when not provided)
        """
        response = client.post("/generate", json=valid_generate_request)
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        # When not provided, project_id should equal session_id
        assert data["project_id"] == data["session_id"]

    def test_generate_with_explicit_project_id(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: Request with explicit project_id
        When: POST /generate is called
        Then: Returns the provided project_id
        """
        valid_generate_request["project_id"] = "my-custom-project"
        response = client.post("/generate", json=valid_generate_request)
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "my-custom-project"

    def test_generate_returns_recommended_timeout(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: Valid request
        When: POST /generate is called
        Then: Returns recommended_timeout_seconds
        """
        response = client.post("/generate", json=valid_generate_request)
        assert response.status_code == 200
        data = response.json()
        assert "recommended_timeout_seconds" in data
        assert isinstance(data["recommended_timeout_seconds"], int)

    def test_generate_preflight_reject_returns_rejected_status(
        self, client, valid_generate_request_with_qa, mock_preflight_reject
    ):
        """
        Given: Request that fails preflight validation
        When: POST /generate is called
        Then: Returns status 'rejected' with preflight_feedback
        """
        response = client.post("/generate", json=valid_generate_request_with_qa)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["session_id"] is None
        assert "preflight_feedback" in data

    def test_generate_accepts_model_alias_for_generator_model(
        self, client, mock_preflight_proceed
    ):
        """
        Given: Generator model using 'model' alias
        When: POST /generate is called
        Then: Request is accepted
        """
        response = client.post("/generate", json={
            "prompt": "Write a comprehensive article about software testing",
            "model": "gpt-4o",  # Using alias
            "qa_layers": []
        })
        assert response.status_code == 200


# ============================================================================
# Status/Result Endpoint Tests (10 tests)
# ============================================================================

class TestStatusEndpoint:
    """Tests for GET /status/{session_id} endpoint."""

    def test_status_unknown_session_returns_404(self, client):
        """
        Given: Non-existent session_id
        When: GET /status/{session_id} is called
        Then: Returns 404 Not Found
        """
        fake_session_id = str(uuid.uuid4())
        response = client.get(f"/status/{fake_session_id}")
        assert response.status_code == 404

    def test_status_valid_session_returns_200(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A valid session
        When: GET /status/{session_id} is called
        Then: Returns 200 OK
        """
        # Create a session first
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        # Check status
        response = client.get(f"/status/{session_id}")
        assert response.status_code == 200

    def test_status_includes_session_id(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A valid session
        When: GET /status/{session_id} is called
        Then: Response includes session_id
        """
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        response = client.get(f"/status/{session_id}")
        data = response.json()
        assert data["session_id"] == session_id

    def test_status_includes_status_field(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A valid session
        When: GET /status/{session_id} is called
        Then: Response includes status field
        """
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        response = client.get(f"/status/{session_id}")
        data = response.json()
        assert "status" in data

    def test_status_includes_iteration_info(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A valid session
        When: GET /status/{session_id} is called
        Then: Response includes current_iteration and max_iterations
        """
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        response = client.get(f"/status/{session_id}")
        data = response.json()
        assert "current_iteration" in data
        assert "max_iterations" in data


class TestResultEndpoint:
    """Tests for GET /result/{session_id} endpoint."""

    def test_result_unknown_session_returns_404(self, client):
        """
        Given: Non-existent session_id
        When: GET /result/{session_id} is called
        Then: Returns 404 Not Found
        """
        fake_session_id = str(uuid.uuid4())
        response = client.get(f"/result/{fake_session_id}")
        assert response.status_code == 404

    def test_result_in_progress_returns_202(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A session still in progress
        When: GET /result/{session_id} is called
        Then: Returns 202 Accepted with Retry-After header
        """
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        # Immediately check result (should be in progress)
        response = client.get(f"/result/{session_id}")
        assert response.status_code == 202
        assert "Retry-After" in response.headers


class TestStopEndpoint:
    """Tests for POST /stop/{session_id} endpoint."""

    def test_stop_unknown_session_returns_404(self, client):
        """
        Given: Non-existent session_id
        When: POST /stop/{session_id} is called
        Then: Returns 404 Not Found
        """
        fake_session_id = str(uuid.uuid4())
        response = client.post(f"/stop/{fake_session_id}")
        assert response.status_code == 404

    def test_stop_valid_session_returns_stopped(
        self, client, valid_generate_request, mock_preflight_proceed
    ):
        """
        Given: A valid session
        When: POST /stop/{session_id} is called
        Then: Returns stopped confirmation
        """
        gen_response = client.post("/generate", json=valid_generate_request)
        session_id = gen_response.json()["session_id"]

        response = client.post(f"/stop/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["stopped"] is True
        assert data["session_id"] == session_id


# ============================================================================
# Project Endpoint Tests (bonus)
# ============================================================================

class TestProjectEndpoints:
    """Tests for project management endpoints."""

    def test_project_new_returns_project_id(self, client):
        """
        Given: A request to allocate new project_id
        When: POST /project/new is called
        Then: Returns a new project_id
        """
        response = client.post("/project/new")
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert len(data["project_id"]) > 0

    def test_project_new_with_custom_id(self, client):
        """
        Given: A request with custom project_id
        When: POST /project/new is called
        Then: Returns the provided project_id
        """
        response = client.post("/project/new", json={"project_id": "my-custom-id"})
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "my-custom-id"

    def test_project_start_invalid_id_returns_400(self, client):
        """
        Given: Empty project_id
        When: POST /project/start/{project_id} is called
        Then: Returns 400 Bad Request
        """
        # Test with empty string (path won't match, so test with very long id)
        long_id = "x" * 129
        response = client.post(f"/project/start/{long_id}")
        assert response.status_code == 400

    def test_project_stop_returns_cancelled_status(self, client):
        """
        Given: A valid project_id
        When: POST /project/stop/{project_id} is called
        Then: Returns cancelled status
        """
        # First create a project
        new_response = client.post("/project/new")
        project_id = new_response.json()["project_id"]

        # Stop it
        response = client.post(f"/project/stop/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["project_id"] == project_id

    def test_project_status_invalid_id_returns_400(self, client):
        """
        Given: Invalid project_id (too long)
        When: GET /status/project/{project_id} is called
        Then: Returns 400 Bad Request
        """
        long_id = "x" * 129
        response = client.get(f"/status/project/{long_id}")
        assert response.status_code == 400
