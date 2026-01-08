"""
Integration tests for error scenarios and cancellation flows.

Tests error handling throughout the application:
- Request validation errors
- Service unavailability
- Cancellation flows
- Preflight rejection
- API error responses
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Note: Fixtures inherited from conftest.py:
# - bypass_ip_filter (autouse)
# - base_test_client
# - valid_generation_request


# ============================================================================
# Request Validation Error Tests
# ============================================================================

class TestValidationErrors:
    """Tests for request validation errors."""

    def test_empty_prompt_returns_422(self, base_test_client):
        """
        Given: Empty prompt
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "",
            "generator_model": "gpt-4o"
        })
        assert response.status_code == 422

    def test_short_prompt_returns_422(self, base_test_client):
        """
        Given: Prompt shorter than minimum length
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Hi",
            "generator_model": "gpt-4o"
        })
        assert response.status_code == 422

    def test_invalid_temperature_returns_422(self, base_test_client):
        """
        Given: Temperature outside valid range
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about temperature validation in systems.",
            "generator_model": "gpt-4o",
            "temperature": 3.0  # Invalid: max is 2.0
        })
        assert response.status_code == 422

    def test_negative_max_iterations_returns_422(self, base_test_client):
        """
        Given: Negative max_iterations
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about iterations in generation.",
            "generator_model": "gpt-4o",
            "max_iterations": -1
        })
        assert response.status_code == 422

    def test_invalid_content_type_returns_422(self, base_test_client):
        """
        Given: Invalid content_type
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about content type validation.",
            "generator_model": "gpt-4o",
            "content_type": "invalid_type_xyz"
        })
        assert response.status_code == 422

    def test_qa_layer_invalid_min_score_returns_422(self, base_test_client):
        """
        Given: QA layer with min_score > 10
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about QA validation.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Test",
                    "description": "Test layer",
                    "criteria": "Test criteria",
                    "min_score": 15.0,  # Invalid: max is 10
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"],
            "gran_sabio_model": "gpt-4o"
        })
        assert response.status_code == 422


# ============================================================================
# Preflight Rejection Tests
# ============================================================================

class TestPreflightRejection:
    """Tests for preflight validation rejection."""

    def test_preflight_rejection_returns_rejected_status(self, rejection_test_client):
        """
        Given: Request that fails preflight validation
        When: POST /generate
        Then: Returns rejected status with feedback
        """
        response = rejection_test_client.post("/generate", json={
            "prompt": "Write a fictional story about dragons with magic powers.",
            "generator_model": "gpt-4o",
            "content_type": "creative",
            "max_iterations": 3,
            "qa_layers": [
                {
                    "name": "Factual Accuracy",
                    "description": "Check for factual errors",
                    "criteria": "Content must be historically accurate",
                    "min_score": 8.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"],
            "gran_sabio_model": "gpt-4o",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["session_id"] is None

    def test_preflight_rejection_includes_feedback(self, rejection_test_client):
        """
        Given: Request that fails preflight
        When: POST /generate
        Then: Response includes preflight_feedback
        """
        response = rejection_test_client.post("/generate", json={
            "prompt": "Write a fictional story about time travel.",
            "generator_model": "gpt-4o",
            "content_type": "creative",
            "qa_layers": [
                {
                    "name": "Accuracy",
                    "description": "Accuracy check",
                    "criteria": "Must be factual",
                    "min_score": 8.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"],
            "gran_sabio_model": "gpt-4o",
        })
        data = response.json()
        assert "preflight_feedback" in data


# ============================================================================
# Cancellation Flow Tests
# ============================================================================

class TestCancellationFlows:
    """Tests for various cancellation scenarios."""

    def test_cancel_initializing_session(self, base_test_client, valid_generation_request):
        """
        Given: Session in initializing state
        When: POST /stop/{session_id}
        Then: Session is cancelled
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        stop_response = base_test_client.post(f"/stop/{session_id}")
        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["session_id"] == session_id

    def test_cancel_via_project_stop(self, base_test_client, valid_generation_request):
        """
        Given: Active sessions in a project
        When: POST /project/stop/{project_id}
        Then: All sessions are cancelled
        """
        project_id = "cancel-project-test"
        valid_generation_request["project_id"] = project_id
        base_test_client.post("/generate", json=valid_generation_request)

        stop_response = base_test_client.post(f"/project/stop/{project_id}")
        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["status"] == "cancelled"

    def test_reactivate_cancelled_project(self, base_test_client, valid_generation_request):
        """
        Given: Cancelled project
        When: POST /project/start/{project_id}
        Then: Project is reactivated
        """
        project_id = "reactivate-test"
        valid_generation_request["project_id"] = project_id

        # Create and cancel
        base_test_client.post("/generate", json=valid_generation_request)
        base_test_client.post(f"/project/stop/{project_id}")

        # Reactivate
        start_response = base_test_client.post(f"/project/start/{project_id}")
        assert start_response.status_code == 200
        data = start_response.json()
        assert data["status"] == "reactivated"
        assert data["was_cancelled"] is True

    def test_generate_after_project_reactivation(self, base_test_client, valid_generation_request):
        """
        Given: Reactivated project
        When: POST /generate with project_id
        Then: Request succeeds (not 403)
        """
        project_id = "generate-after-reactivate"
        valid_generation_request["project_id"] = project_id

        # Create, cancel, and reactivate
        base_test_client.post("/generate", json=valid_generation_request)
        base_test_client.post(f"/project/stop/{project_id}")
        base_test_client.post(f"/project/start/{project_id}")

        # New generation should work
        new_request = valid_generation_request.copy()
        new_request["prompt"] = "New prompt after reactivation for testing purposes and validation."
        response = base_test_client.post("/generate", json=new_request)
        assert response.status_code == 200


# ============================================================================
# API Error Response Tests
# ============================================================================

class TestAPIErrorResponses:
    """Tests for API error response format."""

    def test_404_includes_detail(self, base_test_client):
        """
        Given: Non-existent resource
        When: GET /status/{unknown}
        Then: 404 includes detail message
        """
        response = base_test_client.get("/status/nonexistent-session")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_400_includes_detail(self, base_test_client):
        """
        Given: Invalid request
        When: POST with missing required field
        Then: 400 includes detail message
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Test prompt for error response validation purposes."
            # Missing generator_model
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_422_includes_validation_details(self, base_test_client):
        """
        Given: Invalid field value
        When: POST /generate
        Then: 422 includes validation error details
        """
        response = base_test_client.post("/generate", json={
            "prompt": "x",  # Too short
            "generator_model": "gpt-4o"
        })
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_403_cancelled_project_includes_detail(self, base_test_client, valid_generation_request):
        """
        Given: Cancelled project
        When: POST /generate with that project_id
        Then: 403 includes helpful detail
        """
        project_id = "error-403-test"
        valid_generation_request["project_id"] = project_id
        base_test_client.post("/generate", json=valid_generation_request)
        base_test_client.post(f"/project/stop/{project_id}")

        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 403
        data = response.json()
        assert "detail" in data
        assert "cancelled" in data["detail"].lower()


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_generate_max_project_id_length(self, base_test_client, valid_generation_request):
        """
        Given: project_id at max length (128 chars)
        When: POST /generate
        Then: Request succeeds
        """
        valid_generation_request["project_id"] = "a" * 128
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200

    def test_generate_project_id_too_long(self, base_test_client, valid_generation_request):
        """
        Given: project_id exceeds max length
        When: POST /generate
        Then: Returns 400
        """
        valid_generation_request["project_id"] = "a" * 129
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 400

    def test_stop_same_session_twice(self, base_test_client, valid_generation_request):
        """
        Given: Session already stopped
        When: POST /stop/{session_id} again
        Then: Returns success with appropriate message
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # First stop
        base_test_client.post(f"/stop/{session_id}")
        # Second stop
        response = base_test_client.post(f"/stop/{session_id}")
        assert response.status_code == 200

    def test_project_status_empty_project(self, base_test_client):
        """
        Given: Project with no sessions
        When: GET /status/project/{project_id}
        Then: Returns empty sessions list
        """
        response = base_test_client.get("/status/project/empty-project-xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["summary"]["total_sessions"] == 0

    def test_concurrent_project_ids_unique(self, base_test_client):
        """
        Given: Multiple /project/new requests
        When: No project_id supplied
        Then: Each returns unique ID
        """
        ids = set()
        for _ in range(5):
            response = base_test_client.post("/project/new")
            assert response.status_code == 200
            ids.add(response.json()["project_id"])
        assert len(ids) == 5  # All unique


# ============================================================================
# QA Configuration Error Tests
# ============================================================================

class TestQAConfigurationErrors:
    """Tests for QA configuration validation."""

    def test_qa_layers_without_gran_sabio_model(self, base_test_client):
        """
        Given: QA layers without gran_sabio_model
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about QA configuration validation.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Test",
                    "criteria": "Test",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"]
            # Missing gran_sabio_model
        })
        assert response.status_code == 400

    def test_empty_qa_models_with_qa_layers(self, base_test_client):
        """
        Given: QA layers with empty qa_models list
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about empty QA models list.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Test",
                    "criteria": "Test",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": [],
            "gran_sabio_model": "gpt-4o"
        })
        assert response.status_code == 400

    def test_invalid_qa_model_returns_400(self, base_test_client):
        """
        Given: Unknown model in qa_models
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about invalid QA models.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Test",
                    "criteria": "Test",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["nonexistent-model-xyz"],
            "gran_sabio_model": "gpt-4o"
        })
        assert response.status_code == 400

    def test_invalid_gran_sabio_model_returns_400(self, base_test_client):
        """
        Given: Unknown gran_sabio_model
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Write a test article about invalid Gran Sabio model.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Test",
                    "criteria": "Test",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": ["gpt-4o"],
            "gran_sabio_model": "nonexistent-gransabio-model"
        })
        assert response.status_code == 400
