"""
Integration tests for the full content generation workflow.

Tests the complete lifecycle:
- Project allocation and management
- Generation request initiation
- Session status tracking
- Result retrieval
- Cancellation flows
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Note: Fixtures are inherited from conftest.py:
# - bypass_ip_filter (autouse)
# - base_test_client
# - valid_generation_request
# - generation_request_with_qa


# ============================================================================
# Project Allocation Tests
# ============================================================================

class TestProjectAllocation:
    """Tests for project ID allocation and management."""

    def test_allocate_new_project_id(self, base_test_client):
        """
        Given: No project_id supplied
        When: POST /project/new
        Then: Returns a new unique project_id
        """
        response = base_test_client.post("/project/new")
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        assert len(data["project_id"]) == 32  # UUID hex format

    def test_allocate_with_custom_project_id(self, base_test_client):
        """
        Given: Custom project_id supplied
        When: POST /project/new
        Then: Returns the supplied project_id
        """
        custom_id = "my-custom-project-123"
        response = base_test_client.post(
            "/project/new",
            json={"project_id": custom_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == custom_id

    def test_reuse_existing_project_id(self, base_test_client):
        """
        Given: Previously allocated project_id
        When: POST /project/new with same ID
        Then: Returns the same project_id (reuse)
        """
        custom_id = "reusable-project-456"
        # First allocation
        response1 = base_test_client.post(
            "/project/new",
            json={"project_id": custom_id}
        )
        assert response1.status_code == 200

        # Second allocation with same ID
        response2 = base_test_client.post(
            "/project/new",
            json={"project_id": custom_id}
        )
        assert response2.status_code == 200
        assert response2.json()["project_id"] == custom_id

    def test_project_start_activates_project(self, base_test_client):
        """
        Given: A project_id
        When: POST /project/start/{project_id}
        Then: Project is activated
        """
        project_id = "test-start-project"
        response = base_test_client.post(f"/project/start/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert data["status"] in ["already_active", "reactivated"]

    def test_project_stop_cancels_project(self, base_test_client):
        """
        Given: An active project
        When: POST /project/stop/{project_id}
        Then: Project is cancelled
        """
        project_id = "test-stop-project"
        # First activate the project
        base_test_client.post(f"/project/start/{project_id}")

        # Then stop it
        response = base_test_client.post(f"/project/stop/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert data["status"] == "cancelled"

    def test_project_stop_invalid_id_returns_400(self, base_test_client):
        """
        Given: Invalid project_id (empty or too long)
        When: POST /project/stop/{project_id}
        Then: Returns 400
        """
        # Too long project ID (>128 chars)
        long_id = "a" * 200
        response = base_test_client.post(f"/project/stop/{long_id}")
        assert response.status_code == 400


# ============================================================================
# Generate Endpoint Tests
# ============================================================================

class TestGenerateEndpoint:
    """Tests for the /generate endpoint."""

    def test_generate_returns_session_id(self, base_test_client, valid_generation_request):
        """
        Given: Valid generation request
        When: POST /generate
        Then: Returns session_id and initialized status
        """
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "initialized"

    def test_generate_returns_project_id(self, base_test_client, valid_generation_request):
        """
        Given: Valid generation request without explicit project_id
        When: POST /generate
        Then: Returns project_id equal to session_id
        """
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200
        data = response.json()
        assert "project_id" in data
        # When no explicit project_id, session_id becomes project_id
        assert data["project_id"] == data["session_id"]

    def test_generate_with_explicit_project_id(self, base_test_client, valid_generation_request):
        """
        Given: Request with explicit project_id
        When: POST /generate
        Then: Returns the provided project_id
        """
        valid_generation_request["project_id"] = "explicit-project-123"
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "explicit-project-123"

    def test_generate_missing_prompt_returns_422(self, base_test_client):
        """
        Given: Request without prompt
        When: POST /generate
        Then: Returns 422 validation error
        """
        response = base_test_client.post("/generate", json={
            "generator_model": "gpt-4o",
            "content_type": "article"
        })
        assert response.status_code == 422

    def test_generate_missing_model_returns_400(self, base_test_client):
        """
        Given: Request without generator_model
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Test prompt for generation that is long enough to pass validation.",
            "content_type": "article"
        })
        assert response.status_code == 400
        assert "generator_model" in response.json()["detail"].lower()

    def test_generate_invalid_model_returns_400(self, base_test_client, valid_generation_request):
        """
        Given: Request with unknown model
        When: POST /generate
        Then: Returns 400
        """
        valid_generation_request["generator_model"] = "nonexistent-model-xyz"
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 400

    def test_generate_qa_layers_without_qa_models_returns_400(self, base_test_client):
        """
        Given: Request with QA layers but no QA models
        When: POST /generate
        Then: Returns 400
        """
        response = base_test_client.post("/generate", json={
            "prompt": "Test prompt for generation that is long enough to pass validation.",
            "generator_model": "gpt-4o",
            "qa_layers": [
                {
                    "name": "Quality",
                    "description": "Quality check",
                    "criteria": "Check quality",
                    "min_score": 7.0,
                    "order": 1
                }
            ],
            "qa_models": []  # Empty but QA layers present
        })
        assert response.status_code == 400

    def test_generate_with_word_count_limits(self, base_test_client, valid_generation_request):
        """
        Given: Request with word count limits
        When: POST /generate
        Then: Processes successfully
        """
        valid_generation_request["min_words"] = 500
        valid_generation_request["max_words"] = 1000
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200

    def test_generate_returns_recommended_timeout(self, base_test_client, valid_generation_request):
        """
        Given: Valid generation request
        When: POST /generate
        Then: Returns recommended_timeout_seconds
        """
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200
        data = response.json()
        assert "recommended_timeout_seconds" in data
        assert data["recommended_timeout_seconds"] > 0

    def test_generate_cancelled_project_returns_403(self, base_test_client, valid_generation_request):
        """
        Given: A cancelled project
        When: POST /generate with that project_id
        Then: Returns 403
        """
        project_id = "cancelled-project-test"
        # Cancel the project first
        base_test_client.post(f"/project/stop/{project_id}")

        # Try to generate with cancelled project
        valid_generation_request["project_id"] = project_id
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 403
        assert "cancelled" in response.json()["detail"].lower()


# ============================================================================
# Session Status Tests
# ============================================================================

class TestSessionStatus:
    """Tests for session status tracking."""

    def test_status_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /status/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/status/nonexistent-session-123")
        assert response.status_code == 404

    def test_status_returns_session_info(self, base_test_client, valid_generation_request):
        """
        Given: Valid session
        When: GET /status/{session_id}
        Then: Returns session information
        """
        # Create a session first
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Check status
        status_response = base_test_client.get(f"/status/{session_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["session_id"] == session_id
        assert "status" in data
        assert "current_iteration" in data
        assert "max_iterations" in data

    def test_status_includes_project_id(self, base_test_client, valid_generation_request):
        """
        Given: Session with project_id
        When: GET /status/{session_id}
        Then: Returns project_id in response
        """
        valid_generation_request["project_id"] = "status-test-project"
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        status_response = base_test_client.get(f"/status/{session_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["project_id"] == "status-test-project"

    def test_status_includes_gran_sabio_escalations(self, base_test_client, valid_generation_request):
        """
        Given: Valid session
        When: GET /status/{session_id}
        Then: Includes gran_sabio_escalations info
        """
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        status_response = base_test_client.get(f"/status/{session_id}")
        data = status_response.json()
        assert "gran_sabio_escalations" in data


# ============================================================================
# Result Retrieval Tests
# ============================================================================

class TestResultRetrieval:
    """Tests for result retrieval."""

    def test_result_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: GET /result/{session_id}
        Then: Returns 404
        """
        response = base_test_client.get("/result/nonexistent-session-123")
        assert response.status_code == 404

    def test_result_in_progress_returns_202(self, base_test_client, valid_generation_request):
        """
        Given: Session still in progress
        When: GET /result/{session_id}
        Then: Returns 202 with Retry-After header
        """
        # Create a session
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Immediately check result (likely still in progress)
        result_response = base_test_client.get(f"/result/{session_id}")
        # Could be 200 (if fast) or 202 (if still running)
        assert result_response.status_code in [200, 202]
        if result_response.status_code == 202:
            assert "Retry-After" in result_response.headers


# ============================================================================
# Project Status Tests
# ============================================================================

class TestProjectStatus:
    """Tests for project-level status tracking."""

    def test_project_status_returns_structure(self, base_test_client, valid_generation_request):
        """
        Given: Project with sessions
        When: GET /status/project/{project_id}
        Then: Returns project status structure
        """
        project_id = "project-status-test"
        valid_generation_request["project_id"] = project_id
        base_test_client.post("/generate", json=valid_generation_request)

        response = base_test_client.get(f"/status/project/{project_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == project_id
        assert "status" in data
        assert "sessions" in data
        assert "summary" in data

    def test_project_status_summary_counts(self, base_test_client, valid_generation_request):
        """
        Given: Project with sessions
        When: GET /status/project/{project_id}
        Then: Summary includes session counts
        """
        project_id = "project-summary-test"
        valid_generation_request["project_id"] = project_id
        base_test_client.post("/generate", json=valid_generation_request)

        response = base_test_client.get(f"/status/project/{project_id}")
        data = response.json()
        summary = data["summary"]
        assert "total_sessions" in summary
        assert "active_sessions" in summary
        assert "completed_sessions" in summary

    def test_project_status_invalid_id_returns_400(self, base_test_client):
        """
        Given: Invalid project_id (too long)
        When: GET /status/project/{project_id}
        Then: Returns 400
        """
        long_id = "a" * 200
        response = base_test_client.get(f"/status/project/{long_id}")
        assert response.status_code == 400

    def test_project_status_idle_when_no_sessions(self, base_test_client):
        """
        Given: Project with no sessions
        When: GET /status/project/{project_id}
        Then: Returns idle status
        """
        response = base_test_client.get("/status/project/nonexistent-project")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "idle"
        assert data["summary"]["total_sessions"] == 0


# ============================================================================
# Stop/Cancel Session Tests
# ============================================================================

class TestStopSession:
    """Tests for session cancellation."""

    def test_stop_unknown_session_returns_404(self, base_test_client):
        """
        Given: Unknown session_id
        When: POST /stop/{session_id}
        Then: Returns 404
        """
        response = base_test_client.post("/stop/nonexistent-session-123")
        assert response.status_code == 404

    def test_stop_active_session_returns_success(self, base_test_client, valid_generation_request):
        """
        Given: Active session
        When: POST /stop/{session_id}
        Then: Returns success with cancelled status
        """
        # Create a session
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Stop it
        stop_response = base_test_client.post(f"/stop/{session_id}")
        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["session_id"] == session_id
        # May or may not be stopped depending on race condition
        assert "stopped" in data or "message" in data

    def test_stop_already_finished_session(self, base_test_client, valid_generation_request):
        """
        Given: Already finished session
        When: POST /stop/{session_id}
        Then: Returns message indicating already finished
        """
        # Create and stop a session
        gen_response = base_test_client.post("/generate", json=valid_generation_request)
        session_id = gen_response.json()["session_id"]

        # Stop it twice
        base_test_client.post(f"/stop/{session_id}")
        second_stop = base_test_client.post(f"/stop/{session_id}")

        assert second_stop.status_code == 200
        data = second_stop.json()
        # Already cancelled or finished
        assert "message" in data


# ============================================================================
# Word Count Enforcement Tests
# ============================================================================

class TestWordCountEnforcement:
    """Tests for word count enforcement in generation."""

    def test_word_count_enforcement_config_validation(self, base_test_client, valid_generation_request):
        """
        Given: Request with word count enforcement enabled but no limits
        When: POST /generate
        Then: Returns 400
        """
        valid_generation_request["word_count_enforcement"] = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        # No min_words or max_words set
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 400

    def test_word_count_enforcement_with_limits(self, base_test_client, valid_generation_request):
        """
        Given: Request with word count enforcement and limits
        When: POST /generate
        Then: Processes successfully
        """
        valid_generation_request["word_count_enforcement"] = {
            "enabled": True,
            "flexibility_percent": 10,
            "direction": "both",
            "severity": "important"
        }
        valid_generation_request["min_words"] = 500
        valid_generation_request["max_words"] = 1000
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 200

    def test_word_count_enforcement_invalid_flexibility(self, base_test_client, valid_generation_request):
        """
        Given: Request with invalid flexibility_percent (>100)
        When: POST /generate
        Then: Returns 422 validation error
        """
        valid_generation_request["word_count_enforcement"] = {
            "enabled": True,
            "flexibility_percent": 150,  # Invalid: >100
            "direction": "both",
            "severity": "important"
        }
        valid_generation_request["min_words"] = 500
        valid_generation_request["max_words"] = 1000
        response = base_test_client.post("/generate", json=valid_generation_request)
        assert response.status_code == 422
