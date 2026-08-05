"""Terminal generation failures retain their machine-readable root cause."""
from __future__ import annotations

from datetime import datetime

import pytest

from ai_service import GenerationOutputTruncated
from client import GranSabioClientError, GranSabioGenerationRejected
from client._common import validate_result
from core import app_state
from core.app_state import _store_final_result
from core.generation_processor import _build_generation_error_result
from models import ContentRequest, GenerationStatus


def _truncated_session() -> dict:
    return {
        "status": GenerationStatus.GENERATING,
        "iterations": [],
        "last_generation_error": {
            "reason": "output_token_limit",
            "message": "human-readable detail",
        },
        "generation_finish_metadata": {
            "finish_reason": "max_tokens",
            "truncation_reason": "output_token_limit",
        },
    }


def test_rejected_empty_generation_retains_output_limit_metadata() -> None:
    session = _truncated_session()
    final_result = {"failure_reason": "review replaced the prose reason"}

    _store_final_result(
        session,
        final_result,
        final_status=GenerationStatus.REJECTED.value,
        queue_session_end=False,
    )

    assert final_result["error_type"] == "generation_error"
    assert final_result["error_code"] == "output_token_limit"
    assert final_result["output_truncated"] is True
    assert final_result["finish_reason"] == "max_tokens"
    assert session["error_type"] == "generation_error"
    assert session["error_code"] == "output_token_limit"


def test_failed_empty_generation_retains_output_limit_metadata() -> None:
    session = _truncated_session()
    final_result = {"failure_reason": "No candidate was generated"}

    _store_final_result(
        session,
        final_result,
        final_status=GenerationStatus.FAILED.value,
        queue_session_end=False,
    )

    assert final_result["error_type"] == "generation_error"
    assert final_result["error_code"] == "output_token_limit"
    assert final_result["output_truncated"] is True
    assert final_result["finish_reason"] == "max_tokens"


def test_falsy_terminal_metadata_is_replaced_by_output_limit_cause() -> None:
    session = _truncated_session()
    session.update(
        error_type=None,
        error_code="",
        output_truncated=False,
        finish_reason=None,
    )
    final_result = {
        "error_type": None,
        "error_code": "",
        "output_truncated": False,
        "finish_reason": None,
    }

    _store_final_result(
        session,
        final_result,
        final_status=GenerationStatus.FAILED.value,
        queue_session_end=False,
    )

    for target in (final_result, session):
        assert target["error_type"] == "generation_error"
        assert target["error_code"] == "output_token_limit"
        assert target["output_truncated"] is True
        assert target["finish_reason"] == "max_tokens"


def test_live_review_truncation_retains_metadata_with_prior_iterations() -> None:
    session = {
        "current_iteration": 3,
        "iterations": [{"content": "complete but rejected candidate"}],
    }
    request = ContentRequest(
        prompt="Return compact JSON.",
        generator_model="claude-opus-4-8",
    )
    exc = GenerationOutputTruncated(
        "Gran Sabio streaming JSON response was truncated",
        provider="anthropic",
        model_id="claude-opus-4-8",
        finish_reason="max_tokens",
        max_tokens=8192,
        partial_content='{"incomplete":',
        usage={"output_truncated": True},
    )

    final_result = _build_generation_error_result(session, request, exc)

    for target in (final_result, session):
        assert target["error_type"] == "generation_error"
        assert target["error_code"] == "output_token_limit"
        assert target["output_truncated"] is True
        assert target["finish_reason"] == "max_tokens"
    assert final_result["status"] == GenerationStatus.FAILED.value
    assert (
        final_result["failure_reason"]
        == "Gran Sabio streaming JSON response was truncated"
    )


def test_successful_or_reviewable_generation_does_not_inherit_old_truncation() -> None:
    completed_session = _truncated_session()
    completed_result: dict = {}
    _store_final_result(
        completed_session,
        completed_result,
        final_status=GenerationStatus.COMPLETED.value,
        queue_session_end=False,
    )
    assert "error_code" not in completed_result

    reviewable_session = _truncated_session()
    reviewable_session["iterations"] = [{"content": "complete candidate"}]
    reviewable_result = {"failure_reason": "quality rejection"}
    _store_final_result(
        reviewable_session,
        reviewable_result,
        final_status=GenerationStatus.REJECTED.value,
        queue_session_end=False,
    )
    assert "error_code" not in reviewable_result


def test_existing_terminal_error_code_is_not_overwritten() -> None:
    session = _truncated_session()
    final_result = {"error_type": "internal_error", "error_code": "DifferentFailure"}

    _store_final_result(
        session,
        final_result,
        final_status=GenerationStatus.FAILED.value,
        queue_session_end=False,
    )

    assert final_result["error_type"] == "internal_error"
    assert final_result["error_code"] == "DifferentFailure"


@pytest.mark.asyncio
async def test_project_status_projects_structured_terminal_metadata(monkeypatch) -> None:
    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    now = datetime.utcnow()
    app_state.active_sessions["session-truncated"] = {
        "project_id": "project-truncated",
        "request_name": "Truncated",
        "status": GenerationStatus.REJECTED,
        "current_phase": "rejected",
        "created_at": now,
        "last_activity_at": now,
        "request": None,
        "final_result": {
            "status": GenerationStatus.REJECTED.value,
            "failure_reason": "No complete JSON candidate",
            "error_type": "generation_error",
            "error_code": "output_token_limit",
            "output_truncated": True,
            "finish_reason": "max_tokens",
        },
    }

    try:
        project_status = await app_state.get_project_status("project-truncated")
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    assert project_status["status"] == "rejected"
    session_status = project_status["sessions"][0]
    assert session_status["error_type"] == "generation_error"
    assert session_status["error_code"] == "output_token_limit"
    assert session_status["output_truncated"] is True
    assert session_status["finish_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_project_status_falls_back_from_empty_strings_and_preserves_false(
    monkeypatch,
) -> None:
    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    now = datetime.utcnow()
    app_state.active_sessions["session-falsy"] = {
        "project_id": "project-falsy",
        "request_name": "Falsy metadata",
        "status": GenerationStatus.FAILED,
        "current_phase": "failed",
        "created_at": now,
        "last_activity_at": now,
        "request": None,
        "error_type": "generation_error",
        "error_code": "output_token_limit",
        "output_truncated": True,
        "finish_reason": "max_tokens",
        "final_result": {
            "status": GenerationStatus.FAILED.value,
            "failure_reason": "No complete JSON candidate",
            "error_type": "",
            "error_code": "",
            "output_truncated": False,
            "finish_reason": "",
        },
    }

    try:
        project_status = await app_state.get_project_status("project-falsy")
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    session_status = project_status["sessions"][0]
    assert session_status["error_type"] == "generation_error"
    assert session_status["error_code"] == "output_token_limit"
    assert session_status["output_truncated"] is False
    assert session_status["finish_reason"] == "max_tokens"


def test_sdk_rejected_error_preserves_structured_terminal_metadata() -> None:
    result = {
        "status": "rejected",
        "failure_reason": "No complete JSON candidate",
        "error_type": "generation_error",
        "error_code": "output_token_limit",
        "output_truncated": True,
        "finish_reason": "max_tokens",
        "session_id": "session-truncated",
    }

    with pytest.raises(GranSabioGenerationRejected) as exc_info:
        validate_result(result)

    details = exc_info.value.details
    assert details["error_type"] == "generation_error"
    assert details["error_code"] == "output_token_limit"
    assert details["output_truncated"] is True
    assert details["finish_reason"] == "max_tokens"


def test_sdk_failed_error_preserves_structured_terminal_metadata() -> None:
    result = {
        "status": "failed",
        "failure_reason": "No complete JSON candidate",
        "error_type": "generation_error",
        "error_code": "output_token_limit",
        "output_truncated": True,
        "finish_reason": "max_tokens",
        "session_id": "session-truncated",
    }

    with pytest.raises(GranSabioClientError) as exc_info:
        validate_result(result)

    details = exc_info.value.details
    assert details["error_type"] == "generation_error"
    assert details["error_code"] == "output_token_limit"
    assert details["output_truncated"] is True
    assert details["finish_reason"] == "max_tokens"
