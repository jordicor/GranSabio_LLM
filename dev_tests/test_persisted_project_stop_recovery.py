"""Regression tests for debugger telemetry and runtime-only project stops."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core import generation_routes
from core.cancellation import cancellation_registry
from debug_logger import DebugLogger


async def _runtime_unknown(_project_id: str) -> bool:
    return False


async def _runtime_known(_project_id: str) -> bool:
    return True


def _request(**query_params: str) -> SimpleNamespace:
    return SimpleNamespace(query_params=query_params)


async def _unexpected_hard_stop(project_id: str) -> dict:
    raise AssertionError(f"hard stop must not run for {project_id}")


async def _unexpected_pause(project_id: str) -> dict:
    raise AssertionError(f"pause must not run for {project_id}")


@pytest.mark.asyncio
async def test_runtime_known_stop_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def _hard_stop(project_id: str) -> dict:
        calls.append(project_id)
        return {"project_id": project_id, "status": "cancelled"}

    monkeypatch.setattr(
        generation_routes,
        "_project_runtime_known",
        _runtime_known,
    )
    monkeypatch.setattr(
        generation_routes,
        "hard_stop_project_runtime",
        _hard_stop,
    )

    result = await generation_routes.stop_project("live-project", _request())

    assert result["status"] == "cancelled"
    assert calls == ["live-project"]


@pytest.mark.asyncio
async def test_runtime_unknown_stop_never_mutates_debug_telemetry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = "past-runtime-project"
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "debugger.db"),
    )
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "past-runtime-session",
            request_payload={
                "request_name": "past runtime",
                "prompt": "private prompt must not act as recovery proof",
            },
            status="generating",
            project_id=project_id,
        )
        before = await logger_instance.get_session_details(
            "past-runtime-session"
        )
        monkeypatch.setattr(
            generation_routes,
            "_project_runtime_known",
            _runtime_unknown,
        )
        monkeypatch.setattr(
            generation_routes,
            "hard_stop_project_runtime",
            _unexpected_hard_stop,
        )

        with pytest.raises(HTTPException) as caught:
            await generation_routes.stop_project(
                project_id,
                _request(session_id="past-runtime-session"),
            )

        assert caught.value.status_code == 404
        assert caught.value.detail == "Project not found"
        after = await logger_instance.get_session_details(
            "past-runtime-session"
        )
        assert after == before
        assert await cancellation_registry.has_project_runtime(project_id) is False
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_concurrent_unknown_stops_are_idempotent_and_create_no_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id = "concurrent-past-project"
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "debugger.db"),
    )
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "concurrent-past-session",
            request_payload={"request_name": "past runtime"},
            status="qa_evaluation",
            project_id=project_id,
        )
        before = await logger_instance.get_session_details(
            "concurrent-past-session"
        )
        monkeypatch.setattr(
            generation_routes,
            "_project_runtime_known",
            _runtime_unknown,
        )
        monkeypatch.setattr(
            generation_routes,
            "hard_stop_project_runtime",
            _unexpected_hard_stop,
        )

        results = await asyncio.gather(
            generation_routes.stop_project(
                project_id,
                _request(session_id="concurrent-past-session"),
            ),
            generation_routes.stop_project(
                project_id,
                _request(session_id="concurrent-past-session"),
            ),
            return_exceptions=True,
        )

        assert all(
            isinstance(result, HTTPException) and result.status_code == 404
            for result in results
        )
        after = await logger_instance.get_session_details(
            "concurrent-past-session"
        )
        assert after == before
        assert await cancellation_registry.has_project_runtime(project_id) is False
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_cleanup_ttl_deletes_terminal_and_nonterminal_telemetry(
    tmp_path,
) -> None:
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "debugger.db"),
    )
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "old-terminal",
            request_payload={},
            status="completed",
            project_id="old-project",
        )
        await logger_instance.record_session_start(
            "old-nonterminal",
            request_payload={},
            status="generating",
            project_id="old-project",
        )
        await logger_instance.record_event(
            "old-nonterminal",
            event_type="old-event",
            payload={"state": "observed"},
        )
        await logger_instance._pool.execute(
            """
            UPDATE sessions
            SET created_at = datetime('now', '-30 days'),
                updated_at = datetime('now', '-30 days')
            WHERE session_id IN ('old-terminal', 'old-nonterminal')
            """
        )
        await logger_instance._pool.commit()

        result = await logger_instance.cleanup_old_sessions(retention_days=7)

        assert result["deleted_sessions"] == 2
        assert result["deleted_events"] == 1
        assert await logger_instance.get_session_details("old-terminal") is None
        assert await logger_instance.get_session_details("old-nonterminal") is None
        assert await logger_instance.get_session_events("old-nonterminal") == []
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_initialization_enforces_configured_telemetry_ttl(
    tmp_path,
) -> None:
    """Startup cleanup only exists when retention is configured (C13.1).

    Without DEBUGGER_RETENTION_DAYS the same restart keeps the row: telemetry is
    never deleted by default. The sibling assertion below is what makes the
    difference between the two modes explicit.
    """
    db_path = str(tmp_path / "debugger.db")
    first_logger = DebugLogger(enabled=True, db_path=db_path)
    await first_logger.initialize()
    try:
        await first_logger.record_session_start(
            "stale-after-restart",
            request_payload={},
            status="generating",
            project_id="past-project",
        )
        await first_logger._pool.execute(
            """
            UPDATE sessions
            SET updated_at = datetime('now', '-30 days')
            WHERE session_id = 'stale-after-restart'
            """
        )
        await first_logger._pool.commit()
    finally:
        await first_logger.close()

    kept_logger = DebugLogger(enabled=True, db_path=db_path)
    await kept_logger.initialize()
    try:
        assert (
            await kept_logger.get_session_details("stale-after-restart")
            is not None
        )
    finally:
        await kept_logger.close()

    restarted_logger = DebugLogger(
        enabled=True,
        db_path=db_path,
        retention_days=7,
    )
    await restarted_logger.initialize()
    try:
        assert (
            await restarted_logger.get_session_details("stale-after-restart")
            is None
        )
    finally:
        await restarted_logger.close()


@pytest.mark.asyncio
async def test_pause_does_not_use_persisted_project_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "debugger.db"),
    )
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "persisted-session",
            request_payload={},
            status="generating",
            project_id="persisted-project",
        )
        monkeypatch.setattr(
            generation_routes,
            "_project_runtime_known",
            _runtime_unknown,
        )
        monkeypatch.setattr(
            generation_routes,
            "pause_project_runtime",
            _unexpected_pause,
        )

        with pytest.raises(HTTPException) as caught:
            await generation_routes.pause_project(
                "persisted-project",
                _request(),
            )

        assert caught.value.status_code == 404
    finally:
        await logger_instance.close()
