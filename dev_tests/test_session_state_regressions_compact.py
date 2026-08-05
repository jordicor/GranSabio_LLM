"""Compact regressions for session lifecycle and status-event state."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import pytest

from models import GenerationStatus


@pytest.fixture
def isolated_active_sessions():
    """Restore the process-wide session registry after each test."""
    from core import app_state

    original_sessions = dict(app_state.active_sessions)
    app_state.active_sessions.clear()
    try:
        yield app_state
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)


@pytest.mark.parametrize(
    ("completed_age", "should_expire"),
    [
        pytest.param(timedelta(minutes=1), False, id="recent-completion"),
        pytest.param(timedelta(hours=2), True, id="completion-past-ttl"),
    ],
)
async def test_terminal_session_expiry_uses_completed_at(
    isolated_active_sessions,
    monkeypatch,
    completed_age: timedelta,
    should_expire: bool,
) -> None:
    """A long-running session expires relative to completion, not creation."""
    app_state = isolated_active_sessions
    session_id = "session-terminal-generic"
    now = datetime.now()
    app_state.active_sessions[session_id] = {
        "created_at": now - timedelta(hours=4),
        "last_activity_at": now - timedelta(hours=3),
        "completed_at": now - completed_age,
        "status": GenerationStatus.COMPLETED,
    }
    monkeypatch.setattr(app_state.config, "SESSION_TIMEOUT", 3600)
    monkeypatch.setattr(app_state, "active_sessions_lock", None)

    result = await app_state.perform_session_cleanup()

    assert (session_id not in app_state.active_sessions) is should_expire
    assert result["expired_count"] == int(should_expire)


@pytest.mark.parametrize("terminal_action", ["complete", "cancel"])
def test_terminal_actions_record_completed_at(
    monkeypatch,
    terminal_action: str,
) -> None:
    """Both normal completion and cancellation establish the expiry clock."""
    from core import app_state

    session = {"status": GenerationStatus.GENERATING}
    before = datetime.now()
    if terminal_action == "complete":
        app_state.update_session_status(
            session,
            "session-finish-generic",
            GenerationStatus.COMPLETED,
        )
        expected_status = GenerationStatus.COMPLETED
    else:
        app_state.apply_session_cancelled_state(
            session,
            "session-cancel-generic",
            cancel_mode="soft",
            reason="generic cancellation request",
            hard=False,
            queue_events=False,
        )
        expected_status = GenerationStatus.CANCELLED
    after = datetime.now()

    assert session["status"] == expected_status
    assert before <= session["completed_at"] <= after


async def test_stale_nonterminal_session_is_cooperatively_cancelled(
    isolated_active_sessions,
    monkeypatch,
) -> None:
    """A non-terminal session idle beyond the defensive threshold is sealed."""
    app_state = isolated_active_sessions
    session_id = "session-stale-generic"
    old_activity = datetime.now() - timedelta(minutes=2)
    app_state.active_sessions[session_id] = {
        "created_at": old_activity,
        "last_activity_at": old_activity,
        "status": GenerationStatus.INITIALIZING,
    }
    monkeypatch.setattr(app_state.config, "SESSION_TIMEOUT", 1)
    monkeypatch.setattr(app_state.config, "REQUEST_TIMEOUT", 1)
    monkeypatch.setattr(app_state, "active_sessions_lock", None)

    try:
        result = await app_state.perform_session_cleanup()

        session = app_state.active_sessions[session_id]
        assert result["zombie_count"] == 1
        assert session["status"] == GenerationStatus.CANCELLED
        assert isinstance(session.get("completed_at"), datetime)
        assert await app_state.cancellation_registry.is_soft_cancelled(session_id)
    finally:
        await app_state.cancellation_registry.unregister_session(session_id)


@pytest.mark.parametrize(
    ("initial_status", "error", "expected_status", "blocked"),
    [
        pytest.param(
            GenerationStatus.COMPLETED,
            None,
            GenerationStatus.COMPLETED,
            True,
            id="completed-remains-terminal",
        ),
        pytest.param(
            GenerationStatus.CANCELLED,
            None,
            GenerationStatus.CANCELLED,
            True,
            id="cancelled-without-error-remains-terminal",
        ),
        pytest.param(
            GenerationStatus.CANCELLED,
            "generic teardown failure",
            GenerationStatus.FAILED,
            False,
            id="cancelled-with-error-may-fail",
        ),
    ],
)
def test_terminal_transition_guard(
    caplog,
    initial_status: GenerationStatus,
    error: str | None,
    expected_status: GenerationStatus,
    blocked: bool,
) -> None:
    """Terminal states are sticky except for a real post-cancel failure."""
    from core import app_state

    session = {"status": initial_status}
    if error is not None:
        session["error"] = error

    with caplog.at_level(logging.WARNING, logger=app_state.logger.name):
        app_state.update_session_status(
            session,
            "session-transition-generic",
            GenerationStatus.FAILED,
        )

    assert session["status"] == expected_status
    transition_warnings = [
        record
        for record in caplog.records
        if record.name == app_state.logger.name and record.levelno == logging.WARNING and "terminal session" in record.getMessage()
    ]
    assert bool(transition_warnings) is blocked
    if not blocked:
        assert isinstance(session.get("completed_at"), datetime)


async def test_project_status_burst_coalesces_into_one_tracked_task(
    monkeypatch,
) -> None:
    """Repeated updates publish only the latest snapshot for a project."""
    from core import app_state

    published: list[tuple[str, str, str, int | None]] = []

    async def fake_publish(
        project_id: str,
        session_id: str,
        event_type: str = "status_change",
        expected_project_epoch: int | None = None,
    ) -> None:
        published.append((project_id, session_id, event_type, expected_project_epoch))

    monkeypatch.setattr(app_state, "publish_project_status_event", fake_publish)
    app_state._project_status_event_pending.clear()
    app_state._project_status_event_tasks.clear()
    app_state._project_status_event_task_set.clear()
    session = {"project_id": "project-generic", "project_epoch": 3}

    try:
        app_state.queue_project_status_event(session, "session-one")
        app_state.queue_project_status_event(
            session,
            "session-two",
            "quality_progress",
        )
        app_state.queue_project_status_event(session, "session-three")

        assert len(app_state._project_status_event_tasks) == 1
        publisher = app_state._project_status_event_tasks["project-generic"]
        assert publisher in app_state._project_status_event_task_set

        await publisher

        assert published == [("project-generic", "session-three", "status_change", 3)]
        assert not app_state._project_status_event_pending
    finally:
        tasks = set(app_state._project_status_event_tasks.values())
        tasks.update(app_state._project_status_event_task_set)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        app_state._project_status_event_pending.clear()
        app_state._project_status_event_tasks.clear()
        app_state._project_status_event_task_set.clear()
