import asyncio
import logging
from datetime import datetime, timedelta

import pytest

from models import GenerationStatus


@pytest.mark.asyncio
async def test_perform_session_cleanup_clears_sidecar_state(monkeypatch, caplog):
    from core import app_state

    session_id = "expired-session"
    original_sessions = dict(app_state.active_sessions)
    app_state.active_sessions.clear()
    app_state.active_sessions[session_id] = {
        "created_at": datetime.now() - timedelta(seconds=10),
        "status": GenerationStatus.COMPLETED,
    }
    cleared_sidecars = []

    monkeypatch.setattr(app_state.config, "SESSION_TIMEOUT", 1)
    monkeypatch.setattr(app_state.config, "VERBOSE_MAX_ENTRIES", 100)
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    monkeypatch.setattr(app_state, "cleanup_session_sidecars", cleared_sidecars.append)
    caplog.set_level(logging.INFO, logger=app_state.logger.name)

    try:
        result = await app_state.perform_session_cleanup()
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)

    assert session_id not in app_state.active_sessions
    assert cleared_sidecars == [session_id]
    assert result["expired_count"] == 1
    assert result["remaining_sessions"] == 0
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == app_state.logger.name and record.levelno == logging.INFO
    ]
    assert "Cleaned up 1 expired session(s)" in info_messages
    assert session_id not in "\n".join(info_messages)


@pytest.mark.asyncio
async def test_session_cleanup_loop_enters_idle_cadence(monkeypatch):
    from core import app_state

    waits = []

    async def fake_wait(seconds: int, *, wakeable: bool) -> bool:
        waits.append((seconds, wakeable))
        if len(waits) == 3:
            raise asyncio.CancelledError()
        return False

    async def fake_cleanup():
        return {
            "expired_count": 0,
            "trimmed_log_count": 0,
            "remaining_sessions": 0,
        }

    monkeypatch.setattr(app_state.config, "SESSION_CLEANUP_INTERVAL", 10)
    monkeypatch.setattr(app_state.config, "SESSION_CLEANUP_IDLE_EMPTY_CHECKS", 2)
    monkeypatch.setattr(app_state.config, "SESSION_CLEANUP_IDLE_INTERVAL", 60)
    monkeypatch.setattr(app_state, "_session_cleanup_activity_counter", 0)
    monkeypatch.setattr(app_state, "_wait_for_session_cleanup_interval", fake_wait)
    monkeypatch.setattr(app_state, "perform_session_cleanup", fake_cleanup)

    with pytest.raises(asyncio.CancelledError):
        await app_state.session_cleanup_loop()

    assert waits == [
        (10, False),
        (10, False),
        (60, True),
    ]
