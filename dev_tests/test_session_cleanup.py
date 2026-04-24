from datetime import datetime, timedelta

import pytest

from models import GenerationStatus


@pytest.mark.asyncio
async def test_perform_session_cleanup_clears_sidecar_state(monkeypatch):
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

    try:
        await app_state.perform_session_cleanup()
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)

    assert session_id not in app_state.active_sessions
    assert cleared_sidecars == [session_id]
