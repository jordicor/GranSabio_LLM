"""Regressions for the /status verbose cursor (fiche C12).

The engine keeps a ring buffer of at most ``VERBOSE_MAX_ENTRIES`` verbose
entries, pruned on every append and again in periodic maintenance. A cursor by
list index desynchronizes silently on each prune, so the contract is a
MONOTONIC sequence: ``verbose_log_next_offset`` counts every entry ever
appended and never rewinds, and ``/status`` derives the offset of the oldest
retained entry from it. Callers poll with ``?verbose_since=<seq>`` and learn
about anything that fell out of the window through ``verbose_gap``.

Full history is the debugger's job (C13/C0.8); the engine's memory stays
bounded.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from core import app_state as app_state_module
from core import generation_processor as generation_processor_module
from core import generation_routes as generation_routes_module
from core.generation_processor import add_verbose_log
from core.generation_routes import _verbose_log_window, get_status
from models import GenerationStatus


@pytest.fixture
def isolated_sessions(monkeypatch):
    """Give each test its own active_sessions map without a lock."""
    original_sessions = dict(app_state_module.active_sessions)
    monkeypatch.setattr(app_state_module, "active_sessions_lock", None)
    app_state_module.active_sessions.clear()
    try:
        yield app_state_module.active_sessions
    finally:
        app_state_module.active_sessions.clear()
        app_state_module.active_sessions.update(original_sessions)


def _new_session(sessions, session_id: str) -> dict:
    session = {
        "session_id": session_id,
        "project_id": None,
        "request_name": "verbose cursor",
        "status": GenerationStatus.GENERATING,
        "current_iteration": 1,
        "max_iterations": 3,
        "verbose_log": [],
        "created_at": datetime.now(),
        "last_activity_at": datetime.now(),
    }
    sessions[session_id] = session
    return session


# ---------------------------------------------------------------------------
# The pure cursor window
# ---------------------------------------------------------------------------


def test_window_without_cursor_returns_the_whole_retained_buffer():
    session = {"verbose_log": ["a", "b", "c"], "verbose_log_next_offset": 3}

    window = _verbose_log_window(session, None)

    assert window["verbose_log"] == ["a", "b", "c"]
    assert window["verbose_returned_from"] == 0
    assert window["verbose_base_offset"] == 0
    assert window["verbose_next_offset"] == 3
    assert window["verbose_gap"] is False
    assert window["verbose_dropped"] == 0


def test_window_with_cursor_returns_only_new_entries():
    session = {"verbose_log": ["a", "b", "c", "d"], "verbose_log_next_offset": 4}

    window = _verbose_log_window(session, 2)

    assert window["verbose_log"] == ["c", "d"]
    assert window["verbose_returned_from"] == 2
    assert window["verbose_next_offset"] == 4
    assert window["verbose_gap"] is False


def test_window_at_head_returns_nothing_and_keeps_the_cursor():
    session = {"verbose_log": ["a", "b"], "verbose_log_next_offset": 2}

    window = _verbose_log_window(session, 2)

    assert window["verbose_log"] == []
    assert window["verbose_returned_from"] == 2
    assert window["verbose_next_offset"] == 2
    assert window["verbose_gap"] is False


def test_window_reports_a_gap_when_the_cursor_fell_out_of_the_buffer():
    # 10 entries appended, only the last 3 retained: offsets 7, 8 and 9.
    session = {"verbose_log": ["h", "i", "j"], "verbose_log_next_offset": 10}

    window = _verbose_log_window(session, 4)

    assert window["verbose_log"] == ["h", "i", "j"]
    assert window["verbose_base_offset"] == 7
    assert window["verbose_returned_from"] == 7
    assert window["verbose_next_offset"] == 10
    assert window["verbose_gap"] is True
    assert window["verbose_dropped"] == 3


def test_window_tolerates_a_cursor_ahead_of_the_engine():
    session = {"verbose_log": ["a", "b"], "verbose_log_next_offset": 2}

    window = _verbose_log_window(session, 99)

    assert window["verbose_log"] == []
    assert window["verbose_next_offset"] == 2
    assert window["verbose_gap"] is False


def test_window_of_a_session_that_never_logged_is_empty_at_zero():
    window = _verbose_log_window({}, None)

    assert window["verbose_log"] == []
    assert window["verbose_base_offset"] == 0
    assert window["verbose_next_offset"] == 0


# ---------------------------------------------------------------------------
# The append path keeps the sequence monotonic across pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_append_path_prune_advances_the_base_offset(
    isolated_sessions,
    monkeypatch,
):
    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        3,
        raising=False,
    )
    session_id = f"verbose-{uuid4()}"
    session = _new_session(isolated_sessions, session_id)

    for index in range(7):
        await add_verbose_log(session_id, f"entry-{index}")

    assert len(session["verbose_log"]) == 3
    assert session["verbose_log_next_offset"] == 7

    window = _verbose_log_window(session, None)
    assert window["verbose_base_offset"] == 4
    assert window["verbose_next_offset"] == 7
    assert [entry.split("] ")[-1] for entry in window["verbose_log"]] == [
        "entry-4",
        "entry-5",
        "entry-6",
    ]


@pytest.mark.asyncio
async def test_polling_with_the_cursor_never_repeats_or_skips_entries(
    isolated_sessions,
    monkeypatch,
):
    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        100,
        raising=False,
    )
    session_id = f"verbose-{uuid4()}"
    session = _new_session(isolated_sessions, session_id)

    collected: list[str] = []
    cursor = 0
    for batch in range(3):
        for index in range(4):
            await add_verbose_log(session_id, f"batch-{batch}-entry-{index}")
        window = _verbose_log_window(session, cursor)
        assert window["verbose_gap"] is False
        collected.extend(window["verbose_log"])
        cursor = window["verbose_next_offset"]

    assert cursor == 12
    assert len(collected) == 12
    assert len(set(collected)) == 12
    assert collected == session["verbose_log"]


# ---------------------------------------------------------------------------
# The maintenance path prunes the same way
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maintenance_prune_advances_the_base_offset(
    isolated_sessions,
    monkeypatch,
):
    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        100,
        raising=False,
    )
    session_id = f"verbose-{uuid4()}"
    session = _new_session(isolated_sessions, session_id)

    for index in range(10):
        await add_verbose_log(session_id, f"entry-{index}")
    cursor = _verbose_log_window(session, None)["verbose_next_offset"]
    assert cursor == 10

    # Periodic maintenance trims with a tighter cap than the append path used.
    monkeypatch.setattr(
        app_state_module.config,
        "VERBOSE_MAX_ENTRIES",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        app_state_module.config,
        "SESSION_TIMEOUT",
        3600,
        raising=False,
    )
    result = await app_state_module.perform_session_cleanup()

    assert result["trimmed_log_count"] == 1
    assert len(session["verbose_log"]) == 4
    assert session["verbose_log_next_offset"] == 10

    window = _verbose_log_window(session, cursor - 8)
    assert window["verbose_base_offset"] == 6
    assert window["verbose_gap"] is True, "maintenance pruning must be detectable"
    assert window["verbose_dropped"] == 4
    assert window["verbose_returned_from"] == 6
    assert [entry.split("] ")[-1] for entry in window["verbose_log"]] == [
        "entry-6",
        "entry-7",
        "entry-8",
        "entry-9",
    ]


# ---------------------------------------------------------------------------
# The endpoint contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_endpoint_exposes_the_cursor_fields(
    isolated_sessions,
    monkeypatch,
):
    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        3,
        raising=False,
    )

    class _NoEscalations:
        def get_session_escalations(self, _session_id):
            return []

    monkeypatch.setattr(
        generation_routes_module,
        "get_tracker",
        lambda: _NoEscalations(),
    )
    session_id = f"verbose-{uuid4()}"
    _new_session(isolated_sessions, session_id)

    for index in range(5):
        await add_verbose_log(session_id, f"entry-{index}")

    full = await get_status(session_id)
    assert full["verbose_next_offset"] == 5
    assert full["verbose_base_offset"] == 2
    assert full["verbose_returned_from"] == 2
    assert len(full["verbose_log"]) == 3
    assert full["verbose_gap"] is False

    incremental = await get_status(session_id, verbose_since=4)
    assert len(incremental["verbose_log"]) == 1
    assert incremental["verbose_log"][0].endswith("entry-4")
    assert incremental["verbose_returned_from"] == 4
    assert incremental["verbose_next_offset"] == 5

    stale = await get_status(session_id, verbose_since=0)
    assert stale["verbose_gap"] is True
    assert stale["verbose_dropped"] == 2


@pytest.mark.asyncio
async def test_status_endpoint_returns_more_than_a_hundred_entries(
    isolated_sessions,
    monkeypatch,
):
    """The hardcoded [-100:] slice is gone; the ring buffer size rules."""
    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        150,
        raising=False,
    )

    class _NoEscalations:
        def get_session_escalations(self, _session_id):
            return []

    monkeypatch.setattr(
        generation_routes_module,
        "get_tracker",
        lambda: _NoEscalations(),
    )
    session_id = f"verbose-{uuid4()}"
    _new_session(isolated_sessions, session_id)

    for index in range(120):
        await add_verbose_log(session_id, f"entry-{index}")

    summary = await get_status(session_id)

    assert len(summary["verbose_log"]) == 120
    assert summary["verbose_base_offset"] == 0
    assert summary["verbose_next_offset"] == 120


@pytest.mark.asyncio
async def test_status_route_parses_the_cursor_query_parameter(
    isolated_sessions,
    monkeypatch,
):
    """The Annotated Query default must not leak into the handler."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        generation_processor_module.config,
        "VERBOSE_MAX_ENTRIES",
        100,
        raising=False,
    )

    class _NoEscalations:
        def get_session_escalations(self, _session_id):
            return []

    monkeypatch.setattr(
        generation_routes_module,
        "get_tracker",
        lambda: _NoEscalations(),
    )
    session_id = f"verbose-{uuid4()}"
    _new_session(isolated_sessions, session_id)
    for index in range(3):
        await add_verbose_log(session_id, f"entry-{index}")

    # No context manager: it would run the real app startup and shutdown
    # (AI services, debugger persistence, cleanup loop). Only the route matters.
    client = TestClient(app_state_module.app)
    full = client.get(f"/status/{session_id}")
    incremental = client.get(f"/status/{session_id}", params={"verbose_since": 2})
    invalid = client.get(f"/status/{session_id}", params={"verbose_since": -1})

    assert full.status_code == 200
    assert full.json()["verbose_next_offset"] == 3
    assert len(full.json()["verbose_log"]) == 3
    assert incremental.status_code == 200
    assert len(incremental.json()["verbose_log"]) == 1
    assert incremental.json()["verbose_returned_from"] == 2
    assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_expired_sessions_are_still_removed_by_maintenance(
    isolated_sessions,
    monkeypatch,
):
    """The cursor comment added to the trim block did not change expiry."""
    monkeypatch.setattr(
        app_state_module.config,
        "SESSION_TIMEOUT",
        1,
        raising=False,
    )
    session_id = f"verbose-{uuid4()}"
    session = _new_session(isolated_sessions, session_id)
    session["status"] = GenerationStatus.COMPLETED
    session["completed_at"] = datetime.now() - timedelta(seconds=600)

    result = await app_state_module.perform_session_cleanup()

    assert result["expired_count"] == 1
    assert session_id not in app_state_module.active_sessions
