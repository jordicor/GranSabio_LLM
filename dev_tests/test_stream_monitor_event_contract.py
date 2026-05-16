"""Focused regressions for the /monitor project stream event contract."""

from datetime import datetime
from pathlib import Path

import pytest

from models import GenerationStatus

ROOT = Path(__file__).resolve().parents[1]
STREAM_MONITOR_JS = ROOT / "static" / "js" / "stream_monitor.js"
STREAM_MONITOR_TEMPLATE = ROOT / "templates" / "stream_monitor.html"
APP_STATE = ROOT / "core" / "app_state.py"


def test_frontend_phase_mapping_and_analysis_contract():
    source = STREAM_MONITOR_JS.read_text(encoding="utf-8")
    template = STREAM_MONITOR_TEMPLATE.read_text(encoding="utf-8")

    assert "analysis: ['auto_qa', 'preflight', 'consensus']" in source
    assert "smartedit: ['smart_edit']" in source
    assert "gransabio: ['gran_sabio']" in source
    assert "auto_qa: ''" not in source  # Iteration storage is generated from ITERATION_CONTENT_KEYS.
    assert "const ITERATION_CONTENT_KEYS = ['auto_qa', 'preflight', 'generation', 'qa', 'arbiter', 'consensus', 'gransabio']" in source
    assert "data-filter=\"auto_qa\"" in template
    assert "Auto-QA + Preflight + Consensus" in template


def test_frontend_structured_events_and_terminal_statuses_are_known():
    source = STREAM_MONITOR_JS.read_text(encoding="utf-8")

    for status in ("auto_qa_rejected", "preflight_rejected", "cancelled"):
        assert status in source
    for event_type in (
        "retry_start",
        "retry",
        "error",
        "project_end",
        "project_cancelled",
        "tool_call_start",
        "tool_call_result",
        "tool_call_error",
        "force_finalize",
        "context_overflow_midloop",
        "validate_draft_oversize",
        "tool_loop_error",
        "tool_loop_complete",
    ):
        assert event_type in source

    assert "eventType.startsWith('grounding_')" in source
    assert "handleSessionEnd(data)" in source
    assert "summary.total_sessions" in source
    assert "summary.active_sessions" in source
    assert "summary.completed_sessions" in source
    assert "hydrateRequestsFromProjectSnapshot(project)" in source
    assert "content_snapshot" in source
    assert "shouldRenderRequestScopedPanelEvent(data, 'smartedit')" in source
    assert "appendStructuredEventForRequest(data, line, displayPhase)" in source


def test_monitor_ui_avoids_inline_project_connect_handlers():
    source = STREAM_MONITOR_JS.read_text(encoding="utf-8")
    template = STREAM_MONITOR_TEMPLATE.read_text(encoding="utf-8")

    assert "onclick=\"connectToProject('" not in source
    assert "addEventListener('click', () => connectToProject" in source
    assert "recentStatusFilter" in template
    assert "DEFAULT_PANEL_MAX_CHARS" in source


def test_generation_tool_loop_events_are_forwarded_to_monitor():
    source = (ROOT / "core" / "generation_processor.py").read_text(encoding="utf-8")

    assert "def _make_generation_tool_event_callback" in source
    assert "tool_event_callback=generation_tool_event_callback" in source
    assert 'subphase="tool_loop_final"' in source


def test_backend_terminal_broadcast_phases_include_arbiter():
    source = APP_STATE.read_text(encoding="utf-8")
    expected = '["auto_qa", "preflight", "generation", "qa", "arbiter", "smart_edit", "consensus", "gran_sabio"]'

    assert source.count(expected) >= 2


@pytest.mark.asyncio
async def test_gran_sabio_review_counts_active_and_sets_badge(monkeypatch):
    from core import app_state

    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    app_state.active_sessions["session-gran-sabio"] = {
        "project_id": "project-gran-sabio",
        "request_name": "Gran Sabio review",
        "status": GenerationStatus.GRAN_SABIO_REVIEW,
        "current_phase": "inline_deal_breaker_review",
        "current_iteration": 1,
        "max_iterations": 3,
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
        "request": None,
    }
    app_state.active_sessions["session-gran-sabio-regeneration"] = {
        "project_id": "project-gran-sabio",
        "request_name": "Gran Sabio regeneration",
        "status": "waiting",
        "current_phase": "gran_sabio_regeneration",
        "current_iteration": 2,
        "max_iterations": 3,
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
        "request": None,
    }

    try:
        status = await app_state.get_project_status("project-gran-sabio")
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    assert status["status"] == "running"
    assert status["summary"]["active_sessions"] == 2
    assert status["sessions"][0]["gran_sabio"]["active"] is True
    assert status["sessions"][1]["gran_sabio"]["active"] is True


@pytest.mark.asyncio
async def test_monitor_active_counts_gran_sabio_review(monkeypatch):
    from core import app_state
    from core.monitor_routes import list_active_connections

    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    app_state.active_sessions["session-monitor-gran-sabio"] = {
        "project_id": "project-monitor-gran-sabio",
        "request_name": "Gran Sabio monitor",
        "status": GenerationStatus.GRAN_SABIO_REVIEW,
        "current_phase": "inline_deal_breaker_review",
        "created_at": datetime.utcnow(),
    }
    app_state.active_sessions["session-monitor-gran-sabio-regeneration"] = {
        "project_id": "project-monitor-gran-sabio",
        "request_name": "Gran Sabio regeneration monitor",
        "status": "waiting",
        "current_phase": "gran_sabio_regeneration",
        "created_at": datetime.utcnow(),
    }

    try:
        payload = await list_active_connections()
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    project = next(p for p in payload["projects"] if p["project_id"] == "project-monitor-gran-sabio")
    assert project["status"] == "running"
    assert project["active_sessions"] == 2


@pytest.mark.asyncio
async def test_monitor_active_defaults_to_running_projects(monkeypatch):
    from core import app_state
    from core.monitor_routes import list_active_connections

    original_sessions = dict(app_state.active_sessions)
    original_reserved = set(app_state.reserved_project_ids)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    app_state.reserved_project_ids.clear()
    app_state.reserved_project_ids.add("reserved-idle-project")
    app_state.active_sessions["session-running"] = {
        "project_id": "project-running",
        "request_name": "Running",
        "status": GenerationStatus.GENERATING,
        "current_phase": "generating",
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
    }
    app_state.active_sessions["session-completed"] = {
        "project_id": "project-completed",
        "request_name": "Completed",
        "status": GenerationStatus.COMPLETED,
        "current_phase": "completed",
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
    }

    try:
        default_payload = await list_active_connections()
        all_payload = await list_active_connections(status="all", include_reserved=True)
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        app_state.reserved_project_ids.clear()
        app_state.reserved_project_ids.update(original_reserved)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    default_ids = {project["project_id"] for project in default_payload["projects"]}
    all_ids = {project["project_id"] for project in all_payload["projects"]}
    assert default_ids == {"project-running"}
    assert {"project-running", "project-completed", "reserved-idle-project"} <= all_ids


@pytest.mark.asyncio
async def test_project_status_reports_cancelled_terminal_sessions(monkeypatch):
    from core import app_state

    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    app_state.active_sessions["session-cancelled"] = {
        "project_id": "project-cancelled-session",
        "request_name": "Cancelled",
        "status": GenerationStatus.CANCELLED,
        "current_phase": "cancelled",
        "current_iteration": 1,
        "max_iterations": 3,
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
        "request": None,
    }

    try:
        status = await app_state.get_project_status("project-cancelled-session")
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    assert status["status"] == "cancelled"
    assert status["summary"]["cancelled_sessions"] == 1


@pytest.mark.asyncio
async def test_project_status_can_include_bounded_content_snapshot(monkeypatch):
    from core import app_state

    original_sessions = dict(app_state.active_sessions)
    original_lock = app_state.active_sessions_lock
    monkeypatch.setattr(app_state, "active_sessions_lock", None)
    app_state.active_sessions.clear()
    app_state.active_sessions["session-snapshot"] = {
        "project_id": "project-snapshot",
        "request_name": "Snapshot",
        "status": GenerationStatus.GENERATING,
        "current_phase": "generating",
        "current_iteration": 1,
        "max_iterations": 3,
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
        "request": None,
        "generation_content": "x" * 60000,
        "qa_content": "qa text",
    }

    try:
        without_snapshot = await app_state.get_project_status("project-snapshot")
        with_snapshot = await app_state.get_project_status(
            "project-snapshot",
            include_content_snapshot=True,
        )
    finally:
        app_state.active_sessions.clear()
        app_state.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state, "active_sessions_lock", original_lock)

    session_without = without_snapshot["sessions"][0]
    session_with = with_snapshot["sessions"][0]
    assert "content_snapshot" not in session_without
    assert session_with["content_snapshot"]["generation"]["truncated"] is True
    assert len(session_with["content_snapshot"]["generation"]["text"]) == 50000
    assert session_with["content_snapshot"]["qa"]["text"] == "qa text"
