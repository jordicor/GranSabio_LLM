"""Regressions for the ``visibility`` subscriber filter (C0.6 / C7 plumbing).

``full`` (default) shows everything the engine produces, labelled. ``canonical``
is a convenience filter: it drops drafts and thinking, live AND in the initial
late-join snapshot, while keeping status, control, accepted and terminal events.
"""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

import pytest

from content_events import ContentStatus, ContentSubphase, Visibility
from core import app_state as app_state_module
from core.cancellation import cancellation_registry
from models import GenerationStatus
from services.project_stream import ProjectStreamManager


async def _publish(project_id, project_epoch, session_id, **kwargs):
    await app_state_module.publish_project_phase_chunk(
        project_id,
        "generation",
        kwargs.pop("content", None),
        session_id=session_id,
        project_epoch=project_epoch,
        **kwargs,
    )


async def _collect_generation_events(visibility: str):
    """Drive one manager over the generation phase and return what it yielded."""
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = ProjectStreamManager(project_id, {"generation"}, visibility)
    stream = manager.stream()
    collected = []

    async def _drain():
        async for chunk in stream:
            collected.append(chunk)

    drain_task = asyncio.create_task(_drain())
    try:
        # Give the manager time to emit `connected` and subscribe.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if manager._content_queues:
                break
        assert manager._content_queues, "manager did not subscribe to the phase"

        await _publish(
            project_id,
            project_epoch,
            session_id,
            content="draft body",
            subphase=ContentSubphase.DRAFT_STREAM,
            content_status=ContentStatus.DRAFT,
        )
        await _publish(
            project_id,
            project_epoch,
            session_id,
            content="reasoning",
            subphase=ContentSubphase.THINKING,
            content_status=ContentStatus.DRAFT,
            is_thinking=True,
        )
        await _publish(
            project_id,
            project_epoch,
            session_id,
            content="accepted body",
            subphase=ContentSubphase.ACCEPTED_STREAM,
            content_status=ContentStatus.ACCEPTED,
        )
        await _publish(
            project_id,
            project_epoch,
            session_id,
            event="retry_start",
            reason="network hiccup",
        )
        await _publish(
            project_id,
            project_epoch,
            session_id,
            event="session_end",
            status="completed",
            end_stream=True,
        )
        await asyncio.wait_for(drain_task, timeout=2.0)
    finally:
        if not drain_task.done():
            drain_task.cancel()
        await stream.aclose()
        app_state_module.reset_session_event_sequences(session_id)

    payloads = []
    for raw in collected:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not text.startswith("data: "):
            continue
        payloads.append(json.loads(text[len("data: "):].strip()))
    return payloads


@pytest.mark.asyncio
async def test_full_visibility_streams_drafts_and_thinking():
    payloads = await _collect_generation_events(Visibility.FULL)
    contents = [payload.get("content") for payload in payloads if payload.get("content")]
    types = [payload["type"] for payload in payloads]

    assert payloads[0]["type"] == "connected"
    assert payloads[0]["visibility"] == Visibility.FULL
    assert contents == ["draft body", "reasoning", "accepted body"]
    assert "retry_start" in types
    assert "session_end" in types


@pytest.mark.asyncio
async def test_canonical_visibility_hides_drafts_but_keeps_control_and_commits():
    payloads = await _collect_generation_events(Visibility.CANONICAL)
    contents = [payload.get("content") for payload in payloads if payload.get("content")]
    types = [payload["type"] for payload in payloads]

    assert payloads[0]["visibility"] == Visibility.CANONICAL
    assert contents == ["accepted body"]
    assert "retry_start" in types
    assert "session_end" in types


@pytest.mark.asyncio
async def test_late_join_snapshot_respects_visibility(monkeypatch):
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    original_sessions = dict(app_state_module.active_sessions)
    original_lock = app_state_module.active_sessions_lock
    monkeypatch.setattr(app_state_module, "active_sessions_lock", None)
    app_state_module.active_sessions.clear()
    app_state_module.active_sessions[session_id] = {
        "project_id": project_id,
        "request_name": "Snapshot",
        "status": GenerationStatus.GENERATING,
        "current_phase": "generation",
        "current_iteration": 2,
        "max_iterations": 3,
        "created_at": datetime.utcnow(),
        "last_activity_at": datetime.utcnow(),
        "request": None,
        "generation_content": "live draft text",
        "last_generated_content": "accepted text",
        "qa_content": "qa draft text",
    }

    try:
        full = await app_state_module.get_project_status(
            project_id,
            include_content_snapshot=True,
        )
        canonical = await app_state_module.get_project_status(
            project_id,
            include_content_snapshot=True,
            visibility=Visibility.CANONICAL,
        )
    finally:
        app_state_module.active_sessions.clear()
        app_state_module.active_sessions.update(original_sessions)
        monkeypatch.setattr(app_state_module, "active_sessions_lock", original_lock)

    full_snapshot = full["sessions"][0]["content_snapshot"]
    canonical_snapshot = canonical["sessions"][0]["content_snapshot"]

    assert full_snapshot["generation"]["text"] == "live draft text"
    assert full_snapshot["qa"]["text"] == "qa draft text"
    # The snapshot labels what it exposes (C0.1): the live field is a draft.
    assert full_snapshot["generation"]["content_status"] == ContentStatus.DRAFT
    assert canonical_snapshot == {
        "generation": {
            "text": "accepted text",
            "truncated": False,
            "omitted_chars": 0,
            "content_status": ContentStatus.ACCEPTED,
        }
    }


@pytest.mark.asyncio
async def test_manager_passes_visibility_to_the_snapshot(monkeypatch):
    captured = {}

    async def _fake_status(project_id, *, include_content_snapshot=False, visibility=Visibility.FULL):
        captured["project_id"] = project_id
        captured["include_content_snapshot"] = include_content_snapshot
        captured["visibility"] = visibility
        return {"project_id": project_id, "sessions": []}

    monkeypatch.setattr(app_state_module, "get_project_status", _fake_status)

    project_id = f"test-project-{uuid4()}"
    await cancellation_registry.begin_project_admission(project_id)
    manager = ProjectStreamManager(project_id, {"status"}, Visibility.CANONICAL)
    stream = manager.stream()
    try:
        await stream.__anext__()  # connected
        snapshot_event = json.loads((await stream.__anext__()).decode("utf-8")[len("data: "):])
    finally:
        await stream.aclose()

    assert snapshot_event["type"] == "status_snapshot"
    assert captured["include_content_snapshot"] is True
    assert captured["visibility"] == Visibility.CANONICAL
