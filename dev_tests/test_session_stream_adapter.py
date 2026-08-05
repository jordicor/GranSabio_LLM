"""Contract tests for ``GET /stream/{session_id}`` (C9).

The session stream is an adapter over the project bus, not a second bus. What
it must guarantee:

- it exists and resolves the project of the session (404 when unknown),
- it delivers only the events of ITS session, and never exposes sibling
  sessions of the same project - not through content, not through the
  late-join snapshot, not through status events,
- it projects the lifecycle onto a flat contract (``status``,
  ``current_phase``, ``iteration`` at top level),
- it drains what its session already produced and closes after its own
  ``session_end``, whatever phases it subscribed to (the terminal also travels
  on the status channel) and exactly once,
- ``visibility=canonical`` filters drafts and thinking exactly as on the
  project stream,
- ``/stream/console`` keeps answering as the console (route order).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.streaming_routes as streaming_routes  # noqa: E402
import json_utils as json  # noqa: E402
from content_events import ContentStatus, ContentSubphase, Visibility  # noqa: E402
from core import app_state as app_state_module  # noqa: E402
from core.cancellation import cancellation_registry  # noqa: E402
from services.project_stream import SessionStreamManager  # noqa: E402


def _project_payload(*session_ids: str) -> Dict[str, Any]:
    """Build a project status payload holding the given sessions."""
    return {
        "project_id": "project-x",
        "status": "running",
        "cancel_mode": None,
        "project_epoch": 7,
        "sessions": [
            {
                "session_id": session_id,
                "request_name": f"request-{session_id}",
                "status": "generating",
                "phase": "generating",
                "iteration": 2,
                "max_iterations": 3,
                "qa": {"progress": {"completed": 1, "total": 4}},
                "consensus": {"last_score": 7.5},
                "content_snapshot": {"generation": {"text": f"draft of {session_id}"}},
            }
            for session_id in session_ids
        ],
        "summary": {
            "total_sessions": len(session_ids),
            "active_sessions": len(session_ids),
            "completed_sessions": 0,
            "cancelled_sessions": 0,
        },
    }


def _manager(session_id: str = "session-a", phases: Optional[set] = None) -> SessionStreamManager:
    return SessionStreamManager(
        "project-x",
        session_id,
        phases if phases is not None else {"generation", "status"},
    )


# ---------------------------------------------------------------------------
# Projection: never expose sibling sessions
# ---------------------------------------------------------------------------

def test_snapshot_is_reduced_to_the_requested_session():
    manager = _manager()
    payload = json.loads(
        manager._build_snapshot_event(_project_payload("session-a", "session-b"))
    )

    assert payload["type"] == "status_snapshot"
    assert payload["session_id"] == "session-a"
    assert payload["session"]["session_id"] == "session-a"
    # Flat lifecycle contract.
    assert payload["status"] == "generating"
    assert payload["current_phase"] == "generating"
    assert payload["iteration"] == 2
    assert payload["max_iterations"] == 3
    # Project block carries identity/lifecycle only.
    assert payload["project"] == {
        "project_id": "project-x",
        "status": "running",
        "cancel_mode": None,
        "project_epoch": 7,
    }
    serialized = json.dumps(payload)
    assert "session-b" not in serialized
    assert "summary" not in payload


def test_status_change_is_reduced_and_drops_the_trigger_session():
    manager = _manager()
    raw = json.dumps(
        {
            "type": "status_change",
            "timestamp": 1234,
            "trigger_session": "session-b",
            "project": _project_payload("session-a", "session-b"),
        }
    )

    payload = json.loads(manager._enrich_status_event(raw))

    assert payload["type"] == "status_change"
    assert payload["timestamp"] == 1234
    assert payload["session"]["session_id"] == "session-a"
    assert "trigger_session" not in payload
    assert "session-b" not in json.dumps(payload)


def test_status_change_about_another_session_is_dropped():
    manager = _manager()
    raw = json.dumps(
        {
            "type": "status_change",
            "trigger_session": "session-b",
            "project": _project_payload("session-b"),
        }
    )

    assert manager._enrich_status_event(raw) is None


def test_project_lifecycle_status_events_survive_without_our_session():
    manager = _manager()
    cancelled = json.dumps(
        {"type": "project_cancelled", "project": _project_payload("session-b")}
    )
    stream_end = json.dumps({"type": "stream_end", "reason": "project_cancelled"})

    cancelled_payload = json.loads(manager._enrich_status_event(cancelled))
    end_payload = json.loads(manager._enrich_status_event(stream_end))

    assert cancelled_payload["type"] == "project_cancelled"
    assert cancelled_payload["session"] is None
    assert "session-b" not in json.dumps(cancelled_payload)
    assert end_payload["type"] == "stream_end"
    assert end_payload["reason"] == "project_cancelled"
    assert end_payload["session_id"] == "session-a"


def test_unparseable_status_event_is_dropped_instead_of_forwarded():
    manager = _manager()
    assert manager._enrich_status_event("{not json") is None


def test_session_terminal_on_the_status_channel_is_scoped_to_its_owner():
    """The status channel carries session terminals: ours closes, siblings drop."""
    manager = _manager()
    mine = json.dumps(
        {
            "type": "session_end",
            "phase": "status",
            "session_id": "session-a",
            "status": "completed",
        }
    )
    theirs = json.dumps(
        {
            "type": "session_end",
            "phase": "status",
            "session_id": "session-b",
            "status": "failed",
        }
    )

    projected = json.loads(manager._enrich_status_event(mine))

    assert projected["type"] == "session_end"
    assert projected["session_id"] == "session-a"
    assert projected["status"] == "completed"
    assert manager._closes_on_event(projected) == "completed"
    # A sibling terminal must neither be forwarded nor close this stream.
    assert manager._enrich_status_event(theirs) is None


def test_content_events_of_other_sessions_are_dropped():
    manager = _manager()
    mine = json.dumps({"type": "chunk", "session_id": "session-a", "content": "mine"})
    theirs = json.dumps({"type": "chunk", "session_id": "session-b", "content": "theirs"})
    project_marker = json.dumps({"type": "project_end", "status": "cancelled"})
    anonymous = json.dumps({"type": "chunk", "content": "no owner"})

    assert manager._transform_content_event(mine) == mine
    assert manager._transform_content_event(theirs) is None
    assert manager._transform_content_event(project_marker) == project_marker
    assert manager._transform_content_event(anonymous) is None


# ---------------------------------------------------------------------------
# Live behaviour over the real bus
# ---------------------------------------------------------------------------

async def _publish(project_id, project_epoch, session_id, phase="generation", **kwargs):
    await app_state_module.publish_project_phase_chunk(
        project_id,
        phase,
        kwargs.pop("content", None),
        session_id=session_id,
        project_epoch=project_epoch,
        **kwargs,
    )


async def _drive(
    manager: SessionStreamManager,
    publisher,
) -> List[Dict[str, Any]]:
    """Subscribe, let ``publisher`` fill the queues, then multiplex to the end."""
    await manager._subscribe_to_phases()
    try:
        await publisher()
        events: List[Dict[str, Any]] = []
        async for raw in manager._multiplex_events():
            if raw is None:
                continue
            events.append(json.loads(raw))
        return events
    finally:
        await manager._unsubscribe_all()


@pytest.mark.asyncio
async def test_session_stream_filters_siblings_and_closes_after_its_session_end():
    project_id = f"test-project-{uuid4()}"
    mine = f"session-mine-{uuid4()}"
    sibling = f"session-sibling-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, mine, {"generation", "qa"})

    async def _publisher():
        await _publish(
            project_id, epoch, mine,
            content="my draft",
            subphase=ContentSubphase.DRAFT_STREAM,
            content_status=ContentStatus.DRAFT,
        )
        await _publish(
            project_id, epoch, sibling,
            content="sibling draft",
            subphase=ContentSubphase.DRAFT_STREAM,
            content_status=ContentStatus.DRAFT,
        )
        # Queued behind the session_end of the generation phase: the adapter
        # must still deliver it while draining.
        await _publish(
            project_id, epoch, mine,
            phase="qa",
            content="qa tail",
            content_status=ContentStatus.DRAFT,
        )
        for phase in ("generation", "qa"):
            await _publish(
                project_id, epoch, mine,
                phase=phase,
                event="session_end",
                status="completed",
            )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(mine)
        app_state_module.reset_session_event_sequences(sibling)

    contents = [event.get("content") for event in events if event.get("content")]
    types = [event["type"] for event in events]

    assert "my draft" in contents
    assert "qa tail" in contents, "events queued behind session_end must be drained"
    assert "sibling draft" not in contents
    assert types.count("session_end") == 1, "one session_end, not one per phase"
    assert types[-1] == "stream_end"
    assert events[-1]["session_id"] == mine
    assert events[-1]["reason"] == "completed"


@pytest.mark.asyncio
async def test_status_only_subscription_closes_on_its_own_session_end():
    """C9: `phases=status` has a session terminal instead of living to project end."""
    project_id = f"test-project-{uuid4()}"
    mine = f"session-mine-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, mine, {"status"})

    async def _publisher():
        await app_state_module.publish_project_session_end(
            project_id,
            mine,
            "completed",
            "request-mine",
            epoch,
        )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(mine)

    assert [event["type"] for event in events] == ["session_end", "stream_end"]
    assert events[0]["session_id"] == mine
    assert events[0]["status"] == "completed"
    assert events[-1]["reason"] == "completed"


@pytest.mark.asyncio
async def test_status_channel_terminal_of_a_sibling_never_closes_this_stream():
    """The shared status channel must not end a stream it does not belong to."""
    project_id = f"test-project-{uuid4()}"
    mine = f"session-mine-{uuid4()}"
    sibling = f"session-sibling-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, mine, {"status"})

    async def _publisher():
        await app_state_module.publish_project_session_end(
            project_id, sibling, "failed", None, epoch
        )
        await app_state_module.publish_project_session_end(
            project_id, mine, "completed", None, epoch
        )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(mine)
        app_state_module.reset_session_event_sequences(sibling)

    assert [event["type"] for event in events] == ["session_end", "stream_end"]
    assert events[0]["session_id"] == mine
    assert events[0]["status"] == "completed"
    assert sibling not in str(events)


@pytest.mark.asyncio
async def test_multi_phase_subscription_closes_once_on_the_session_terminal():
    """The content copies and the status copy collapse into ONE terminal."""
    project_id = f"test-project-{uuid4()}"
    mine = f"session-mine-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, mine, {"generation", "qa", "status"})

    async def _publisher():
        await _publish(
            project_id, epoch, mine,
            content="my draft",
            subphase=ContentSubphase.DRAFT_STREAM,
            content_status=ContentStatus.DRAFT,
        )
        await app_state_module.publish_project_session_end(
            project_id, mine, "completed", None, epoch
        )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(mine)

    types = [event["type"] for event in events]
    contents = [event.get("content") for event in events if event.get("content")]

    assert "my draft" in contents
    assert types.count("session_end") == 1, "one terminal, not one per channel"
    assert types.count("stream_end") == 1
    assert types[-1] == "stream_end"
    assert events[-1]["reason"] == "completed"


@pytest.mark.asyncio
async def test_content_phase_session_end_survives_a_full_subscriber_queue():
    """A terminal is never the event backpressure drops.

    ``session_end`` travels on the content phases with ``end_stream=False`` (the
    channel stays open for sibling sessions), so it used to go through the plain
    best-effort put and vanish on a full queue. A subscriber of content phases
    only would then never learn its session ended and would live until the
    project itself finished.
    """
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    app_state_module.project_phase_streams[project_id] = {"generation": [queue]}
    try:
        for index in range(2):
            queue.put_nowait(f"already-queued-{index}")

        # An ordinary draft on a full queue is still dropped: best effort.
        await _publish(
            project_id, epoch, session_id,
            content="draft nobody will read",
            content_status=ContentStatus.DRAFT,
        )
        assert queue.qsize() == 2

        await _publish(
            project_id, epoch, session_id,
            event="session_end",
            status="completed",
        )
        drained = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        app_state_module.project_phase_streams.pop(project_id, None)
        app_state_module.reset_session_event_sequences(session_id)

    terminals = [
        json.loads(raw)
        for raw in drained
        if raw is not None and not raw.startswith("already-queued")
    ]
    assert [event["type"] for event in terminals] == ["session_end"]
    assert terminals[0]["session_id"] == session_id
    assert terminals[0]["status"] == "completed"
    # Exactly one older event paid for the terminal, not the whole queue.
    assert "already-queued-1" in drained


@pytest.mark.asyncio
async def test_a_queued_close_sentinel_is_never_traded_for_a_session_terminal():
    """The eviction must not buy a terminal with the sentinel that ends the stream.

    A close sentinel at the head of a full queue is what makes the subscriber
    exit for good (kept from the hardening, §4). Dropping it to fit a
    ``session_end`` would leave the stream open forever instead of one event
    short.
    """
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    app_state_module.project_phase_streams[project_id] = {"generation": [queue]}
    try:
        queue.put_nowait(None)
        queue.put_nowait("queued-after-the-sentinel")

        await _publish(
            project_id, epoch, session_id,
            event="session_end",
            status="completed",
        )
        drained = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        app_state_module.project_phase_streams.pop(project_id, None)
        app_state_module.reset_session_event_sequences(session_id)

    assert None in drained, "the close sentinel must survive the eviction"
    assert not any(
        raw is not None and "session_end" in raw for raw in drained
    ), "the terminal is dropped rather than the sentinel"


@pytest.mark.asyncio
async def test_session_terminal_carries_the_same_identity_on_both_channels():
    """``phases=all`` receives two copies of one terminal: same shape either way.

    Which copy reaches the client first is a race between the content queue and
    the status queue, so a field present on only one of them would make the
    payload non-deterministic.
    """
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    content_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    status_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    app_state_module.project_phase_streams[project_id] = {"generation": [content_queue]}
    app_state_module.project_status_streams[project_id] = [status_queue]
    manager = SessionStreamManager(project_id, session_id, {"generation", "status"})
    try:
        await app_state_module.publish_project_session_end(
            project_id,
            session_id,
            "completed",
            "request-mine",
            epoch,
        )
        content_copy = json.loads(
            manager._prepare_event("generation", content_queue.get_nowait())
        )
        status_copy = json.loads(
            manager._prepare_event("status", status_queue.get_nowait())
        )
    finally:
        app_state_module.project_phase_streams.pop(project_id, None)
        app_state_module.project_status_streams.pop(project_id, None)
        app_state_module.reset_session_event_sequences(session_id)

    for field in ("type", "session_id", "status", "request_name", "project_epoch"):
        assert status_copy[field] == content_copy[field], field
    assert status_copy["request_name"] == "request-mine"
    assert status_copy["project_epoch"] == epoch
    # The channel each copy came from is the ONE field meant to differ.
    assert content_copy["phase"] == "generation"
    assert status_copy["phase"] == "status"


@pytest.mark.asyncio
async def test_session_stream_canonical_visibility_drops_drafts_and_thinking():
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(
        project_id, session_id, {"generation"}, Visibility.CANONICAL
    )

    async def _publisher():
        await _publish(
            project_id, epoch, session_id,
            content="draft body",
            subphase=ContentSubphase.DRAFT_STREAM,
            content_status=ContentStatus.DRAFT,
        )
        await _publish(
            project_id, epoch, session_id,
            content="reasoning",
            subphase=ContentSubphase.THINKING,
            content_status=ContentStatus.DRAFT,
            is_thinking=True,
        )
        await _publish(
            project_id, epoch, session_id,
            content="accepted body",
            subphase=ContentSubphase.ACCEPTED_STREAM,
            content_status=ContentStatus.ACCEPTED,
        )
        await _publish(
            project_id, epoch, session_id,
            event="session_end",
            status="completed",
        )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(session_id)

    contents = [event.get("content") for event in events if event.get("content")]
    assert contents == ["accepted body"]


@pytest.mark.asyncio
async def test_session_stream_content_events_carry_the_session_sequence():
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, session_id, {"generation"})

    async def _publisher():
        for index in range(3):
            await _publish(
                project_id, epoch, session_id,
                content=f"chunk-{index}",
                content_status=ContentStatus.DRAFT,
            )
        await _publish(
            project_id, epoch, session_id,
            event="session_end",
            status="completed",
        )

    try:
        events = await _drive(manager, _publisher)
    finally:
        app_state_module.reset_session_event_sequences(session_id)

    sequences = [event["sequence"] for event in events if event.get("content")]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


@pytest.mark.asyncio
async def test_late_join_snapshot_is_scoped_and_carries_the_visibility(monkeypatch):
    """The snapshot asks for the subscriber's visibility and drops siblings."""
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    await cancellation_registry.begin_project_admission(project_id)
    captured: Dict[str, Any] = {}

    async def _fake_status(project, *, include_content_snapshot=False, visibility=Visibility.FULL):
        captured["include_content_snapshot"] = include_content_snapshot
        captured["visibility"] = visibility
        payload = _project_payload(session_id, "session-other")
        payload["project_id"] = project
        return payload

    monkeypatch.setattr(app_state_module, "get_project_status", _fake_status)

    manager = SessionStreamManager(
        project_id, session_id, {"status"}, Visibility.CANONICAL
    )
    stream = manager.stream()
    try:
        await stream.__anext__()  # connected
        snapshot = json.loads((await stream.__anext__()).decode("utf-8")[len("data: "):])
    finally:
        await stream.aclose()

    assert captured["include_content_snapshot"] is True
    assert captured["visibility"] == Visibility.CANONICAL
    assert snapshot["type"] == "status_snapshot"
    assert snapshot["session"]["session_id"] == session_id
    assert "session-other" not in json.dumps(snapshot)


@pytest.mark.asyncio
async def test_connected_event_is_sent_after_the_subscriptions_exist():
    """The greeting must not promise a delivery window that has not started."""
    project_id = f"test-project-{uuid4()}"
    session_id = f"session-{uuid4()}"
    await cancellation_registry.begin_project_admission(project_id)

    manager = SessionStreamManager(project_id, session_id, {"generation"})
    stream = manager.stream()
    try:
        first = await stream.__anext__()
        # Read before aclose(): the generator unsubscribes on exit.
        subscribed_when_greeted = set(manager._subscribed_phases)
    finally:
        await stream.aclose()

    payload = json.loads(first.decode("utf-8")[len("data: "):])
    assert payload["type"] == "connected"
    assert payload["session_id"] == session_id
    assert payload["project_id"] == project_id
    assert subscribed_when_greeted == {"generation"}, (
        "subscription must exist before `connected` reaches the client"
    )


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_stream_route_rejects_unknown_session_and_bad_params():
    with pytest.raises(HTTPException) as unknown:
        await streaming_routes.stream_session_unified(
            f"missing-{uuid4()}", phases="all", visibility="full"
        )
    assert unknown.value.status_code == 404

    with pytest.raises(HTTPException) as too_long:
        await streaming_routes.stream_session_unified(
            "x" * 129, phases="all", visibility="full"
        )
    assert too_long.value.status_code == 400

    with pytest.raises(HTTPException) as bad_phases:
        await streaming_routes.stream_session_unified(
            "some-session", phases="all,generation", visibility="full"
        )
    assert bad_phases.value.status_code == 400

    with pytest.raises(HTTPException) as bad_visibility:
        await streaming_routes.stream_session_unified(
            "some-session", phases="all", visibility="secret"
        )
    assert bad_visibility.value.status_code == 400


@pytest.mark.asyncio
async def test_session_stream_route_uses_the_session_project(monkeypatch):
    session_id = f"session-{uuid4()}"
    captured: Dict[str, Any] = {}

    class _FakeManager:
        def __init__(self, project_id, manager_session_id, phases, visibility):
            captured["project_id"] = project_id
            captured["session_id"] = manager_session_id
            captured["phases"] = phases
            captured["visibility"] = visibility

        async def stream(self):
            yield b"data: {}\n\n"

    monkeypatch.setattr(streaming_routes, "SessionStreamManager", _FakeManager)

    original_lock = app_state_module.active_sessions_lock
    monkeypatch.setattr(app_state_module, "active_sessions_lock", None)
    app_state_module.active_sessions[session_id] = {"project_id": "project-of-session"}
    try:
        response = await streaming_routes.stream_session_unified(
            session_id,
            phases="generation,status",
            visibility="canonical",
        )
    finally:
        app_state_module.active_sessions.pop(session_id, None)
        monkeypatch.setattr(app_state_module, "active_sessions_lock", original_lock)

    assert response.media_type == "text/event-stream"
    assert captured["project_id"] == "project-of-session"
    assert captured["session_id"] == session_id
    assert captured["phases"] == {"generation", "status"}
    assert captured["visibility"] == Visibility.CANONICAL


@pytest.mark.asyncio
async def test_session_without_project_is_rejected(monkeypatch):
    session_id = f"session-{uuid4()}"
    original_lock = app_state_module.active_sessions_lock
    monkeypatch.setattr(app_state_module, "active_sessions_lock", None)
    app_state_module.active_sessions[session_id] = {"project_id": None}
    try:
        with pytest.raises(HTTPException) as excinfo:
            await streaming_routes.stream_session_unified(
                session_id, phases="all", visibility="full"
            )
    finally:
        app_state_module.active_sessions.pop(session_id, None)
        monkeypatch.setattr(app_state_module, "active_sessions_lock", original_lock)

    assert excinfo.value.status_code == 409


def test_console_route_wins_over_the_session_catch_all():
    """/stream/console and /stream/{session_id} share a shape: order decides."""
    from core.app_state import app

    scope = {"type": "http", "method": "GET", "path": "/stream/console"}
    matched = None
    for route in app.router.routes:
        match, _ = route.matches(scope)
        if match.name == "FULL":
            matched = route
            break

    assert matched is not None
    assert matched.path == "/stream/console"

    paths = [getattr(route, "path", None) for route in app.router.routes]
    assert paths.index("/stream/console") < paths.index("/stream/{session_id}")
