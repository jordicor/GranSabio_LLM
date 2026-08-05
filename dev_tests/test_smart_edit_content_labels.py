"""Regressions for the ``smart_edit`` phase under the content contract (C7).

Smart edit emits real content in ``edit_data`` (``text_before``/``text_after``),
so its events must say what that content IS: ``edit_start`` is the paragraph
about to be rewritten (``draft``), ``edit_complete`` is the applied result
(``accepted``) and ``edit_error`` is the attempt that failed and raises
(``rejected``). A subscriber asking for ``visibility=canonical`` must therefore
see the commits and errors but not the in-flight drafts.
"""

import ast
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from content_events import ContentStatus, Visibility
from core import app_state as app_state_module
from core.cancellation import cancellation_registry
from services.project_stream import ProjectStreamManager

GENERATION_PROCESSOR = Path(__file__).resolve().parents[1] / "core" / "generation_processor.py"

# event name -> content label the publish call must carry.
EXPECTED_SMART_EDIT_LABELS = {
    "edit_start": ContentStatus.DRAFT,
    "edit_complete": ContentStatus.ACCEPTED,
    "edit_error": ContentStatus.REJECTED,
}


def _keyword_value(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _content_status_constant(node) -> str:
    """Resolve ``ContentStatus.X`` (or a literal) to its string value."""
    if isinstance(node, ast.Constant):
        return node.value
    assert isinstance(node, ast.Attribute), f"Unexpected content_status node: {ast.dump(node)}"
    assert isinstance(node.value, ast.Name) and node.value.id == "ContentStatus"
    return getattr(ContentStatus, node.attr)


def test_smart_edit_publish_sites_carry_their_content_label():
    """Every smart-edit publish call labels its content and locates it."""
    tree = ast.parse(GENERATION_PROCESSOR.read_text(encoding="utf-8"))

    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "publish_project_phase_chunk":
            continue
        event = _keyword_value(node, "event")
        if not isinstance(event, ast.Constant) or event.value not in EXPECTED_SMART_EDIT_LABELS:
            continue

        content_status = _keyword_value(node, "content_status")
        assert content_status is not None, f"{event.value} publishes without content_status"
        found[event.value] = _content_status_constant(content_status)

        assert _keyword_value(node, "iteration") is not None, (
            f"{event.value} publishes without iteration"
        )
        assert _keyword_value(node, "edit_data") is not None, (
            f"{event.value} publishes without edit_data"
        )

    assert found == EXPECTED_SMART_EDIT_LABELS


async def _collect_smart_edit_events(visibility: str):
    """Drive one manager over the smart_edit phase and return what it yielded."""
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)

    manager = ProjectStreamManager(project_id, {"smart_edit"}, visibility)
    stream = manager.stream()
    collected = []

    async def _drain():
        async for chunk in stream:
            collected.append(chunk)

    drain_task = asyncio.create_task(_drain())
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            if manager._content_queues:
                break
        assert manager._content_queues, "manager did not subscribe to the phase"

        for event, content_status in (
            ("edit_start", ContentStatus.DRAFT),
            ("edit_complete", ContentStatus.ACCEPTED),
            ("edit_error", ContentStatus.REJECTED),
        ):
            await app_state_module.publish_project_phase_chunk(
                project_id,
                "smart_edit",
                None,
                session_id=session_id,
                project_epoch=project_epoch,
                event=event,
                content_status=content_status,
                iteration=2,
                edit_data={"paragraph_index": 1, "text_before": "before"},
            )
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "smart_edit",
            None,
            session_id=session_id,
            project_epoch=project_epoch,
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
async def test_full_visibility_streams_the_whole_smart_edit_operation():
    payloads = await _collect_smart_edit_events(Visibility.FULL)
    types = [payload["type"] for payload in payloads]

    assert types[0] == "connected"
    assert types[1:] == [
        "edit_start",
        "edit_complete",
        "edit_error",
        "session_end",
        "stream_end",
    ]
    labelled = {
        payload["type"]: payload.get("content_status")
        for payload in payloads
        if payload["type"] in EXPECTED_SMART_EDIT_LABELS
    }
    assert labelled == EXPECTED_SMART_EDIT_LABELS
    assert all(
        payload["iteration"] == 2
        for payload in payloads
        if payload["type"] in EXPECTED_SMART_EDIT_LABELS
    )


@pytest.mark.asyncio
async def test_canonical_visibility_hides_the_smart_edit_draft_only():
    payloads = await _collect_smart_edit_events(Visibility.CANONICAL)
    types = [payload["type"] for payload in payloads]

    assert "edit_start" not in types
    assert types[1:] == ["edit_complete", "edit_error", "session_end", "stream_end"]
