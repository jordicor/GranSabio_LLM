"""Contract regressions for the transversal content-event fields (C0.1/C0.2/C0.5).

These cover the payload contract itself: the new ``content_status`` label never
collides with the lifecycle ``status``, iteration/attempt travel with the
content, and the per-(session, phase) sequence makes best-effort delivery
auditable instead of silent.
"""

import json
from uuid import uuid4

import pytest

from content_events import (
    ContentEvent,
    ContentEventTemplate,
    ContentStatus,
    ContentSubphase,
    Visibility,
    is_hidden_by_canonical_visibility,
    parse_visibility,
    sink_from_chunk_model_layer_callback,
    sink_from_chunk_model_layer_thinking_callback,
    sink_from_text_callback,
    sink_from_tool_event_callback,
)
from core import app_state as app_state_module
from core.app_state import _build_project_phase_event
from core.cancellation import cancellation_registry


def test_content_status_does_not_collide_with_lifecycle_status():
    payload = json.loads(
        _build_project_phase_event(
            event="chunk",
            phase="generation",
            content="draft text",
            status="cancelled",
            content_status=ContentStatus.DRAFT,
            iteration=2,
            attempt_id=5,
            sequence=17,
        )
    )

    assert payload["status"] == "cancelled"
    assert payload["content_status"] == "draft"
    assert payload["iteration"] == 2
    assert payload["attempt_id"] == 5
    assert payload["sequence"] == 17


def test_content_contract_fields_are_optional():
    payload = json.loads(
        _build_project_phase_event(
            event="chunk",
            phase="generation",
            content="plain chunk",
        )
    )

    assert "content_status" not in payload
    assert "iteration" not in payload
    assert "attempt_id" not in payload
    assert "sequence" not in payload


def test_invalid_content_status_fails_fast():
    with pytest.raises(ValueError):
        _build_project_phase_event(
            event="chunk",
            phase="generation",
            content="draft text",
            content_status="partial",
        )


def test_content_event_rejects_unknown_status_and_missing_phase():
    with pytest.raises(ValueError):
        ContentEvent(phase="generation", content_status="partial")
    with pytest.raises(ValueError):
        ContentEvent(phase="")


def test_content_event_template_builds_labelled_chunks():
    template = ContentEventTemplate(
        phase="qa",
        session_id="session-a",
        project_id="project-a",
        model="fake-model",
        layer="coherence",
        attempt=3,
        iteration=1,
    )

    thinking = template.chunk("reasoning", is_thinking=True, subphase=ContentSubphase.THINKING)
    accepted = template.chunk("verdict", content_status=ContentStatus.ACCEPTED)

    assert thinking.is_thinking is True
    assert thinking.subphase == "thinking"
    assert thinking.content_status == ContentStatus.DRAFT
    assert thinking.is_draft_chunk is True
    assert accepted.content_status == ContentStatus.ACCEPTED
    assert accepted.is_draft_chunk is False
    assert accepted.model == "fake-model"
    assert accepted.attempt == 3


@pytest.mark.asyncio
async def test_publish_stamps_monotonic_sequence_per_session_and_phase():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)
    generation_queue = await app_state_module.subscribe_project_phase(project_id, "generation")
    qa_queue = await app_state_module.subscribe_project_phase(project_id, "qa")

    try:
        for index in range(3):
            await app_state_module.publish_project_phase_chunk(
                project_id,
                "generation",
                f"chunk-{index}",
                session_id=session_id,
                project_epoch=project_epoch,
                content_status=ContentStatus.DRAFT,
                iteration=1,
                attempt_id=1,
            )
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "qa",
            "qa-chunk",
            session_id=session_id,
            project_epoch=project_epoch,
            content_status=ContentStatus.DRAFT,
        )

        generation_sequences = [
            json.loads(generation_queue.get_nowait())["sequence"] for _ in range(3)
        ]
        qa_sequences = [json.loads(qa_queue.get_nowait())["sequence"]]
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", generation_queue)
        await app_state_module.unsubscribe_project_phase(project_id, "qa", qa_queue)
        app_state_module.reset_session_event_sequences(session_id)

    assert generation_sequences == [1, 2, 3]
    assert qa_sequences == [1]


@pytest.mark.asyncio
async def test_dropped_events_leave_a_detectable_sequence_gap():
    project_id = f"test-project-{uuid4()}"
    session_id = f"test-session-{uuid4()}"
    project_epoch = await cancellation_registry.begin_project_admission(project_id)
    queue = await app_state_module.subscribe_project_phase(project_id, "generation")

    async def _publish(content: str) -> None:
        await app_state_module.publish_project_phase_chunk(
            project_id,
            "generation",
            content,
            session_id=session_id,
            project_epoch=project_epoch,
            content_status=ContentStatus.DRAFT,
        )

    try:
        await _publish("first")
        first = json.loads(queue.get_nowait())

        # Slow subscriber: the queue is full, the next event is dropped.
        for _ in range(queue.maxsize):
            queue.put_nowait("filler")
        await _publish("lost")
        while not queue.empty():
            queue.get_nowait()

        await _publish("third")
        third = json.loads(queue.get_nowait())
    finally:
        await app_state_module.unsubscribe_project_phase(project_id, "generation", queue)
        app_state_module.reset_session_event_sequences(session_id)

    assert first["sequence"] == 1
    assert third["sequence"] == 3, "a dropped event must leave a gap, not renumber"


@pytest.mark.asyncio
async def test_session_sequences_are_released_with_the_session():
    session_id = f"test-session-{uuid4()}"
    app_state_module._session_phase_sequences[session_id] = {"generation": 4}

    app_state_module.cleanup_session_sidecars(session_id)

    assert session_id not in app_state_module._session_phase_sequences


@pytest.mark.asyncio
async def test_adapters_preserve_the_structured_fields():
    event = ContentEvent(
        phase="qa",
        content="partial verdict",
        is_thinking=True,
        model="fake-model",
        layer="coherence",
        attempt=2,
        iteration=3,
        session_id="session-a",
    )

    text_calls = []
    await sink_from_text_callback(lambda chunk: text_calls.append(chunk))(event)

    three_arg_calls = []

    async def _three_arg(chunk, model, layer):
        three_arg_calls.append((chunk, model, layer))

    await sink_from_chunk_model_layer_callback(_three_arg)(event)

    four_arg_calls = []

    async def _four_arg(chunk, model, layer, is_thinking):
        four_arg_calls.append((chunk, model, layer, is_thinking))

    await sink_from_chunk_model_layer_thinking_callback(_four_arg)(event)

    tool_calls = []

    async def _tool(event_type, payload):
        tool_calls.append((event_type, payload))

    await sink_from_tool_event_callback(_tool)(event)

    assert text_calls == ["partial verdict"]
    assert three_arg_calls == [("partial verdict", "fake-model", "coherence")]
    assert four_arg_calls == [("partial verdict", "fake-model", "coherence", True)]
    assert tool_calls[0][0] == "chunk"
    assert tool_calls[0][1]["is_thinking"] is True
    assert tool_calls[0][1]["attempt_id"] == 2
    assert tool_calls[0][1]["iteration"] == 3
    assert tool_calls[0][1]["content_status"] == ContentStatus.DRAFT


def test_parse_visibility_defaults_to_full_and_rejects_unknown():
    assert parse_visibility(None) == Visibility.FULL
    assert parse_visibility("") == Visibility.FULL
    assert parse_visibility("CANONICAL") == Visibility.CANONICAL

    with pytest.raises(ValueError):
        parse_visibility("admin")


def test_canonical_visibility_hides_only_drafts_and_thinking():
    assert is_hidden_by_canonical_visibility(
        content_status=ContentStatus.DRAFT, subphase="draft_stream", is_thinking=None
    )
    assert is_hidden_by_canonical_visibility(
        content_status=None, subphase=ContentSubphase.THINKING, is_thinking=None
    )
    assert is_hidden_by_canonical_visibility(
        content_status=None, subphase=None, is_thinking=True
    )
    assert not is_hidden_by_canonical_visibility(
        content_status=ContentStatus.ACCEPTED, subphase="accepted_stream", is_thinking=None
    )
    assert not is_hidden_by_canonical_visibility(
        content_status=ContentStatus.REJECTED, subphase=None, is_thinking=False
    )
    assert not is_hidden_by_canonical_visibility(
        content_status=None, subphase=None, is_thinking=None
    )
