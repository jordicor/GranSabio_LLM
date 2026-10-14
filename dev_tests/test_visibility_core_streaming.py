"""Offline contract tests for the core streaming restoration (C1/C2/C3/C15/C16).

They cover the parts of the visibility contract owned by the generation
pipeline and the per-session endpoints:

- the five per-session SSE endpoints reset their cursor when the attempt
  changes or the field shrinks, and gate on shared phase constants,
- the generation stream recorder publishes labelled drafts and keeps the
  thinking view separated from the visible content,
- ``content_filter`` stops purging live surfaces while still purging every
  canonical/structured field,
- the usage block is part of the default result contract.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.app_state as app_state  # noqa: E402
import core.generation_processor as generation_processor  # noqa: E402
import core.streaming_routes as streaming_routes  # noqa: E402
import json_utils as json  # noqa: E402
from ai_service import GenerationStoppedUnexpectedly, StreamChunk  # noqa: E402
from content_events import (  # noqa: E402
    ContentEventType,
    ContentPhase,
    ContentStatus,
    ContentSubphase,
    SessionPhase,
    Visibility,
    live_attempt_key,
)
from models import ContentRequest, GenerationStatus  # noqa: E402
from usage_tracking import UsageTracker  # noqa: E402

# (endpoint, live field, session phase while the producer is live)
_SESSION_ENDPOINTS = [
    ("stream_content", "partial_content", SessionPhase.GENERATING),
    ("stream_generation_content", "generation_content", SessionPhase.GENERATING),
    ("stream_qa_content", "qa_content", SessionPhase.QA_EVALUATION),
    ("stream_preflight_content", "preflight_content", SessionPhase.PREFLIGHT_VALIDATION),
    ("stream_gran_sabio_content", "gransabio_content", SessionPhase.GRAN_SABIO_REVIEW),
]


def _decode_chunks(raw_chunks: List[bytes]) -> List[Dict[str, Any]]:
    """Parse the SSE payloads of a per-session endpoint."""
    events: List[Dict[str, Any]] = []
    for raw in raw_chunks:
        text = raw.decode("utf-8")
        if not text.startswith("data: "):
            continue
        events.append(json.loads(text[len("data: ") :].strip()))
    return events


def _install_scripted_session(
    monkeypatch: pytest.MonkeyPatch,
    states: List[Dict[str, Any]],
) -> None:
    """Serve one scripted session state per polling round."""
    remaining = list(states)

    async def _mutate(_session_id: str, mutator: Any) -> Any:
        result = mutator(remaining[0])
        # The existence probe of the endpoint does not consume a round.
        if isinstance(result, dict) and len(remaining) > 1:
            remaining.pop(0)
        return result

    monkeypatch.setattr(streaming_routes, "mutate_session", _mutate)
    monkeypatch.setattr(streaming_routes, "_SESSION_STREAM_POLL_SECONDS", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_name", "field_name", "session_phase"),
    _SESSION_ENDPOINTS,
    ids=[endpoint for endpoint, _field, _phase in _SESSION_ENDPOINTS],
)
async def test_short_attempt_after_long_one_emits_from_the_start(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
    field_name: str,
    session_phase: str,
) -> None:
    """C2: a shorter new attempt never stalls the length cursor."""
    attempt_key = live_attempt_key(field_name)
    long_draft = "LONG DRAFT THAT GETS REJECTED"
    short_draft = "SHORT"

    def _state(value: str, attempt_id: int, status: Any) -> Dict[str, Any]:
        return {
            "status": status,
            "current_phase": session_phase,
            field_name: value,
            attempt_key: attempt_id,
            "last_generated_content": "",
        }

    _install_scripted_session(
        monkeypatch,
        [
            _state(long_draft, 1, GenerationStatus.GENERATING),
            _state(short_draft, 2, GenerationStatus.GENERATING),
            _state(short_draft, 2, GenerationStatus.FAILED),
        ],
    )

    endpoint = getattr(streaming_routes, endpoint_name)
    response = await endpoint("session-cursor")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert [event["content"] for event in events] == [long_draft, short_draft]
    # The second attempt is a replacement, not a continuation.
    assert events[0].get("reset") is None
    assert events[1]["reset"] is True


@pytest.mark.asyncio
async def test_cursor_resets_when_the_live_field_shrinks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: a replaced field within the same attempt is never a silent blackout."""
    _install_scripted_session(
        monkeypatch,
        [
            {
                "status": GenerationStatus.GENERATING,
                "current_phase": SessionPhase.GENERATING,
                "partial_content": "a long unaccepted draft",
                live_attempt_key("partial_content"): 7,
                "last_generated_content": "",
            },
            {
                "status": GenerationStatus.GENERATING,
                "current_phase": SessionPhase.GENERATING,
                "partial_content": "final",
                live_attempt_key("partial_content"): 7,
                "last_generated_content": "",
            },
            {
                "status": GenerationStatus.COMPLETED,
                "current_phase": SessionPhase.GENERATING,
                "partial_content": "final",
                live_attempt_key("partial_content"): 7,
                "last_generated_content": "final",
            },
        ],
    )

    response = await streaming_routes.stream_content("session-shrink")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert [event["content"] for event in events] == ["a long unaccepted draft", "final"]
    assert events[1]["reset"] is True


@pytest.mark.asyncio
async def test_cursor_resets_when_a_longer_body_does_not_continue_the_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: a LONGER replacement is still a replacement, never a splice.

    A normalized or guard-repaired body (``json.dumps`` re-spacing, repaired
    JSON) can be longer than the raw draft it replaces while sharing no prefix
    with it. Slicing it by the length cursor would glue new text onto the stale
    prefix the consumer already has.
    """
    raw_draft = '{"a":1,"b":2}'
    normalized = '{\n  "a": 1,\n  "b": 2\n}'
    assert len(normalized) > len(raw_draft)
    assert not normalized.startswith(raw_draft)

    def _state(value: str) -> Dict[str, Any]:
        return {
            "status": GenerationStatus.GENERATING,
            "current_phase": SessionPhase.GENERATING,
            "partial_content": value,
            live_attempt_key("partial_content"): 4,
            "last_generated_content": "",
        }

    terminal = _state(normalized)
    terminal["status"] = GenerationStatus.COMPLETED

    _install_scripted_session(
        monkeypatch,
        [_state(raw_draft), _state(normalized), terminal],
    )

    response = await streaming_routes.stream_content("session-splice")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert [event["content"] for event in events] == [raw_draft, normalized]
    assert events[0].get("reset") is None
    # The full replacement travels, flagged: no splice onto the stale prefix.
    assert events[1]["reset"] is True


@pytest.mark.asyncio
async def test_cleared_live_field_tells_the_consumer_to_drop_the_dead_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: an EMPTY replacement still has to reach the consumer.

    The reset flag used to travel only on a chunk that had characters, so a
    failed attempt that cleared the field left the consumer showing text the
    engine had already discarded, with no further poll able to correct it.
    """
    def _state(value: str, status: Any) -> Dict[str, Any]:
        return {
            "status": status,
            "current_phase": SessionPhase.GENERATING,
            "partial_content": value,
            live_attempt_key("partial_content"): 1,
            "last_generated_content": "",
        }

    _install_scripted_session(
        monkeypatch,
        [
            _state("draft of a doomed attempt", GenerationStatus.GENERATING),
            _state("", GenerationStatus.GENERATING),
            _state("", GenerationStatus.FAILED),
        ],
    )

    response = await streaming_routes.stream_content("session-cleared")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert [event["content"] for event in events] == [
        "draft of a doomed attempt",
        "",
    ]
    assert events[0].get("reset") is None
    assert events[1]["reset"] is True
    # The empty reset is announced ONCE, not on every poll from then on.
    assert len(events) == 2


@pytest.mark.asyncio
async def test_terminal_without_accepted_content_clears_what_was_streamed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: the terminal flush cannot go silent when the field died off-phase.

    ``/stream-generation`` only polls while the session is in a generation
    phase. If the attempt fails after the phase moved on, the clear happens
    where the poll cannot see it and the terminal is the last chance to tell the
    consumer that what it holds is dead.
    """
    _install_scripted_session(
        monkeypatch,
        [
            {
                "status": GenerationStatus.GENERATING,
                "current_phase": SessionPhase.GENERATING,
                "generation_content": "draft of a doomed attempt",
                live_attempt_key("generation_content"): 1,
            },
            {
                "status": GenerationStatus.FAILED,
                "current_phase": SessionPhase.QA_EVALUATION,
                "generation_content": "",
                live_attempt_key("generation_content"): 1,
            },
        ],
    )

    response = await streaming_routes.stream_generation_content("session-terminal")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert [event["content"] for event in events] == [
        "draft of a doomed attempt",
        "",
    ]
    assert events[1]["reset"] is True


@pytest.mark.asyncio
async def test_a_stream_that_never_emitted_never_asks_for_a_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer cannot be told to drop content it was never given."""
    _install_scripted_session(
        monkeypatch,
        [
            {
                "status": GenerationStatus.GENERATING,
                "current_phase": SessionPhase.GENERATING,
                "partial_content": "",
                live_attempt_key("partial_content"): 1,
                "last_generated_content": "",
            },
            {
                "status": GenerationStatus.FAILED,
                "current_phase": SessionPhase.GENERATING,
                "partial_content": "",
                live_attempt_key("partial_content"): 2,
                "last_generated_content": "",
            },
        ],
    )

    response = await streaming_routes.stream_content("session-never")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert events == []


@pytest.mark.asyncio
async def test_generation_endpoint_stays_silent_outside_its_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: phase gating uses the shared constants, not a stale literal."""
    _install_scripted_session(
        monkeypatch,
        [
            {
                "status": GenerationStatus.QA_EVALUATION,
                "current_phase": SessionPhase.QA_EVALUATION,
                "generation_content": "draft under evaluation",
                live_attempt_key("generation_content"): 1,
            },
            {
                "status": GenerationStatus.COMPLETED,
                "current_phase": SessionPhase.QA_EVALUATION,
                "generation_content": "draft under evaluation",
                live_attempt_key("generation_content"): 1,
            },
        ],
    )

    response = await streaming_routes.stream_generation_content("session-phase")
    events = _decode_chunks([chunk async for chunk in response.body_iterator])

    assert events == []


def test_session_endpoints_carry_no_hardcoded_phase_literals() -> None:
    """C2: the endpoint module gates on SessionPhase, never on raw strings."""
    source = Path(streaming_routes.__file__).read_text(encoding="utf-8")

    for literal in (
        '"generating"',
        '"qa_evaluation"',
        '"gran_sabio_review"',
        '"gran_sabio_regeneration"',
        '"preflight_validation"',
    ):
        assert literal not in source

    assert SessionPhase.GENERATING == "generating"
    assert SessionPhase.QA_EVALUATION == "qa_evaluation"
    assert generation_processor.SessionPhase is SessionPhase


def test_flush_snapshot_keeps_thinking_and_content_separated() -> None:
    """C2.1: content arriving later never erases the thinking already shown."""
    session: Dict[str, Any] = {}

    generation_processor._flush_stream_session_snapshot(session, [], ["reasoning "])
    assert session["thinking_content"] == "reasoning "
    assert session.get("generation_content") is None

    generation_processor._flush_stream_session_snapshot(
        session,
        ["visible"],
        ["reasoning ", "more"],
    )
    assert session["thinking_content"] == "reasoning more"
    assert session["generation_content"] == "visible"
    assert session["partial_content"] == "visible"
    assert session["generation_content_length"] == len("visible")


@pytest.mark.asyncio
async def test_stream_recorder_publishes_labelled_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: every chunk reaches the bus labelled draft with its coordinates."""
    published: List[Any] = []

    async def _capture(event: Any) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    monkeypatch.setattr(generation_processor, "flush_content_events", AsyncMock())

    session: Dict[str, Any] = {
        "project_id": "project-recorder",
        "project_epoch": 3,
        "request_name": "recorder",
    }
    attempt_id = generation_processor._start_generation_attempt_snapshot(session)
    recorder = generation_processor._GenerationStreamRecorder(
        session,
        "session-recorder",
        iteration=2,
        attempt=attempt_id,
        model="test-model",
        operation="generation.main.streaming",
    )

    await recorder.record("thinking ", is_thinking=True)
    await recorder.record("visible", is_thinking=False)
    await recorder.finish()

    assert [event.content for event in published] == ["thinking ", "visible"]
    for event in published:
        assert event.phase == ContentPhase.GENERATION
        assert event.content_status == ContentStatus.DRAFT
        assert event.event == ContentEventType.CHUNK
        assert event.iteration == 2
        assert event.attempt == attempt_id
        assert event.session_id == "session-recorder"
        assert event.project_id == "project-recorder"
    assert published[0].is_thinking is True
    assert published[0].subphase == ContentSubphase.THINKING
    assert published[1].is_thinking is False
    assert published[1].subphase == ContentSubphase.DRAFT_STREAM

    assert recorder.content == "visible"
    assert session["generation_content"] == "visible"
    assert session["thinking_content"] == "thinking "
    assert session[live_attempt_key("partial_content")] == attempt_id


def test_every_generation_stream_loop_publishes_through_the_recorder() -> None:
    """C2/C3: the JSON retry loop is not a silent second streaming path."""
    tree = ast.parse(Path(generation_processor.__file__).read_text(encoding="utf-8"))

    recorders: set[str] = set()
    driven: Dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "_GenerationStreamRecorder"
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    recorders.add(target.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            driven.setdefault(node.func.value.id, set()).add(node.func.attr)

    # Both loops of the generation phase: the main stream and the JSON retry.
    assert recorders == {"stream_recorder", "json_retry_recorder"}
    for recorder in recorders:
        assert {"record", "finish"} <= driven.get(recorder, set()), (
            f"{recorder} does not publish and close its attempt"
        )


@pytest.mark.asyncio
async def test_new_attempt_bumps_the_live_attempt_id() -> None:
    """C2: consumers can tell one attempt from the next."""
    session: Dict[str, Any] = {}

    first = generation_processor._start_generation_attempt_snapshot(session)
    second = generation_processor._start_generation_attempt_snapshot(session)

    assert second == first + 1
    for field_name in generation_processor._LIVE_GENERATION_FIELDS:
        assert session[live_attempt_key(field_name)] == second


@pytest.mark.asyncio
async def test_iteration_rejected_event_summarizes_without_touching_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: a discarded iteration is announced with its reason."""
    published: List[Any] = []

    async def _capture(event: Any) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)

    session = {
        "project_id": "project-rejected",
        "project_epoch": 1,
        "request_name": "rejected",
        "generation_attempt_id": 4,
    }
    await generation_processor._publish_iteration_rejected(
        session,
        "session-rejected",
        iteration=2,
        consensus_result=SimpleNamespace(average_score=6.25),
        approval_result={"approved": False, "reason": "Layer style below 8.0", "final_rejection": False},
        deal_breaker_found=True,
    )

    assert len(published) == 1
    event = published[0]
    assert event.event == ContentEventType.ITERATION_REJECTED
    assert event.content_status == ContentStatus.REJECTED
    assert event.iteration == 2
    assert event.attempt == 4
    assert "Layer style below 8.0" in event.reason
    assert "average_score=6.25" in event.reason
    assert "deal_breakers=yes" in event.reason
    assert event.data["final_rejection"] is False


def _filtered_generation_request() -> SimpleNamespace:
    """Minimal request the non-tools generation path needs."""
    return SimpleNamespace(
        generator_model="gpt-4o",
        generation_tools_mode="never",
        max_iterations=1,
        temperature=0.2,
        max_tokens=64,
        system_prompt=None,
        extra_verbose=False,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        content_type="article",
        json_output=False,
        json_schema=None,
        json_expectations=None,
        target_field=None,
        min_global_score=8.0,
        llm_accent_guard=SimpleNamespace(
            mode="off",
            on_error="fail_open",
            criteria=None,
            min_score=None,
            max_inline_calls=1,
        ),
        _model_alias_registry=None,
        _llm_routing=None,
    )


@pytest.mark.asyncio
async def test_content_filter_keeps_the_labelled_draft_through_clear_and_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C16 end to end: the pipeline order itself must preserve the draft.

    The regression this replaces called the purge in isolation and passed while
    the pipeline erased the draft one step earlier: the ``finally`` of the
    streaming loop cleared the live snapshot of every unaccepted attempt, and a
    content-filter stop is by definition unaccepted. The sequence exercised here
    is the real one - stream, terminal stop, ``finally``, then the field-by-field
    purge - and it ends on the late-join snapshot a subscriber would receive.
    """
    filtered_draft = "PROVIDER_DRAFT_BEFORE_THE_POLICY_STOP"
    session: Dict[str, Any] = {
        "project_id": "project-filter-order",
        "project_epoch": 2,
        "request_name": "filter order",
        "last_generated_content": "accepted body of an earlier iteration",
        "qa_content": "qa draft already shown",
    }

    async def _filtered_stream():
        yield filtered_draft
        yield StreamChunk(
            "",
            metadata={
                "provider": "test-provider",
                "finish_reason": "content_filter",
                "finish_reason_category": "content_filter",
                "finish_unusable": True,
                "output_truncated": False,
            },
        )

    monkeypatch.setattr(
        generation_processor,
        "_should_use_generation_tools",
        lambda _request: False,
    )
    monkeypatch.setattr(
        generation_processor,
        "_resolve_generation_reasoning_telemetry",
        lambda _request: {
            "is_thinking_model": False,
            "reasoning_effort": None,
            "thinking_budget_tokens": None,
            "requested_reasoning_effort": None,
            "requested_thinking_budget_tokens": None,
        },
    )
    monkeypatch.setattr(
        generation_processor,
        "check_session_cancelled",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(generation_processor, "add_verbose_log", AsyncMock())
    monkeypatch.setattr(generation_processor, "_debug_record_event", AsyncMock())
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        AsyncMock(),
    )
    monkeypatch.setattr(generation_processor, "publish_content_event", AsyncMock())
    monkeypatch.setattr(generation_processor, "flush_content_events", AsyncMock())

    request = _filtered_generation_request()
    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a short article.",
            request=request,
            ai_service=SimpleNamespace(
                generate_content_stream=lambda **_kwargs: _filtered_stream(),
            ),
            usage_tracker=None,
            session_id="session-filter-order",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    # Step 1: the ``finally`` of the streaming loop no longer erases it.
    assert session["generation_content"] == filtered_draft
    assert session["partial_content"] == filtered_draft

    # Step 2: the terminal result builder purges the canonical state only.
    final_result = generation_processor._build_generation_error_result(
        session,
        request,
        raised.value,
    )
    assert final_result["content"] == ""
    assert final_result["error_code"] == "content_filter"
    assert final_result["retryable"] is False
    assert session["generation_content"] == filtered_draft
    assert session["qa_content"] == "qa draft already shown"
    assert session["last_generated_content"] == ""

    # Step 3: a late-joining subscriber sees it, labelled for what it is.
    snapshot = app_state._build_monitor_content_snapshot(session)
    assert snapshot["generation"]["text"] == filtered_draft
    assert snapshot["generation"]["content_status"] == ContentStatus.FILTERED
    assert snapshot["qa"]["content_status"] == ContentStatus.FILTERED
    # Canonical subscribers still get nothing: the accepted text was purged.
    assert (
        app_state._build_monitor_content_snapshot(
            session,
            visibility=Visibility.CANONICAL,
        )
        == {}
    )


def test_content_filter_purge_keeps_live_surfaces_and_clears_canonical() -> None:
    """C16: field by field, live drafts survive labelled, canonical is purged."""
    session: Dict[str, Any] = {
        "generation_content": "live draft",
        "generation_content_length": 10,
        "generation_content_word_count": 2,
        "last_generated_content": "accepted earlier",
        "last_generated_content_length": 16,
        "last_generated_content_word_count": 2,
        "partial_content": "live partial",
        "thinking_content": "live thinking",
        "auto_qa_content": "auto qa draft",
        "preflight_content": "preflight draft",
        "qa_content": "qa draft",
        "arbiter_content": "arbiter draft",
        "gransabio_content": "gran sabio draft",
        "_qa_content_stream_pending": ["tail"],
        "_qa_content_stream_last_flush": 12.0,
        "json_parsed_content": {"a": 1},
        "gran_sabio_json_parsed_content": {"b": 2},
        "json_context": {"original": "x"},
        "last_json_error": {"failed_content": "y"},
        "smart_edit_data": {"base_content": "z"},
        "long_text_state": {"section": "s"},
        "generation_tool_metadata": {"result": "r"},
        "json_guard_history": [{"iteration": 1}],
        "iterations": [
            {
                "content": "iteration body",
                "content_summary": "summary",
                "json_guard": {"result": {}},
            }
        ],
    }

    generation_processor._purge_content_filter_session_state(session)

    for live_field in (
        "generation_content",
        "partial_content",
        "thinking_content",
        "auto_qa_content",
        "preflight_content",
        "qa_content",
        "arbiter_content",
        "gransabio_content",
    ):
        assert session[live_field], live_field
    assert session["generation_content_length"] == 10
    assert session["_qa_content_stream_pending"] == ["tail"]
    assert session["_qa_content_stream_last_flush"] == 12.0

    assert session["last_generated_content"] == ""
    assert session["last_generated_content_length"] == 0
    assert session["last_generated_content_word_count"] == 0
    for structured_field in (
        "json_parsed_content",
        "gran_sabio_json_parsed_content",
        "json_context",
        "last_json_error",
        "smart_edit_data",
        "long_text_state",
        "generation_tool_metadata",
    ):
        assert session[structured_field] is None
    assert session["json_guard_history"] == []
    assert session["iterations"] == [
        {"content": "", "content_summary": "", "json_guard": None}
    ]


def test_snapshot_labels_the_auto_qa_summary_as_the_commit_it_is() -> None:
    """C0.1: the snapshot label must agree with what the live stream said.

    ``auto_qa_content`` is written once by the planner, AFTER validating the QA
    contract, and published labelled ``accepted``. Calling it ``draft`` in the
    snapshot - or ``filtered`` because the LATER generation hit a policy stop -
    would contradict the live stream about the very same text.
    """
    session: Dict[str, Any] = {
        "auto_qa_content": "Auto-QA plan: 3 layers, rigor standard",
        "qa_content": "qa draft in flight",
        "generation_content": "live draft",
    }

    snapshot = app_state._build_monitor_content_snapshot(session)

    assert snapshot["auto_qa"]["content_status"] == ContentStatus.ACCEPTED
    assert snapshot["qa"]["content_status"] == ContentStatus.DRAFT

    session["finish_reason_category"] = "content_filter"
    filtered_snapshot = app_state._build_monitor_content_snapshot(session)

    assert filtered_snapshot["auto_qa"]["content_status"] == ContentStatus.ACCEPTED
    assert filtered_snapshot["qa"]["content_status"] == ContentStatus.FILTERED
    assert filtered_snapshot["generation"]["content_status"] == ContentStatus.FILTERED


def test_error_result_content_never_falls_back_to_the_live_draft() -> None:
    """C2.4: only accepted content can reach a canonical payload."""
    session = {"generation_content": "unvalidated draft"}

    assert generation_processor._get_error_result_content(session, default="") == ""

    session["last_generated_content"] = "accepted body"
    assert (
        generation_processor._get_error_result_content(session, default="")
        == "accepted body"
    )


def _unusable_finish(category: str, *, partial: str) -> GenerationStoppedUnexpectedly:
    return GenerationStoppedUnexpectedly(
        f"provider stopped with {category}",
        provider="test-provider",
        model_id="gpt-4o",
        finish_reason=category,
        finish_reason_category=category,
        partial_content=partial,
    )


@pytest.mark.asyncio
async def test_filtered_verdict_is_emitted_only_for_policy_stops_with_a_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6: no phantom verdicts, and no ``filtered`` label on other stops."""
    published: List[Any] = []

    async def _capture(event: Any) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    monkeypatch.setattr(generation_processor, "flush_content_events", AsyncMock())

    session: Dict[str, Any] = {
        "project_id": "project-verdict",
        "project_epoch": 1,
        "request_name": "verdict",
    }
    recorder = generation_processor._GenerationStreamRecorder(
        session,
        "session-verdict",
        iteration=1,
        attempt=1,
        model="gpt-4o",
        operation="generation.main.streaming",
    )
    filtered = _unusable_finish("content_filter", partial="")
    other = _unusable_finish("pause_turn", partial="PARTIAL")

    async def _publish(exc: GenerationStoppedUnexpectedly) -> None:
        await generation_processor._publish_content_filter_terminal(
            session,
            "session-verdict",
            exc=exc,
            recorder=recorder,
            layer="generation.main.streaming",
            iteration=1,
            attempt=1,
            model="gpt-4o",
        )

    # Nothing streamed: not even a policy stop has a draft to supersede.
    await _publish(filtered)
    assert published == []

    await recorder.record("visible draft", is_thinking=False)
    published.clear()

    # An unusable finish that is not a policy stop keeps its own terminal path.
    await _publish(other)
    assert published == []

    await _publish(filtered)
    assert len(published) == 1
    assert published[0].content_status == ContentStatus.FILTERED
    assert published[0].content == ""


@pytest.mark.parametrize(
    "finish_category",
    ["pause_turn", "provider_error", "unknown"],
)
def test_unusable_finish_never_leaks_its_partial_draft_into_the_result(
    finish_category: str,
) -> None:
    """C2.4: an unusable finish is unusable for every category, not only filter.

    The ``GenerationStoppedUnexpectedly`` branch of the result builder used to
    embed ``exc.partial_content`` verbatim for every non-filter category, which
    is the raw join of the streamed chunks - a draft that passed no validation.
    It is the only builder branch that ever did, and it does not any more.
    """
    partial = "MID_STREAM_PARTIAL_THAT_NEVER_VALIDATED"
    session: Dict[str, Any] = {
        "current_iteration": 1,
        # The live surface holds the same draft; it is not canonical either.
        "generation_content": partial,
    }
    request = _filtered_generation_request()
    stop = _unusable_finish(finish_category, partial=partial)

    result = generation_processor._build_generation_error_result(
        session,
        request,
        stop,
    )

    assert result["content"] == ""
    assert partial not in str(result)
    assert result["error_code"] == "unusable_finish_reason"
    assert result["finish_reason_category"] == finish_category
    assert result["retryable"] is False

    # When an earlier iteration WAS accepted, that body is what travels.
    session["last_generated_content"] = "accepted body of an earlier iteration"
    recovered = generation_processor._build_generation_error_result(
        session,
        request,
        stop,
    )
    assert recovered["content"] == "accepted body of an earlier iteration"


def _usage_request(**overrides: Any) -> ContentRequest:
    payload: Dict[str, Any] = {
        "prompt": "Write a short paragraph about testing.",
        "content_type": "article",
        "generator_model": "gpt-4o",
    }
    payload.update(overrides)
    return ContentRequest(**payload)


def _tracker_with_usage() -> UsageTracker:
    tracker = UsageTracker(detail_level=1)
    tracker.record(
        phase="generation",
        role="generator",
        model="gpt-4o",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
    )
    return tracker


def test_usage_block_is_part_of_the_default_result_contract() -> None:
    """C15: costs and stats travel by default, without touching the content."""
    request = _usage_request()
    assert request.show_query_costs == 1
    assert request.show_query_stats == 1

    session = {"usage_tracker": _tracker_with_usage()}
    result: Dict[str, Any] = {"content": "generated body"}
    generation_processor._attach_usage_metadata(session, result, request)

    assert result["costs"]["grand_totals"]["total_tokens"] == 150
    assert result["query_stats"]
    # The default never mutates canonical content.
    assert result["content"] == "generated body"


def test_usage_block_can_be_opted_out_and_embedded_on_request() -> None:
    """C15: 0 opts out; an explicit level keeps the legacy in-content block."""
    session = {"usage_tracker": _tracker_with_usage()}

    opted_out: Dict[str, Any] = {"content": "generated body"}
    generation_processor._attach_usage_metadata(
        session,
        opted_out,
        _usage_request(show_query_costs=0, show_query_stats=0),
    )
    assert "costs" not in opted_out
    assert "query_stats" not in opted_out
    assert opted_out["content"] == "generated body"

    embedded: Dict[str, Any] = {"content": "generated body"}
    generation_processor._attach_usage_metadata(
        session,
        embedded,
        _usage_request(show_query_costs=1),
    )
    assert "costs" in embedded
    assert embedded["content"].startswith("generated body")
    assert "QUERY_COSTS" in embedded["content"]
