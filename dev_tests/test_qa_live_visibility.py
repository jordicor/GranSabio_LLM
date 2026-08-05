"""Offline contract tests for live QA visibility (fiches C4 and C10).

C10 - QA progress follows the scheduler lifecycle: an evaluation counts when
it is dispatched and when it finishes, never when it happens to stream a
chunk. Concurrency, retries, algorithmic bypass and skipped evaluators are all
covered, and completion is idempotent.

C4 - QA content is emitted live, one labelled event per provider chunk, with
the provider-reported thinking flag preserved and each provider attempt of the
same evaluation carrying its own attempt id.

Everything here is deterministic and offline: no provider is contacted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.generation_processor as generation_processor
import qa_engine
from ai_service import StreamChunk
from content_events import ContentEvent, ContentStatus, ContentSubphase
from content_stream import ContentStreamCoalescer
from core.app_state import (
    claim_qa_stream_header,
    update_qa_evaluation_completed,
    update_qa_evaluation_started,
    update_qa_progress_reset,
)
from models import QAEvaluation, QALayer
from qa_engine import QAEngine, qa_evaluation_key
from qa_evaluation_service import QAEvaluationService
from qa_scheduler import (
    QAEvaluationLifecycleEvent,
    QAScheduler,
    QASchedulerPolicy,
    QASchedulerSlot,
    QASchedulerTechnicalFailure,
)

SESSION_ID = "session-qa-visibility"


def _session() -> Dict[str, Any]:
    """A session with no project: status events short-circuit, counters do not."""
    return {"request_name": "test-request"}


def _qa_consumer(
    session: Dict[str, Any],
) -> Tuple[Any, List[ContentEvent], List[QAEvaluationLifecycleEvent]]:
    """Mirror of the QA stream consumer wired in the generation processor."""
    content_events: List[ContentEvent] = []
    lifecycle_events: List[QAEvaluationLifecycleEvent] = []

    async def consumer(data: Any) -> None:
        if isinstance(data, ContentEvent):
            content_events.append(data)
            return
        if isinstance(data, QAEvaluationLifecycleEvent):
            lifecycle_events.append(data)
            if data.dispatched:
                update_qa_evaluation_started(
                    session,
                    SESSION_ID,
                    data.model_name,
                    data.layer_name,
                )
            else:
                update_qa_evaluation_completed(
                    session,
                    SESSION_ID,
                    data.evaluation_key,
                )
            return
        raise TypeError(f"Unsupported QA stream payload: {type(data).__name__}")

    return consumer, content_events, lifecycle_events


def _layer(name: str = "Accuracy", deal_breaker_criteria: Optional[str] = None) -> QALayer:
    return QALayer(
        name=name,
        description="Check facts.",
        criteria="Preserve facts.",
        deal_breaker_criteria=deal_breaker_criteria,
        min_score=8.0,
        order=1,
    )


def _request(execution_mode: str = "parallel") -> SimpleNamespace:
    return SimpleNamespace(
        qa_execution_mode=execution_mode,
        smart_editing_mode="never",
        content_type="other",
        extra_verbose=False,
        qa_timeout_retries=0,
        qa_parse_error_retries=1,
        _cancellation_token=None,
    )


def _evaluation(model: str, layer_name: str, *, deal_breaker: bool = False) -> QAEvaluation:
    return QAEvaluation(
        model=model,
        layer=layer_name,
        score=9.0,
        feedback="Fine.",
        deal_breaker=deal_breaker,
        deal_breaker_reason="Broken." if deal_breaker else None,
        passes_score=True,
    )


def _slots(model_names: List[str]) -> List[QASchedulerSlot]:
    return [
        QASchedulerSlot(
            index=index,
            model=name,
            model_name=name,
            result_key=name,
            evaluator_label=name,
            timeout_seconds=30.0,
        )
        for index, name in enumerate(model_names)
    ]


# ---------------------------------------------------------------------------
# C10 - progress counter from the scheduler lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qa_counter_reaches_total_with_concurrent_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent evaluators all count, and the counter lands on the total.

    The evaluations here stream NOTHING: the counter must not depend on
    chunks arriving.
    """
    models = ["model-a", "model-b", "model-c"]
    layers = [_layer("Accuracy"), _layer("Style")]
    session = _session()
    update_qa_progress_reset(session, SESSION_ID, len(models) * len(layers))
    consumer, content_events, lifecycle_events = _qa_consumer(session)

    engine = QAEngine(ai_service=MagicMock(), bypass_engine=MagicMock())
    engine._admit_qa_fanout = AsyncMock()
    monkeypatch.setattr(qa_engine, "resolve_qa_timeout_for_model", lambda *_a, **_k: 30.0)

    in_flight = 0
    peak_in_flight = 0
    all_dispatched = asyncio.Event()

    async def evaluate_content(_content, layer, model, *_args, **_kwargs):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        if in_flight == len(models):
            all_dispatched.set()
        await all_dispatched.wait()
        in_flight -= 1
        return _evaluation(model, layer.name)

    engine.evaluate_content = evaluate_content

    for layer in layers:
        all_dispatched = asyncio.Event()
        in_flight = 0
        await engine._evaluate_ai_layer_with_scheduler(
            content="Candidate.",
            layer=layer,
            qa_models=list(models),
            qa_model_names=list(models),
            original_request=_request(),
            stream_callback=consumer,
        )

    assert peak_in_flight == len(models)
    assert content_events == []
    assert session["qa_evaluations_completed"] == session["qa_evaluations_total"] == 6
    assert sum(1 for event in lifecycle_events if event.dispatched) == 6
    assert sum(1 for event in lifecycle_events if not event.dispatched) == 6


@pytest.mark.asyncio
async def test_qa_counter_counts_algorithmic_bypass_without_any_chunk() -> None:
    """A bypassed layer never reaches a provider, yet it is an evaluation."""
    models = ["model-a", "model-b"]
    layer = _layer("Word Count")
    session = _session()
    update_qa_progress_reset(session, SESSION_ID, len(models))
    consumer, content_events, _lifecycle = _qa_consumer(session)

    bypass_engine = MagicMock()
    bypass_engine.should_bypass_qa_layer.return_value = True
    bypass_engine.bypass_layer_evaluation.return_value = {
        name: _evaluation(name, layer.name) for name in models
    }
    engine = QAEngine(ai_service=MagicMock(), bypass_engine=bypass_engine)

    layer_results, deal_breaker = await engine._evaluate_single_semantic_layer(
        content="Candidate.",
        layer=layer,
        qa_models=list(models),
        qa_model_names=list(models),
        original_request=_request(),
        stream_callback=consumer,
    )

    assert deal_breaker is None
    assert len(layer_results) == len(models)
    assert content_events == []
    assert session["qa_evaluations_completed"] == session["qa_evaluations_total"] == 2


@pytest.mark.asyncio
async def test_qa_counter_completion_is_idempotent_across_retries_and_failures() -> None:
    """Retries and raised failures report completion exactly once."""
    session = _session()
    update_qa_progress_reset(session, SESSION_ID, 1)
    consumer, _content, lifecycle_events = _qa_consumer(session)

    slots = _slots(["model-a"])
    attempts = 0

    async def evaluate_slot(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
        nonlocal attempts
        attempts += 1
        raise QASchedulerTechnicalFailure(
            "parse_error",
            "invalid json",
            metadata={"attempts": attempt},
        )

    async def dispatched(slot: QASchedulerSlot, layer_name: str) -> None:
        await consumer(
            QAEvaluationLifecycleEvent(
                layer_name=layer_name,
                model_name=slot.model_name,
                evaluation_key=qa_evaluation_key(layer_name, slot.result_key),
                dispatched=True,
            )
        )

    async def finished(slot: QASchedulerSlot, layer_name: str) -> None:
        await consumer(
            QAEvaluationLifecycleEvent(
                layer_name=layer_name,
                model_name=slot.model_name,
                evaluation_key=qa_evaluation_key(layer_name, slot.result_key),
                dispatched=False,
            )
        )

    scheduler = QAScheduler(
        QASchedulerPolicy(
            execution_mode="parallel",
            min_valid_models=1,
            parse_error_retries=1,
            on_model_unavailable="skip_if_quorum",
        )
    )
    with pytest.raises(Exception):
        await scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=slots,
            evaluate_slot=evaluate_slot,
            on_evaluation_dispatched=dispatched,
            on_evaluation_finished=finished,
        )

    assert attempts == 2, "the parse-error retry must be spent inside one evaluation"
    assert sum(1 for event in lifecycle_events if not event.dispatched) == 1
    assert session["qa_evaluations_completed"] == 1

    # A repeated completion for the same evaluation never double counts.
    update_qa_evaluation_completed(
        session,
        SESSION_ID,
        qa_evaluation_key("Accuracy", "model-a"),
    )
    assert session["qa_evaluations_completed"] == 1


@pytest.mark.asyncio
async def test_qa_counter_reaches_total_when_the_layer_skips_evaluators() -> None:
    """Evaluators skipped once the layer has its answer still finish."""
    session = _session()
    update_qa_progress_reset(session, SESSION_ID, 3)
    consumer, _content, lifecycle_events = _qa_consumer(session)

    slots = _slots(["model-a", "model-b", "model-c"])
    evaluated: List[str] = []

    async def evaluate_slot(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
        evaluated.append(slot.model_name)
        return _evaluation(slot.model_name, "Accuracy", deal_breaker=True)

    async def lifecycle(slot: QASchedulerSlot, layer_name: str, *, dispatched: bool) -> None:
        await consumer(
            QAEvaluationLifecycleEvent(
                layer_name=layer_name,
                model_name=slot.model_name,
                evaluation_key=qa_evaluation_key(layer_name, slot.result_key),
                dispatched=dispatched,
            )
        )

    scheduler = QAScheduler(QASchedulerPolicy(execution_mode="progressive_quorum"))
    await scheduler.evaluate_layer(
        layer_name="Accuracy",
        has_deal_breaker_criteria=True,
        slots=slots,
        evaluate_slot=evaluate_slot,
        on_evaluation_dispatched=lambda slot, layer: lifecycle(slot, layer, dispatched=True),
        on_evaluation_finished=lambda slot, layer: lifecycle(slot, layer, dispatched=False),
    )

    assert len(evaluated) < len(slots), "progressive quorum must skip at least one slot"
    assert sum(1 for event in lifecycle_events if not event.dispatched) == 3
    assert session["qa_evaluations_completed"] == session["qa_evaluations_total"] == 3


def test_qa_stream_header_is_claimed_once_per_model_and_layer() -> None:
    """Interleaved evaluations each get exactly one header."""
    session = _session()
    update_qa_progress_reset(session, SESSION_ID, 4)

    interleaved = [
        ("model-a", "Accuracy"),
        ("model-b", "Accuracy"),
        ("model-a", "Accuracy"),
        ("model-a", "Style"),
        ("model-b", "Accuracy"),
        ("model-b", "Style"),
        ("model-a", "Style"),
    ]
    claimed = [claim_qa_stream_header(session, model, layer) for model, layer in interleaved]

    assert claimed == [True, True, False, True, False, True, False]

    # A new QA run starts over with fresh headers.
    update_qa_progress_reset(session, SESSION_ID, 4)
    assert claim_qa_stream_header(session, "model-a", "Accuracy") is True


# ---------------------------------------------------------------------------
# C4 - live QA content
# ---------------------------------------------------------------------------


class _FakeStreamingService:
    """Minimal AI service double that replays scripted stream responses."""

    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_content_stream(self, **_kwargs: Any):
        payload = self._responses[self.calls]
        self.calls += 1

        async def _stream():
            if isinstance(payload, Exception):
                raise payload
            for chunk in payload:
                yield chunk

        return _stream()


_PASSING_QA_JSON = (
    '{"score": 9, "feedback": "Passed", "deal_breaker": false, '
    '"deal_breaker_reason": null}'
)


@pytest.mark.asyncio
async def test_qa_chunks_are_delivered_live_and_keep_the_thinking_flag() -> None:
    events: List[ContentEvent] = []

    async def callback(event: ContentEvent) -> None:
        events.append(event)

    async def scripted():
        yield StreamChunk("reasoning...", is_thinking=True)
        # Live delivery: the thinking chunk is already out before the
        # evaluation produces (or parses) anything else.
        assert len(events) == 1
        yield StreamChunk(_PASSING_QA_JSON)
        assert len(events) == 2

    service = SimpleNamespace(generate_content_stream=lambda **_kwargs: scripted())
    evaluator = QAEvaluationService(service)

    result = await evaluator.evaluate_content(
        content="Candidate.",
        criteria="Check accuracy.",
        model="gpt-4o",
        layer_name="Accuracy",
        min_score=8.0,
        original_request=_request(),
        request_edit_info=False,
        max_tokens=128,
        stream_callback=callback,
        session_id=SESSION_ID,
        project_id="project-1",
    )

    assert result.score == 9.0
    assert len(events) == 2

    thinking, visible = events
    assert thinking.is_thinking is True
    assert thinking.subphase == ContentSubphase.THINKING
    assert thinking.content == "reasoning..."
    assert visible.is_thinking is False
    assert visible.subphase == ContentSubphase.DRAFT_STREAM
    assert visible.content == _PASSING_QA_JSON

    for event in events:
        assert event.phase == "qa"
        assert event.content_status == ContentStatus.DRAFT
        assert event.model == "gpt-4o"
        assert event.layer == "Accuracy"
        assert event.attempt == 1
        assert event.session_id == SESSION_ID
        assert event.project_id == "project-1"


@pytest.mark.asyncio
async def test_concurrent_qa_evaluations_never_share_a_coalescing_buffer() -> None:
    """Two evaluators streaming at once stay separate and correctly labelled."""
    published: List[Dict[str, Any]] = []

    async def publisher(**kwargs: Any) -> None:
        published.append(kwargs)

    async def trace_writer(**_kwargs: Any) -> None:
        return None

    # A frozen clock keeps every chunk inside the sampling window, so the
    # coalescer really buffers instead of publishing one event per chunk.
    coalescer = ContentStreamCoalescer(
        sampling_interval=0.05,
        trace_interval=0.0,
        clock=lambda: 1000.0,
        publisher=publisher,
        trace_writer=trace_writer,
    )

    for model, text in [("model-a", "A1"), ("model-b", "B1"), ("model-a", "A2")]:
        await coalescer.publish(
            ContentEvent(
                phase="qa",
                content=text,
                model=model,
                layer="Accuracy",
                attempt=1,
                iteration=1,
                subphase=ContentSubphase.DRAFT_STREAM,
                session_id=SESSION_ID,
                project_id="project-1",
            )
        )
    await asyncio.sleep(0.2)

    assert sorted((event["qa_model"], event["content"]) for event in published) == [
        ("model-a", "A1A2"),
        ("model-b", "B1"),
    ]


@pytest.mark.asyncio
async def test_schema_less_retry_labels_its_own_attempt() -> None:
    events: List[ContentEvent] = []

    async def callback(event: ContentEvent) -> None:
        events.append(event)

    service = _FakeStreamingService(
        [
            RuntimeError("Invalid schema for response_format: unsupported schema"),
            [StreamChunk(_PASSING_QA_JSON)],
        ]
    )
    evaluator = QAEvaluationService(service)

    result = await evaluator.evaluate_content(
        content="Candidate.",
        criteria="Check accuracy.",
        model="gpt-4o",
        layer_name="Accuracy",
        min_score=8.0,
        original_request=_request(),
        request_edit_info=True,
        max_tokens=128,
        stream_callback=callback,
        session_id=SESSION_ID,
    )

    assert result.score == 9.0
    assert service.calls == 2
    assert [event.attempt for event in events] == [2]
    assert events[0].content == _PASSING_QA_JSON


# ---------------------------------------------------------------------------
# C0.2: one consumer serves a whole QA run, so IT tells two evaluations of the
# same evaluator apart (outer timeout retry of the fallback, edit rounds of a
# layer, truncation retry inside one evaluation)
# ---------------------------------------------------------------------------


def _qa_tool_event_session() -> Dict[str, Any]:
    """A session with a project, so the QA tool-loop consumer publishes."""
    return {
        "project_id": "project-1",
        "request_name": "test-request",
        "current_iteration": 2,
        "qa_content": "",
    }


def _qa_tool_payload(**overrides: Any) -> Dict[str, Any]:
    """One QA tool-loop event payload as the evaluator enriches it."""
    payload: Dict[str, Any] = {
        "qa_model": "gpt-4o",
        "qa_layer": "Accuracy",
        "model": "gpt-4o",
        "loop_scope": "qa",
        "provider": "openai",
        "api_surface": "responses",
        "turn": 1,
        "index": 0,
    }
    payload.update(overrides)
    return payload


def _qa_tool_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[Any, Dict[str, Any], List[ContentEvent]]:
    """Build the QA tool-loop consumer with its session and captured events."""
    events: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        events.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    session = _qa_tool_event_session()
    callback = generation_processor._make_qa_tool_event_callback(
        "session-qa-tool-events",
        session,
    )
    return callback, session, events


@pytest.mark.asyncio
async def test_qa_tool_loop_restart_opens_a_new_attempt_and_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second tool loop of one evaluator is a new attempt, not a longer one.

    The Gran Sabio fallback restarts its WHOLE comprehensive evaluation when the
    outer wait times out, and a layer re-evaluates itself on every edit round -
    both through the same consumer. Turn 1 of the new loop is the boundary: the
    restarted evaluation must not inherit the attempt id it replaced nor be glued
    to it on the flat ``qa_content`` surface.
    """
    callback, session, events = _qa_tool_consumer(monkeypatch)

    for chunk in ("FIRST", "SECOND"):
        await callback("tool_loop_turn_start", _qa_tool_payload())
        await callback("assistant_delta", _qa_tool_payload(content=chunk))

    assert [event.attempt for event in events] == [1, 1, 2, 2]
    assert session["qa_content"].count("gpt-4o evaluating Accuracy") == 2
    assert session["qa_content"].endswith("SECOND")


@pytest.mark.asyncio
async def test_qa_truncation_retry_never_reuses_the_truncated_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncation retry and the loop it starts both leave attempt 1 behind.

    The telemetry that announces the restart is not filed under the attempt that
    truncated, and the loop it starts opens its own id: the ids skip a number on
    purpose, because a dense numbering would have to assume the announced loop
    really started and would reuse the id of an attempt that already streamed.
    """
    callback, _session, events = _qa_tool_consumer(monkeypatch)

    await callback("tool_loop_turn_start", _qa_tool_payload())
    await callback("assistant_delta", _qa_tool_payload(content="FIRST"))
    await callback(
        generation_processor._QA_TOOL_LOOP_RETRY_EVENT,
        _qa_tool_payload(),
    )
    await callback("tool_loop_turn_start", _qa_tool_payload())
    await callback("assistant_delta", _qa_tool_payload(content="SECOND"))

    attempts = [event.attempt for event in events]
    assert attempts == [1, 1, 2, 3, 3]
    # What the contract needs: the text of the restarted loop never shares an id
    # with the text of the attempt it replaced.
    assert attempts[1] != attempts[-1]


@pytest.mark.asyncio
async def test_qa_restarted_tool_arguments_decode_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new attempt decodes its streamed arguments without the stale prefix.

    The decoding buffer belongs to the attempt: a restarted loop streams its
    ``text`` argument from the beginning, so a buffer carried over from the
    attempt it replaced would publish garbage instead of the new text.
    """
    callback, _session, events = _qa_tool_consumer(monkeypatch)

    await callback("tool_loop_turn_start", _qa_tool_payload())
    await callback("tool_call_delta", _qa_tool_payload(delta='{"text":"AAA'))
    await callback(
        generation_processor._QA_TOOL_LOOP_RETRY_EVENT,
        _qa_tool_payload(),
    )
    await callback("tool_loop_turn_start", _qa_tool_payload())
    await callback("tool_call_delta", _qa_tool_payload(delta='{"text":"BBB'))

    visible = [event.content for event in events if event.content]
    assert len(visible) == 2
    assert visible[0].endswith("AAA")
    assert visible[1].endswith("BBB")
