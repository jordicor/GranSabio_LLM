"""Regressions for live labelled tool-loop visibility (C6).

The tool loop used to buffer every non-terminal event and DISCARD the whole
batch when the envelope was not accepted, and the generation consumer published
those events stripped of their content. Under the restored contract the loop
emits as it happens, each event carries the honest label of what it is, and a
rejected attempt is announced and logged instead of being silently dropped.

Everything here is offline and deterministic: no provider is contacted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch

import pytest

import core.generation_processor as generation_processor
import json_utils as json
from ai_service import (
    AIService,
    GenerationStoppedUnexpectedly,
    _tool_loop_event_content_status,
    is_content_filter_stop,
)
from content_events import (
    ContentEvent,
    ContentStatus,
    ContentSubphase,
    is_hidden_by_canonical_visibility,
    live_attempt_key,
)
from content_stream import ContentStreamCoalescer
from core.app_state import _build_project_phase_event
from tool_loop_models import LoopScope, ToolLoopEnvelope


class _FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ToolLoopHarness:
    """Context manager that runs ``call_ai_with_validation_tools`` offline.

    Only the provider dispatch is faked; the labelling wrapper, the envelope
    construction and the acceptance branch under test are the real code.
    """

    def __init__(self, provider_loop: Any) -> None:
        self._provider_loop = provider_loop
        self._patches: List[Any] = []

    def __enter__(self) -> None:
        self._patches = [
            patch("ai_service.config"),
            patch.object(
                AIService,
                "_normalize_tool_loop_provider",
                return_value="openai",
            ),
            patch.object(
                AIService,
                "_is_openai_responses_api_model",
                return_value=False,
            ),
            patch.object(AIService, "_supports_tool_calling", return_value=True),
            patch.object(
                AIService,
                "_apply_temperature_policies",
                return_value=(0.7, None, False),
            ),
            patch.object(
                AIService,
                "_should_inject_json_prompt",
                return_value=False,
            ),
            patch.object(AIService, "_assert_model_blind_prompt", return_value=None),
            patch.object(
                AIService,
                "_run_openai_compatible_validation_tool_loop",
                new=self._provider_loop,
            ),
        ]
        started = [patcher.start() for patcher in self._patches]
        mock_config = started[0]
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 128,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

    def __exit__(self, *_exc: Any) -> None:
        for patcher in reversed(self._patches):
            patcher.stop()
        self._patches = []


@pytest.mark.asyncio
async def test_non_terminal_tool_events_reach_the_consumer_live() -> None:
    """Every turn event is delivered while it happens, never buffered."""

    service = AIService.__new__(AIService)
    events: List[Tuple[str, Dict[str, Any]]] = []
    delivered_during_loop: List[int] = []

    async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def provider_loop(*_args, **kwargs):
        emit = kwargs["tool_event_callback"]
        await emit("tool_loop_turn_start", {"turn": 1, "model": "gpt-4o"})
        delivered_during_loop.append(len(events))
        await emit("assistant_delta", {"turn": 1, "content": "LIVE_DRAFT_ONE"})
        delivered_during_loop.append(len(events))
        await emit("thinking_delta", {"turn": 1, "content": "REASONING", "is_thinking": True})
        delivered_during_loop.append(len(events))
        await emit("tool_call_ready", {"turn": 1, "tool_name": "validate_draft"})
        delivered_during_loop.append(len(events))
        return "ACCEPTED_TEXT", {"accepted": "validate_draft", "turns": 1, "trace": []}

    with _ToolLoopHarness(AsyncMock(side_effect=provider_loop)):
        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write something.",
            model="gpt-4o",
            validation_callback=lambda _text: None,
            tool_event_callback=tool_event_callback,
        )

    assert content == "ACCEPTED_TEXT"
    assert envelope.accepted is True
    # Each emission was already in the consumer's hands before the next one.
    assert delivered_during_loop == [1, 2, 3, 4]

    turn_events = [
        (event_type, payload)
        for event_type, payload in events
        if event_type != "tool_loop_complete"
    ]
    assert [event_type for event_type, _payload in turn_events] == [
        "tool_loop_turn_start",
        "assistant_delta",
        "thinking_delta",
        "tool_call_ready",
    ]
    assert all(
        payload["content_status"] == ContentStatus.DRAFT
        for _event_type, payload in turn_events
    )
    thinking = dict(turn_events[2][1])
    assert thinking["is_thinking"] is True
    assert thinking["content"] == "REASONING"
    # An accepted envelope announces no rejection.
    assert all(event_type != "tool_loop_rejected" for event_type, _p in events)
    assert events[-1][0] == "tool_loop_complete"


@pytest.mark.asyncio
async def test_rejected_envelope_keeps_drafts_and_emits_explicit_rejection() -> None:
    """A rejected attempt announces the verdict; the drafts already shown stay."""

    service = AIService.__new__(AIService)
    events: List[Tuple[str, Dict[str, Any]]] = []
    phase_logger = Mock()

    async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def provider_loop(*_args, **kwargs):
        await kwargs["tool_event_callback"](
            "assistant_delta",
            {"turn": 1, "content": "REJECTED_DRAFT_TEXT"},
        )
        return "REJECTED_DRAFT_TEXT", {"accepted": "", "turns": 2, "trace": []}

    with _ToolLoopHarness(AsyncMock(side_effect=provider_loop)):
        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write something.",
            model="gpt-4o",
            validation_callback=lambda _text: None,
            tool_event_callback=tool_event_callback,
            phase_logger=phase_logger,
        )

    assert content == "REJECTED_DRAFT_TEXT"
    assert envelope.accepted is False

    event_types = [event_type for event_type, _payload in events]
    # The draft is NOT retracted: it was labelled a draft when it was shown.
    assert event_types == [
        "assistant_delta",
        "tool_loop_rejected",
        "tool_loop_complete",
    ]
    draft_payload = events[0][1]
    assert draft_payload["content"] == "REJECTED_DRAFT_TEXT"
    assert draft_payload["content_status"] == ContentStatus.DRAFT

    rejection = events[1][1]
    assert rejection["content_status"] == ContentStatus.REJECTED
    assert rejection["turns"] == 2
    assert rejection["accepted_via"] is None
    assert rejection["model_id"] == "gpt-4o"
    assert rejection["content_chars"] == len("REJECTED_DRAFT_TEXT")

    # The rejected attempt is logged too; before C6 it was never logged at all.
    phase_logger.log_response.assert_called_once()
    logged = phase_logger.log_response.call_args.kwargs
    assert logged["response"] == "REJECTED_DRAFT_TEXT"
    assert logged["metadata"]["accepted"] is False
    assert logged["metadata"]["turns"] == 2


@pytest.mark.asyncio
async def test_accepted_envelope_logs_the_accepted_response_only() -> None:
    """The accepted branch keeps its historical log shape and emits no rejection."""

    service = AIService.__new__(AIService)
    events: List[Tuple[str, Dict[str, Any]]] = []
    phase_logger = Mock()

    async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def provider_loop(*_args, **kwargs):
        return "GOOD_TEXT", {"accepted": "validate_draft", "turns": 3, "trace": []}

    with _ToolLoopHarness(AsyncMock(side_effect=provider_loop)):
        _content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write something.",
            model="gpt-4o",
            validation_callback=lambda _text: None,
            tool_event_callback=tool_event_callback,
            phase_logger=phase_logger,
        )

    assert envelope.accepted is True
    assert [event_type for event_type, _payload in events] == ["tool_loop_complete"]
    phase_logger.log_response.assert_called_once()
    assert phase_logger.log_response.call_args.kwargs["metadata"]["accepted"] == (
        "validate_draft"
    )


@pytest.mark.asyncio
async def test_provider_terminal_event_is_labelled_rejected() -> None:
    """A provider-side stop is labelled a dead attempt, not a draft."""

    service = AIService.__new__(AIService)
    events: List[Tuple[str, Dict[str, Any]]] = []

    async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def provider_loop(*_args, **kwargs):
        emit = kwargs["tool_event_callback"]
        await emit("assistant_delta", {"turn": 1, "content": "PARTIAL"})
        await emit(
            "tool_loop_provider_terminal",
            {"turn": 1, "status": "length", "api_surface": "chat_completions"},
        )
        raise RuntimeError("provider stream ended unusable")

    with _ToolLoopHarness(AsyncMock(side_effect=provider_loop)):
        with pytest.raises(RuntimeError):
            await service.call_ai_with_validation_tools(
                prompt="Write something.",
                model="gpt-4o",
                validation_callback=lambda _text: None,
                tool_event_callback=tool_event_callback,
            )

    labels = {event_type: payload["content_status"] for event_type, payload in events}
    assert labels["assistant_delta"] == ContentStatus.DRAFT
    assert labels["tool_loop_provider_terminal"] == ContentStatus.REJECTED
    assert labels["tool_loop_error"] == ContentStatus.REJECTED


def _generation_session() -> Dict[str, Any]:
    return {
        "project_id": "project-tool-visibility",
        "project_epoch": 4,
        "request_name": "tool visibility",
        "current_iteration": 2,
    }


@pytest.mark.asyncio
async def test_generation_tool_callback_publishes_labelled_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generation consumer forwards content, thinking and control events."""

    session = _generation_session()
    published: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    callback = generation_processor._make_generation_tool_event_callback(
        "session-tool-visibility",
        session,
    )
    assert callback is not None

    await callback(
        "assistant_delta",
        {
            "model": "gpt-4o",
            "turn": 1,
            "content": "VISIBLE_DRAFT",
            "content_status": ContentStatus.DRAFT,
        },
    )
    await callback(
        "thinking_delta",
        {
            "model": "gpt-4o",
            "turn": 1,
            "content": "VISIBLE_REASONING",
            "is_thinking": True,
            "content_status": ContentStatus.DRAFT,
        },
    )
    await callback(
        "tool_call_delta",
        {
            "model": "gpt-4o",
            "turn": 1,
            "index": 0,
            "delta": '{"text":"STREAMED_ARGUMENT',
            "content_status": ContentStatus.DRAFT,
        },
    )
    await callback(
        "tool_loop_turn_start",
        {"model": "gpt-4o", "turn": 1, "content_status": ContentStatus.DRAFT},
    )
    await callback(
        "tool_loop_rejected",
        {
            "model": "gpt-4o",
            "turns": 1,
            "content_status": ContentStatus.REJECTED,
        },
    )

    assert [event.event for event in published] == [
        "chunk",
        "chunk",
        "chunk",
        "tool_loop_turn_start",
        "tool_loop_rejected",
    ]
    assert [event.content for event in published] == [
        "VISIBLE_DRAFT",
        "VISIBLE_REASONING",
        "STREAMED_ARGUMENT",
        "",
        "",
    ]
    assert published[0].subphase == ContentSubphase.DRAFT_STREAM
    assert published[0].is_thinking is False
    assert published[1].subphase == ContentSubphase.THINKING
    assert published[1].is_thinking is True
    assert published[4].content_status == ContentStatus.REJECTED
    for event in published:
        assert event.phase == "generation"
        assert event.session_id == "session-tool-visibility"
        assert event.project_id == "project-tool-visibility"
        assert event.project_epoch == 4
        assert event.iteration == 2
        assert event.attempt == 1
        assert event.layer == "generation.tool_loop"
    # The raw telemetry travels in ``data`` so the monitor keeps the details.
    assert published[0].data["turn"] == 1


@pytest.mark.asyncio
async def test_generation_tool_callback_allocates_one_attempt_id_per_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tool loop of a session is a new attempt for cursor resets."""

    session = _generation_session()
    published: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)

    first = generation_processor._make_generation_tool_event_callback("s-1", session)
    await first("assistant_delta", {"content": "A", "content_status": ContentStatus.DRAFT})
    second = generation_processor._make_generation_tool_event_callback("s-1", session)
    await second("assistant_delta", {"content": "B", "content_status": ContentStatus.DRAFT})

    assert [event.attempt for event in published] == [1, 2]
    # The live session surfaces are not fed by the tool-loop path.
    assert session.get("generation_content") is None
    assert session.get("partial_content") is None


@pytest.mark.asyncio
async def test_generation_tool_callback_without_project_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No project means no stream to publish to; the loop runs callback-free."""

    published: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    callback = generation_processor._make_generation_tool_event_callback("s-2", {})

    assert callback is None
    assert published == []


@pytest.mark.asyncio
async def test_generation_tool_drafts_reach_the_central_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draft deltas go through the C0 handler: sampled bus + debugger trace."""

    clock = _FakeClock()
    published: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []

    async def _publisher(**kwargs: Any) -> None:
        published.append(kwargs)

    async def _tracer(**kwargs: Any) -> None:
        traces.append(kwargs)

    coalescer = ContentStreamCoalescer(
        clock=clock,
        publisher=_publisher,
        trace_writer=_tracer,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        coalescer.publish,
    )

    session = _generation_session()
    callback = generation_processor._make_generation_tool_event_callback(
        "session-central",
        session,
    )
    assert callback is not None

    try:
        for fragment in ("Once ", "upon ", "a time"):
            await callback(
                "assistant_delta",
                {
                    "model": "gpt-4o",
                    "turn": 1,
                    "content": fragment,
                    "content_status": ContentStatus.DRAFT,
                },
            )
        # Inside the sampling window the bus sees nothing yet.
        assert published == []

        await coalescer.flush("session-central")
    finally:
        coalescer.discard_session("session-central")

    assert len(published) == 1
    assert published[0]["content"] == "Once upon a time"
    assert published[0]["content_status"] == ContentStatus.DRAFT
    assert published[0]["phase"] == "generation"
    assert published[0]["iteration"] == 2
    assert published[0]["attempt_id"] == 1
    assert published[0]["qa_model"] == "gpt-4o"

    assert len(traces) == 1
    assert traces[0]["content_delta"] == "Once upon a time"
    assert traces[0]["session_id"] == "session-central"
    assert traces[0]["phase"] == "generation"
    assert traces[0]["attempt"] == 1


_CANONICAL_LEAK_SENTINEL = "TOOL_LOOP_DRAFT_A_CANONICAL_SUBSCRIBER_MUST_NOT_SEE"

# (bus phase, factory, expected layer of the published event). Gran Sabio and
# the Arbiter have no per-layer identity in the tool-loop payload, so the loop
# is the layer; QA keeps the evaluated layer, which the monitor indexes by.
_TOOL_LOOP_CONSUMERS = [
    (
        "gran_sabio",
        generation_processor._make_gran_sabio_tool_event_callback,
        "gran_sabio.tool_loop",
    ),
    (
        "arbiter",
        generation_processor._make_arbiter_tool_event_callback,
        "arbiter.tool_loop",
    ),
    ("qa", generation_processor._make_qa_tool_event_callback, "coherence"),
]


def _tool_loop_payload(content: str, *, content_status: str) -> Dict[str, Any]:
    return {
        "model": "gpt-4o",
        "turn": 1,
        "content": content,
        "qa_layer": "coherence",
        "qa_model": "gpt-4o",
        "content_status": content_status,
    }


def _hidden_from_canonical(published: Dict[str, Any]) -> bool:
    """Apply the transversal canonical filter to one published bus event.

    The event is serialized exactly as the bus does it, so the assertion proves
    the label travels TOP LEVEL: buried inside ``data`` (the pre-fix shape) the
    filter reads ``None`` and the draft survives.
    """
    bus_kwargs = dict(published)
    bus_kwargs.pop("project_id", None)
    serialized = json.loads(_build_project_phase_event(**bus_kwargs))
    return is_hidden_by_canonical_visibility(
        content_status=serialized.get("content_status"),
        subphase=serialized.get("subphase"),
        is_thinking=serialized.get("is_thinking"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "factory", "expected_layer"),
    _TOOL_LOOP_CONSUMERS,
    ids=[phase for phase, _factory, _layer in _TOOL_LOOP_CONSUMERS],
)
async def test_canonical_visibility_hides_every_tool_loop_draft(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    factory: Any,
    expected_layer: str,
) -> None:
    """The four tool loops publish through the contract, so canonical filters.

    Gran Sabio, the Arbiter and QA used to publish their loop turns raw with the
    label buried inside ``data``: a ``visibility=canonical`` subscriber received
    their unvalidated drafts as plain text. The label now travels where the
    filter reads it, and the terminal verdict still reaches everybody.
    """
    clock = _FakeClock()
    published: List[Dict[str, Any]] = []

    async def _publisher(**kwargs: Any) -> None:
        published.append(kwargs)

    async def _tracer(**_kwargs: Any) -> None:
        return None

    coalescer = ContentStreamCoalescer(
        clock=clock,
        publisher=_publisher,
        trace_writer=_tracer,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        coalescer.publish,
    )

    session = _generation_session()
    session[live_attempt_key("gransabio_content")] = 3
    callback = factory("session-canonical", session)
    try:
        await callback(
            "assistant_delta",
            _tool_loop_payload(
                _CANONICAL_LEAK_SENTINEL,
                content_status=ContentStatus.DRAFT,
            ),
        )
        await coalescer.flush("session-canonical")
        await callback(
            "tool_loop_rejected",
            _tool_loop_payload("", content_status=ContentStatus.REJECTED),
        )
    finally:
        coalescer.discard_session("session-canonical")

    assert [event["phase"] for event in published] == [phase, phase]
    draft, verdict = published

    assert _CANONICAL_LEAK_SENTINEL in draft["content"]
    assert draft["content_status"] == ContentStatus.DRAFT
    assert draft["subphase"] == ContentSubphase.DRAFT_STREAM
    assert draft["qa_layer"] == expected_layer
    assert draft["attempt_id"] is not None
    assert draft["iteration"] == 2
    assert _hidden_from_canonical(draft) is True

    # A verdict is not a draft: canonical subscribers must keep receiving it.
    assert verdict["content_status"] == ContentStatus.REJECTED
    assert verdict["event"] == "tool_loop_rejected"
    assert _hidden_from_canonical(verdict) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [
        generation_processor._make_gran_sabio_tool_event_callback,
        generation_processor._make_arbiter_tool_event_callback,
        generation_processor._make_qa_tool_event_callback,
    ],
    ids=["gran_sabio", "arbiter", "qa"],
)
async def test_tool_loop_thinking_keeps_its_own_labels(
    monkeypatch: pytest.MonkeyPatch,
    factory: Any,
) -> None:
    """Provider reasoning keeps ``is_thinking`` and the thinking subphase.

    The three consumers used to drop it: ``_tool_loop_visible_chunk`` only
    decodes assistant/tool deltas, so a ``thinking_delta`` produced no visible
    text at all. They now read it from the payload like the generation consumer
    does, and it stays filterable as thinking (C8).
    """
    published: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    callback = factory("session-thinking", _generation_session())

    payload = _tool_loop_payload("REASONING", content_status=ContentStatus.DRAFT)
    payload["is_thinking"] = True
    await callback("thinking_delta", payload)

    assert len(published) == 1
    event = published[0]
    assert "REASONING" in event.content
    assert event.is_thinking is True
    assert event.subphase == ContentSubphase.THINKING
    assert is_hidden_by_canonical_visibility(
        content_status=event.content_status,
        subphase=event.subphase,
        is_thinking=event.is_thinking,
    ) is True


@pytest.mark.asyncio
async def test_qa_tool_loop_retry_opens_a_new_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QA truncation retry is a new attempt, never a longer first one."""
    published: List[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    callback = generation_processor._make_qa_tool_event_callback(
        "session-qa-retry",
        _generation_session(),
    )

    await callback(
        "assistant_delta",
        _tool_loop_payload("FIRST", content_status=ContentStatus.DRAFT),
    )
    await callback(
        generation_processor._QA_TOOL_LOOP_RETRY_EVENT,
        _tool_loop_payload("", content_status=ContentStatus.DRAFT),
    )
    await callback(
        "assistant_delta",
        _tool_loop_payload("SECOND", content_status=ContentStatus.DRAFT),
    )

    assert [event.attempt for event in published] == [1, 2, 2]


def test_event_data_cap_guards_data_and_never_the_content() -> None:
    """The ~8 KB guard degrades ``data``; the visible content is untouched."""

    visible = "V" * 40_000
    payload = {"delta": "D" * 20_000, "turn": 1}

    event = _build_project_phase_event(
        event="chunk",
        phase="generation",
        content=visible,
        content_status=ContentStatus.DRAFT,
        iteration=2,
        attempt_id=1,
        data=payload,
    )

    decoded = json.loads(event)
    assert decoded["content"] == visible
    assert decoded["content_status"] == ContentStatus.DRAFT
    assert decoded["data"]["truncated"] is True
    assert set(decoded["data"]["preview_keys"]) == set(payload.keys())


def test_small_event_data_is_forwarded_verbatim() -> None:
    """Payloads under the guard keep every field the monitor renders."""

    event = _build_project_phase_event(
        event="tool_call_ready",
        phase="generation",
        content=None,
        content_status=ContentStatus.DRAFT,
        data={"tool_name": "validate_draft", "turn": 2},
    )

    decoded = json.loads(event)
    assert decoded["data"] == {"tool_name": "validate_draft", "turn": 2}
    assert decoded.get("content") is None


def test_rejected_envelope_shape_is_reusable_by_consumers() -> None:
    """``ToolLoopEnvelope`` still reports the rejection the event announces."""

    envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.GENERATOR,
        turns=2,
        accepted=False,
        accepted_via="",
        output_schema_valid=False,
    )

    assert envelope.accepted is False
    assert envelope.turns == 2


def test_content_filter_stop_is_still_terminal_for_the_tool_loop() -> None:
    """Visibility never softens the content-filter verdict (keep-list §4)."""

    stop = GenerationStoppedUnexpectedly(
        "provider refused",
        provider="openai",
        model_id="gpt-4o",
        finish_reason="content_filter",
        finish_reason_category="content_filter",
        partial_content="PARTIAL",
    )

    assert is_content_filter_stop(stop) is True
    assert (
        _tool_loop_event_content_status(
            "tool_loop_error",
            {"error_code": "content_filter"},
        )
        == ContentStatus.FILTERED
    )
    assert (
        _tool_loop_event_content_status("assistant_delta", {"content": "x"})
        == ContentStatus.DRAFT
    )
