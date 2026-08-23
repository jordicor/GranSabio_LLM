"""Adversarial live-visibility coverage for Arbiter and preflight streams.

Contract under test (C5 / C5-ter): both surfaces publish every provider chunk
AS IT ARRIVES as a labelled draft, and neither one ever labels anything
``accepted`` - the Arbiter verdict and the preflight decision are structured
results, not stream commits. A filtered, cancelled or unparseable attempt is
therefore never retracted: it simply stops producing drafts, and the strict
validation of the verdict/decision is unchanged.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.generation_processor as generation_processor
import llm_routing
import preflight_validator
from ai_service import GenerationStoppedUnexpectedly, StreamChunk
from arbiter import Arbiter, ArbiterContext, LayerEditHistory
from content_events import ContentEvent, ContentStatus, ContentSubphase
from models import ContentRequest, QALayer


def _arbiter_context() -> ArbiterContext:
    return ArbiterContext(
        original_prompt="Resolve the edits.",
        content_type="analysis",
        system_prompt=None,
        layer_name="Integrity",
        layer_criteria="Return a complete decision.",
        layer_min_score=8.0,
        current_content="Candidate.",
        content_excerpt=None,
        proposed_edits=[],
        evaluator_scores={},
        layer_history=LayerEditHistory(layer_name="Integrity"),
        gran_sabio_model="test-escalation-model",
        qa_model_count=1,
        arbiter_tools_mode="never",
    )


def _runtime_parameters(*_args, **_kwargs):
    return 11_000, {"source": "test"}, 0.2, "high", 7_000


def _events(callback: AsyncMock) -> list[ContentEvent]:
    """Return every content event the stream callback received, in order."""
    return [awaited.args[0] for awaited in callback.await_args_list]


def _assert_labelled_drafts(
    callback: AsyncMock,
    *,
    phase: str,
    model: str,
    layer: str,
) -> None:
    """Assert every DRAFT emission belongs to the expected producer.

    A terminal verdict (``filtered``) may follow the drafts on the same
    callback; it is checked by the test that provokes it, not here.
    """
    events = [
        event
        for event in _events(callback)
        if event.content_status == ContentStatus.DRAFT
    ]
    assert events, "the provider chunks must be visible as labelled drafts"
    for event in events:
        assert event.phase == phase
        assert event.model == model
        assert event.layer == layer
        assert event.attempt == 1
        assert event.subphase == (
            ContentSubphase.THINKING
            if event.is_thinking
            else ContentSubphase.DRAFT_STREAM
        )


def _drafted_text(callback: AsyncMock) -> str:
    """Return the concatenated visible draft text shown to the consumer."""
    return "".join(
        event.content for event in _events(callback) if not event.is_thinking
    )


@pytest.mark.asyncio
async def test_arbiter_stream_shows_draft_before_content_filter() -> None:
    """A filtered Arbiter attempt keeps its draft visible and never commits."""
    private_draft = '{"reasoning":"PRIVATE_DRAFT"'

    async def filtered_stream(**_kwargs):
        yield StreamChunk(private_draft)
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "content_filter",
                "finish_reason_category": "content_filter",
                "finish_unusable": True,
            },
        )

    callback = AsyncMock()
    service = SimpleNamespace(generate_content_stream=filtered_stream)
    arbiter = Arbiter(
        ai_service=service,
        model="test-arbiter-model",
        stream_callback=callback,
    )
    arbiter._resolve_runtime_parameters = _runtime_parameters

    with pytest.raises(GenerationStoppedUnexpectedly):
        await arbiter._resolve_single_shot(
            context=_arbiter_context(),
            prompt="Resolve.",
            selected_model="test-arbiter-model",
            prompt_safety_parts=None,
        )

    _assert_labelled_drafts(
        callback,
        phase="arbiter",
        model="test-arbiter-model",
        layer="arbitration",
    )
    assert _drafted_text(callback) == private_draft
    # The attempt is closed by exactly one filtered verdict, and it resends
    # nothing: the draft above is never repeated.
    verdicts = [
        event
        for event in _events(callback)
        if event.content_status == ContentStatus.FILTERED
    ]
    assert len(verdicts) == 1
    assert verdicts[0].content == ""


@pytest.mark.asyncio
async def test_arbiter_stream_publishes_each_chunk_and_forwards_runtime_params() -> None:
    """Chunks reach the consumer one by one, thinking labelled apart."""
    response = (
        '{"reasoning":"accepted","decisions":[],"conflicts_resolved":[]}'
    )
    first, second = response[:25], response[25:]
    captured = {}

    async def accepted_stream(**kwargs):
        captured.update(kwargs)
        yield StreamChunk(first)
        yield StreamChunk("private reasoning", is_thinking=True)
        yield StreamChunk(second)
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
            },
        )

    callback = AsyncMock()
    service = SimpleNamespace(generate_content_stream=accepted_stream)
    arbiter = Arbiter(
        ai_service=service,
        model="test-arbiter-model",
        stream_callback=callback,
    )
    arbiter._resolve_runtime_parameters = _runtime_parameters

    parsed = await arbiter._resolve_single_shot(
        context=_arbiter_context(),
        prompt="Resolve.",
        selected_model="test-arbiter-model",
        prompt_safety_parts=None,
    )

    assert parsed["reasoning"] == "accepted"
    _assert_labelled_drafts(
        callback,
        phase="arbiter",
        model="test-arbiter-model",
        layer="arbitration",
    )
    # One emission per provider chunk, in order, thinking included but never
    # mixed into the visible verdict text.
    assert [event.content for event in _events(callback)] == [
        first,
        "private reasoning",
        second,
    ]
    assert [event.is_thinking for event in _events(callback)] == [
        False,
        True,
        False,
    ]
    assert _drafted_text(callback) == response
    assert captured["max_tokens"] == 11_000
    assert captured["temperature"] == 0.2
    assert captured["reasoning_effort"] == "high"
    assert captured["thinking_budget_tokens"] == 7_000


@pytest.mark.asyncio
async def test_failing_arbiter_verdict_sink_never_replaces_the_terminal_stop() -> None:
    """A broken consumer must not turn a content filter into another failure.

    The verdict is published from inside the ``except`` block that re-raises the
    provider stop: an exception from the sink would replace it and the caller
    would classify the failure with the wrong category.
    """
    delivered: list[ContentEvent] = []

    async def failing_sink(event: ContentEvent) -> None:
        if event.content_status == ContentStatus.FILTERED:
            raise RuntimeError("consumer is down")
        delivered.append(event)

    async def filtered_stream(**_kwargs):
        yield StreamChunk('{"reasoning":"PRIVATE_DRAFT"')
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "content_filter",
                "finish_reason_category": "content_filter",
                "finish_unusable": True,
            },
        )

    service = SimpleNamespace(generate_content_stream=filtered_stream)
    arbiter = Arbiter(
        ai_service=service,
        model="test-arbiter-model",
        stream_callback=failing_sink,
    )
    arbiter._resolve_runtime_parameters = _runtime_parameters

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await arbiter._resolve_single_shot(
            context=_arbiter_context(),
            prompt="Resolve.",
            selected_model="test-arbiter-model",
            prompt_safety_parts=None,
        )

    assert raised.value.finish_reason_category == "content_filter"
    assert [event.content_status for event in delivered] == [ContentStatus.DRAFT]


@pytest.mark.asyncio
async def test_consecutive_arbiter_resolutions_are_separate_attempts() -> None:
    """C0.2: one Arbiter serves a whole QA run, so each resolution is its own attempt.

    Sharing ``attempt=1`` across resolutions would let a consumer append the
    drafts of the second arbitration onto the first instead of resetting.
    """

    async def accepted_stream(**_kwargs):
        yield StreamChunk(
            '{"reasoning":"ok","decisions":[],"conflicts_resolved":[]}'
        )
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
            },
        )

    callback = AsyncMock()
    service = SimpleNamespace(generate_content_stream=accepted_stream)
    arbiter = Arbiter(
        ai_service=service,
        model="test-arbiter-model",
        stream_callback=callback,
    )
    arbiter._resolve_runtime_parameters = _runtime_parameters

    for _resolution in range(3):
        await arbiter._resolve_single_shot(
            context=_arbiter_context(),
            prompt="Resolve.",
            selected_model="test-arbiter-model",
            prompt_safety_parts=None,
        )

    assert [event.attempt for event in _events(callback)] == [1, 2, 3]


@pytest.mark.asyncio
async def test_arbiter_tool_loop_counts_one_attempt_per_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C0.2: the shared tool-event callback labels each loop as its own attempt.

    ``_make_arbiter_tool_event_callback`` is built once per QA run, so a fixed
    ``attempt=1`` would put every resolution of the run under the same bus key.
    The first turn of a loop is the boundary that opens the next attempt.
    """
    published: list[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    monkeypatch.setattr(generation_processor, "publish_content_event", _capture)
    callback = generation_processor._make_arbiter_tool_event_callback(
        "session-arbiter-attempts",
        {
            "project_id": "project-arbiter-attempts",
            "project_epoch": 1,
            "request_name": "arbiter attempts",
            "current_iteration": 2,
        },
    )

    for _resolution in range(2):
        await callback(
            generation_processor._TOOL_LOOP_TURN_START_EVENT,
            {"model": "test-arbiter-model", "turn": 1},
        )
        await callback(
            "assistant_delta",
            {
                "model": "test-arbiter-model",
                "turn": 1,
                "content": "reasoning",
                "content_status": ContentStatus.DRAFT,
            },
        )
        # A second turn of the SAME loop is not a new attempt.
        await callback(
            generation_processor._TOOL_LOOP_TURN_START_EVENT,
            {"model": "test-arbiter-model", "turn": 2},
        )

    assert [event.attempt for event in published] == [1, 1, 1, 2, 2, 2]


@pytest.mark.asyncio
async def test_arbiter_tool_loop_forwards_runtime_reasoning_params() -> None:
    envelope = SimpleNamespace(
        output_truncated=False,
        payload={
            "reasoning": "accepted",
            "decisions": [],
            "conflicts_resolved": [],
        },
    )
    call = AsyncMock(return_value=("accepted", envelope))
    service = SimpleNamespace(call_ai_with_validation_tools=call)
    arbiter = Arbiter(ai_service=service, model="test-arbiter-model")
    arbiter._resolve_runtime_parameters = _runtime_parameters

    parsed = await arbiter._resolve_with_tool_loop(
        context=_arbiter_context(),
        prompt="Resolve.",
        selected_model="test-arbiter-model",
        prompt_safety_parts=None,
    )

    assert parsed["reasoning"] == "accepted"
    kwargs = call.await_args.kwargs
    assert kwargs["max_tokens"] == 11_000
    assert kwargs["temperature"] == 0.2
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["thinking_budget_tokens"] == 7_000


@pytest.mark.asyncio
async def test_arbiter_escalation_uses_its_own_canonical_route() -> None:
    request = ContentRequest(
        prompt="Resolve a difficult conflict.",
        content_type="analysis",
        generator_model="gpt-5-mini",
        llm_routing={
            "calls": {
                "arbiter.resolve": {
                    "model": "gpt-5-mini",
                    "params": {
                        "max_tokens": 11_000,
                        "reasoning_effort": "high",
                    },
                },
                "gransabio.escalation": {
                    "model": "claude-opus-4-6",
                    "params": {
                        "max_tokens": 13_000,
                        "thinking_budget_tokens": 4_000,
                    },
                },
            }
        },
    )
    llm_routing.attach_request_llm_routing(
        request,
        set(request.model_fields_set),
    )
    context = _arbiter_context()
    context.original_request = request
    envelope = SimpleNamespace(
        output_truncated=False,
        payload={
            "reasoning": "accepted",
            "decisions": [],
            "conflicts_resolved": [],
        },
    )
    call = AsyncMock(return_value=("accepted", envelope))
    arbiter = Arbiter(
        ai_service=SimpleNamespace(call_ai_with_validation_tools=call),
        model="gpt-5-mini",
    )

    selected_model = arbiter._resolve_escalation_model(context)
    parsed = await arbiter._resolve_with_tool_loop(
        context=context,
        prompt="Resolve.",
        selected_model=selected_model,
        prompt_safety_parts=None,
        call_id="gransabio.escalation",
    )

    assert parsed["reasoning"] == "accepted"
    assert selected_model == "claude-opus-4-6"
    kwargs = call.await_args.kwargs
    assert kwargs["model"] == "claude-opus-4-6"
    assert kwargs["max_tokens"] == 13_000
    assert kwargs["thinking_budget_tokens"] == 4_000


def _preflight_request() -> ContentRequest:
    request = ContentRequest(
        prompt="Produce a complete response.",
        content_type="analysis",
        generator_model="gpt-4o",
        qa_layers=[
            QALayer(
                name="Integrity",
                description="Check completeness.",
                criteria="Reject incomplete output.",
                min_score=8.0,
                order=1,
            )
        ],
        qa_models=["gpt-4o"],
    )
    request._preflight_model = "test-preflight-model"
    return request


def _stub_preflight_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    route = SimpleNamespace(
        model="test-preflight-model",
        provider_options={
            "max_price": {"prompt": 0, "completion": 0},
            "allow_fallbacks": False,
        },
        params={
            "temperature": 0.15,
            "max_tokens": 512,
            "reasoning_effort": "medium",
            "thinking_budget_tokens": 3_000,
        },
    )
    monkeypatch.setattr(
        preflight_validator,
        "resolve_call",
        lambda *_args, **_kwargs: route,
    )
    monkeypatch.setattr(
        preflight_validator,
        "resolve_temperature",
        lambda *_args, **_kwargs: 0.15,
    )
    monkeypatch.setattr(
        type(preflight_validator.config),
        "resolve_output_max_tokens",
        lambda _self, *_args, **_kwargs: {"max_tokens": 512},
    )
    monkeypatch.setattr(
        preflight_validator.config,
        "PREFLIGHT_SYSTEM_PROMPT",
        "Validate.",
    )


@pytest.mark.asyncio
async def test_preflight_nonstream_forwards_runtime_reasoning_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_preflight_routing(monkeypatch)
    generate = AsyncMock(
        return_value='{"decision":"proceed","summary":"accepted"}'
    )
    service = SimpleNamespace(generate_content=generate)

    result = await preflight_validator.run_preflight_validation(
        service,
        _preflight_request(),
    )

    assert result.decision == "proceed"
    kwargs = generate.await_args.kwargs
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.15
    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["thinking_budget_tokens"] == 3_000
    assert kwargs["provider_options"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "allow_fallbacks": False,
    }


@pytest.mark.asyncio
async def test_preflight_invalid_json_never_reaches_phase_or_console_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_preflight_routing(monkeypatch)
    private_invalid_json = "PREFLIGHT_INVALID_PRIVATE_BODY"
    phase_logger = MagicMock()

    with caplog.at_level("WARNING"):
        result = await preflight_validator.run_preflight_validation(
            SimpleNamespace(
                generate_content=AsyncMock(return_value=private_invalid_json)
            ),
            _preflight_request(),
            phase_logger=phase_logger,
        )

    assert result.decision == "reject"
    assert result.issues[0].code == "preflight_invalid_response"
    phase_logger.log_response.assert_not_called()
    assert private_invalid_json not in caplog.text


@pytest.mark.asyncio
async def test_preflight_stream_shows_draft_before_content_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered preflight keeps its draft visible; the verdict stays terminal."""
    _stub_preflight_routing(monkeypatch)
    private_draft = '{"decision":"PRIVATE_DRAFT"'

    async def filtered_stream(**_kwargs):
        yield StreamChunk(private_draft)
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "content_filter",
                "finish_reason_category": "content_filter",
                "finish_unusable": True,
            },
        )

    callback = AsyncMock()
    service = SimpleNamespace(generate_content_stream=filtered_stream)

    result = await preflight_validator.run_preflight_validation(
        service,
        _preflight_request(),
        stream_callback=callback,
    )

    assert result.decision == "reject"
    assert result.issues[0].code == "content_filter"
    _assert_labelled_drafts(
        callback,
        phase="preflight",
        model="test-preflight-model",
        layer="preflight_validation",
    )
    assert _drafted_text(callback) == private_draft


@pytest.mark.asyncio
async def test_preflight_stream_publishes_each_chunk_as_a_labelled_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preflight streams live; the decision stays a validated structured result."""
    _stub_preflight_routing(monkeypatch)
    response = '{"decision":"proceed","summary":"accepted"}'
    captured = {}

    async def accepted_stream(**kwargs):
        captured.update(kwargs)
        yield StreamChunk(response[:20])
        yield StreamChunk("private reasoning", is_thinking=True)
        yield StreamChunk(response[20:])
        yield StreamChunk(
            "",
            metadata={
                "provider": "test",
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
            },
        )

    callback = AsyncMock()
    service = SimpleNamespace(generate_content_stream=accepted_stream)

    result = await preflight_validator.run_preflight_validation(
        service,
        _preflight_request(),
        stream_callback=callback,
    )

    assert result.decision == "proceed"
    _assert_labelled_drafts(
        callback,
        phase="preflight",
        model="test-preflight-model",
        layer="preflight_validation",
    )
    assert [event.content for event in _events(callback)] == [
        response[:20],
        "private reasoning",
        response[20:],
    ]
    # Thinking never contaminates the text the validator parses.
    assert _drafted_text(callback) == response
    assert captured["max_tokens"] == 512
    assert captured["temperature"] == 0.15
    assert captured["provider_options"] == {
        "max_price": {"prompt": 0, "completion": 0},
        "allow_fallbacks": False,
    }
    assert captured["reasoning_effort"] == "medium"
    assert captured["thinking_budget_tokens"] == 3_000
