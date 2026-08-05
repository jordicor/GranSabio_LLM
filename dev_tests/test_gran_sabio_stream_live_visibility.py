"""Offline visibility regressions for Gran Sabio streaming paths (C5-bis).

Contract under test: **labelled live visibility, strict canonicity**.

- Every provider chunk of the three Gran Sabio streaming loops reaches the
  consumer AS IT ARRIVES, as a :class:`ContentEvent` labelled
  ``content_status="draft"`` and carrying its model, layer, attempt and the
  provider-reported ``is_thinking`` flag.
- Exactly ONE emission per operation is labelled ``accepted``, and only after
  the operation validated its own contract (JSON normalization for the
  regeneration, envelope schema plus nested caller contract for the reviews).
- Nothing already shown is retracted when the attempt dies: a content filter,
  a cancellation, a timeout or an invalid JSON simply never produces an
  ``accepted`` event, and the tool-event stream announces the rejection.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

import gran_sabio as gran_sabio_module
from ai_service import (
    GenerationStoppedUnexpectedly,
    StreamChunk,
    is_content_filter_stop,
)
from config import config
from content_events import ContentEvent, ContentStatus, ContentSubphase
from gran_sabio import (
    REVIEW_ATTEMPT_REJECTED_EVENT,
    GranSabioEngine,
    GranSabioProcessCancelled,
)
from models import ContentRequest

MODEL = "claude-opus-4-8"
PRIVATE_PARTIAL = "PRIVATE_PARTIAL_MUST_NOT_COMMIT"
REGENERATED = "Complete regenerated content."
REVIEW_JSON = '{"ok":true}'
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
PRIVATE_NESTED_JSON = "NESTED_PRIVATE_INVALID_JSON_MUST_NOT_COMMIT"


def _request(*, json_output: bool = False) -> ContentRequest:
    return ContentRequest(
        prompt="Produce a bounded generic response for offline testing.",
        content_type="json" if json_output else "other",
        json_output=json_output,
        model=MODEL,
        gran_sabio_model=MODEL,
        max_tokens=128,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )


def _terminal(
    *,
    category: str | None = None,
    truncated: bool = False,
    max_tokens: int = 128,
) -> StreamChunk:
    finish_reason = "max_tokens" if truncated else "end_turn"
    metadata: dict[str, Any] = {
        "provider": "anthropic",
        "finish_reason": finish_reason,
        "provider_stop_reason": finish_reason,
        "finish_unusable": bool(category),
        "output_truncated": truncated,
        "max_tokens": max_tokens,
    }
    if category:
        metadata["finish_reason"] = "refusal"
        metadata["provider_stop_reason"] = "refusal"
        metadata["finish_reason_category"] = category
    return StreamChunk("", metadata=metadata)


def _retry_plan() -> dict[str, Any]:
    return {
        "requested_max_tokens": 128,
        "effective_max_tokens": 128,
        "retry_max_tokens": 256,
        "model_safe_max_tokens": 4096,
        "retry_ordinal": 1,
        "growth_policy": "bounded_test_growth",
    }


class _ScriptedStreamingService:
    def __init__(self, scripts: list[list[Any]]) -> None:
        self.scripts = scripts
        self.calls = 0
        self.budgets: list[int] = []

    def generate_content_stream(self, **kwargs: Any):
        script = self.scripts[self.calls]
        self.calls += 1
        self.budgets.append(int(kwargs["max_tokens"]))

        async def _chunks():
            for event in script:
                if isinstance(event, BaseException):
                    raise event
                yield event

        return _chunks()


class _CancelAfter:
    def __init__(self, call_number: int) -> None:
        self.call_number = call_number
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.call_number


def _events(callback: AsyncMock) -> list[ContentEvent]:
    """Return every content event the stream callback received, in order."""
    return [awaited.args[0] for awaited in callback.await_args_list]


def _events_with_status(callback: AsyncMock, status: str) -> list[ContentEvent]:
    """Return the received events carrying one ``content_status``."""
    return [event for event in _events(callback) if event.content_status == status]


def _drafted_text(callback: AsyncMock) -> str:
    """Return the concatenated visible draft text shown to the consumer."""
    return "".join(
        event.content
        for event in _events_with_status(callback, ContentStatus.DRAFT)
        if not event.is_thinking
    )


def _assert_labelled_drafts(callback: AsyncMock, *, layer: str) -> None:
    """Assert every draft carries the labels the contract requires."""
    drafts = _events_with_status(callback, ContentStatus.DRAFT)
    assert drafts, "the provider chunks must be visible as labelled drafts"
    for event in drafts:
        assert event.phase == "gran_sabio"
        assert event.model == MODEL
        assert event.layer == layer
        assert event.attempt is not None and event.attempt >= 1
        assert event.subphase == (
            ContentSubphase.THINKING
            if event.is_thinking
            else ContentSubphase.DRAFT_STREAM
        )


def _scripts(scenario: str, *, review: bool) -> list[list[Any]]:
    complete = REVIEW_JSON if review else REGENERATED
    if scenario == "content_filter":
        return [
            [
                StreamChunk(PRIVATE_PARTIAL),
                _terminal(category="content_filter"),
            ]
        ]
    if scenario == "truncation_retry":
        return [
            [
                StreamChunk(PRIVATE_PARTIAL),
                _terminal(truncated=True),
            ],
            [
                StreamChunk(complete),
                _terminal(max_tokens=256),
            ],
        ]
    if scenario == "timeout":
        return [
            [
                StreamChunk(PRIVATE_PARTIAL),
                asyncio.TimeoutError("provider stream timeout"),
            ]
        ]
    if scenario == "success":
        if review:
            return [
                [
                    StreamChunk('{"ok":'),
                    StreamChunk("private reasoning", is_thinking=True),
                    StreamChunk("true}"),
                    _terminal(),
                ]
            ]
        return [
            [
                StreamChunk("Complete "),
                StreamChunk("private reasoning", is_thinking=True),
                StreamChunk("regenerated content."),
                _terminal(),
            ]
        ]
    if scenario == "cancel":
        return [[StreamChunk(PRIVATE_PARTIAL), _terminal()]]
    raise AssertionError(f"unknown scenario: {scenario}")


def _configure_regeneration_engine(
    service: _ScriptedStreamingService,
    monkeypatch: pytest.MonkeyPatch,
) -> GranSabioEngine:
    engine = GranSabioEngine(ai_service=service)
    engine._get_default_content_generation_model = Mock(
        return_value=(MODEL, "high", None, 0.2)
    )
    engine._resolve_gran_sabio_max_tokens = Mock(return_value=128)
    engine._resolve_request_timeout = Mock(return_value=10.0)
    monkeypatch.setattr(
        gran_sabio_module,
        "_should_use_gransabio_tools",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(config, "MAX_RETRIES", 1)
    monkeypatch.setattr(config, "RETRY_STREAMING_AFTER_PARTIAL", False)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)
    return engine


async def _invoke_review(
    engine: GranSabioEngine,
    request: ContentRequest,
    *,
    callback: AsyncMock,
    cancel_callback: Any = None,
) -> tuple[dict[str, Any], str]:
    return await engine._invoke_gran_sabio_llm(
        request=request,
        prompt="Return strict JSON.",
        model=MODEL,
        reasoning_effort="high",
        thinking_tokens=None,
        temperature=0.2,
        system_prompt="Return strict JSON.",
        response_schema=REVIEW_SCHEMA,
        max_tool_rounds=1,
        initial_measurement_text=None,
        usage_callback=None,
        phase_logger=None,
        alias_registry=None,
        prompt_safety_parts=None,
        stream_callback=callback,
        stream_stage_label="review_stream",
        cancel_callback=cancel_callback,
        abort_stage="review_stream",
        max_tokens=128,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["content_filter", "truncation_retry", "cancel", "timeout", "success"],
)
async def test_regeneration_stream_shows_labelled_drafts_and_one_commit(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regeneration streams labelled drafts; only a valid body is accepted."""
    service = _ScriptedStreamingService(_scripts(scenario, review=False))
    engine = _configure_regeneration_engine(service, monkeypatch)
    callback = AsyncMock()
    cancel_callback = _CancelAfter(3) if scenario == "cancel" else None
    monkeypatch.setattr(
        gran_sabio_module,
        "_resolve_output_truncation_retry_plan",
        (
            (lambda *_args, **_kwargs: _retry_plan())
            if scenario == "truncation_retry"
            else (lambda *_args, **_kwargs: None)
        ),
    )

    if scenario == "content_filter":
        with pytest.raises(GenerationStoppedUnexpectedly) as raised:
            await engine.regenerate_content(
                session_id=f"regeneration-{scenario}",
                original_request=_request(),
                stream_callback=callback,
            )
        assert is_content_filter_stop(raised.value)
        # Terminal verdict unchanged: the filtered attempt carries no partial
        # content into the failure payload. What the provider had already
        # delivered stays visible, labelled as the draft it was.
        assert raised.value.partial_content == ""
        _assert_labelled_drafts(callback, layer="content_regeneration")
        assert _drafted_text(callback) == PRIVATE_PARTIAL
        assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
        return

    if scenario == "cancel":
        with pytest.raises(GranSabioProcessCancelled):
            await engine.regenerate_content(
                session_id=f"regeneration-{scenario}",
                original_request=_request(),
                stream_callback=callback,
                cancel_callback=cancel_callback,
            )
        _assert_labelled_drafts(callback, layer="content_regeneration")
        assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
        return

    result = await engine.regenerate_content(
        session_id=f"regeneration-{scenario}",
        original_request=_request(),
        stream_callback=callback,
    )
    if scenario == "timeout":
        assert result.approved is False
        assert result.final_content == ""
        assert PRIVATE_PARTIAL not in (result.error or "")
        _assert_labelled_drafts(callback, layer="content_regeneration")
        assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
        return

    assert result.approved is True
    assert result.final_content == REGENERATED
    _assert_labelled_drafts(callback, layer="content_regeneration")

    accepted = _events_with_status(callback, ContentStatus.ACCEPTED)
    assert len(accepted) == 1
    commit = accepted[0]
    assert commit.content == REGENERATED
    assert commit.model == MODEL
    assert commit.layer == "content_regeneration"
    assert commit.subphase == ContentSubphase.ACCEPTED_STREAM
    assert commit.is_thinking is False
    # The commit is the LAST thing the consumer sees, after every draft.
    assert _events(callback)[-1] is commit
    assert PRIVATE_PARTIAL not in commit.content
    # C0.2/C0.7: the commit REPLACES the drafts of the attempt that produced the
    # body, so it must carry THAT attempt id - an unlabelled commit leaves the
    # consumer unable to tell which draft stream it supersedes.
    draft_attempts = [
        event.attempt
        for event in _events_with_status(callback, ContentStatus.DRAFT)
    ]
    assert commit.attempt == draft_attempts[-1]

    if scenario == "truncation_retry":
        assert service.budgets == [128, 256]
        # The retry restarts the body: its drafts must belong to a NEW attempt
        # so a consumer resets instead of gluing both tries together.
        attempts = [
            event.attempt
            for event in _events_with_status(callback, ContentStatus.DRAFT)
        ]
        assert attempts == sorted(attempts)
        assert set(attempts) == {1, 2}
        assert PRIVATE_PARTIAL in _drafted_text(callback)
    else:
        thinking = [
            event
            for event in _events_with_status(callback, ContentStatus.DRAFT)
            if event.is_thinking
        ]
        assert [event.content for event in thinking] == ["private reasoning"]
        assert _drafted_text(callback) == REGENERATED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["content_filter", "truncation_retry", "cancel", "timeout", "success"],
)
async def test_review_stream_shows_drafts_and_defers_commit_to_public_review(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The review provider loop only ever emits drafts.

    ``_invoke_gran_sabio_llm`` returns the parsed envelope without committing
    it: the canonical ``accepted`` emission belongs to the public review
    method, which owns the nested caller output contract.
    """
    service = _ScriptedStreamingService(_scripts(scenario, review=True))
    engine = GranSabioEngine(ai_service=service)
    engine._resolve_request_timeout = Mock(return_value=10.0)
    callback = AsyncMock()
    cancel_callback = _CancelAfter(2) if scenario == "cancel" else None
    monkeypatch.setattr(
        gran_sabio_module,
        "_resolve_output_truncation_retry_plan",
        (
            (lambda *_args, **_kwargs: _retry_plan())
            if scenario == "truncation_retry"
            else (lambda *_args, **_kwargs: None)
        ),
    )

    if scenario == "content_filter":
        with pytest.raises(GenerationStoppedUnexpectedly) as raised:
            await _invoke_review(
                engine,
                _request(json_output=True),
                callback=callback,
            )
        assert is_content_filter_stop(raised.value)
        assert raised.value.partial_content == ""
        _assert_labelled_drafts(callback, layer="review_stream")
        assert _drafted_text(callback) == PRIVATE_PARTIAL
        assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
        return

    expected_exception: type[BaseException] | None = None
    if scenario == "cancel":
        expected_exception = GranSabioProcessCancelled
    elif scenario == "timeout":
        expected_exception = asyncio.TimeoutError
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            await _invoke_review(
                engine,
                _request(json_output=True),
                callback=callback,
                cancel_callback=cancel_callback,
            )
        _assert_labelled_drafts(callback, layer="review_stream")
        assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
        return

    payload, raw = await _invoke_review(
        engine,
        _request(json_output=True),
        callback=callback,
    )
    assert payload == {"ok": True}
    assert raw == REVIEW_JSON
    _assert_labelled_drafts(callback, layer="review_stream")
    # Canonicity: the provider loop never commits, whatever it streamed.
    assert _events_with_status(callback, ContentStatus.ACCEPTED) == []
    if scenario == "truncation_retry":
        assert service.budgets == [128, 256]
        assert {
            event.attempt
            for event in _events_with_status(callback, ContentStatus.DRAFT)
        } == {1, 2}
    else:
        assert _drafted_text(callback) == REVIEW_JSON
        assert [
            event.content
            for event in _events_with_status(callback, ContentStatus.DRAFT)
            if event.is_thinking
        ] == ["private reasoning"]


@pytest.mark.asyncio
async def test_review_stream_invalid_json_is_visible_but_never_accepted() -> None:
    """An unparseable review is watched live and simply never commits."""
    invalid = '{"ok":"not-a-boolean"}'
    service = _ScriptedStreamingService([[StreamChunk(invalid), _terminal()]])
    engine = GranSabioEngine(ai_service=service)
    engine._resolve_request_timeout = Mock(return_value=10.0)
    callback = AsyncMock()

    with pytest.raises(ValueError, match="not valid JSON"):
        await _invoke_review(
            engine,
            _request(json_output=True),
            callback=callback,
        )

    _assert_labelled_drafts(callback, layer="review_stream")
    assert _drafted_text(callback) == invalid
    assert _events_with_status(callback, ContentStatus.ACCEPTED) == []


@pytest.mark.asyncio
async def test_regeneration_stream_invalid_json_is_visible_but_never_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate that fails JSON normalization stays a draft forever."""
    invalid = '{"ok":"not-a-boolean"}'
    service = _ScriptedStreamingService([[StreamChunk(invalid), _terminal()]])
    engine = _configure_regeneration_engine(service, monkeypatch)
    callback = AsyncMock()
    request = _request(json_output=True).model_copy(
        update={"json_schema": REVIEW_SCHEMA}
    )

    result = await engine.regenerate_content(
        session_id="regeneration-invalid-json",
        original_request=request,
        stream_callback=callback,
    )

    assert result.approved is False
    assert result.final_content == ""
    assert "failed validation" in (result.error or "")
    _assert_labelled_drafts(callback, layer="content_regeneration")
    assert _drafted_text(callback) == invalid
    assert _events_with_status(callback, ContentStatus.ACCEPTED) == []


def _public_review_engine(
    payload: dict[str, Any],
    raw_response: str,
    *,
    prebound_tool_event_callback: AsyncMock | None = None,
) -> GranSabioEngine:
    engine = GranSabioEngine(
        ai_service=Mock(),
        tool_event_callback=prebound_tool_event_callback,
    )
    engine._get_default_critical_analysis_model = Mock(
        return_value=(MODEL, "high", None, 0.2)
    )
    engine._resolve_gran_sabio_max_tokens = Mock(return_value=128)

    async def _invoke(**kwargs: Any):
        review_tool_events = kwargs.get("review_tool_events")
        event_callback = (
            review_tool_events.capture_callback
            if review_tool_events is not None
            else None
        )
        if event_callback is not None:
            await event_callback(
                "tool_call",
                {"args_preview": payload["final_content"]},
            )
            await event_callback(
                "tool_loop_complete",
                {"accepted_via": "assistant_final"},
            )
        return payload, raw_response

    engine._invoke_gran_sabio_llm = AsyncMock(side_effect=_invoke)
    return engine


def _review_payload(
    review_kind: str,
    final_content: str,
    *,
    modifications_made: bool = True,
) -> dict[str, Any]:
    if review_kind == "iterations":
        return {
            "decision": (
                "APPROVE_WITH_MODIFICATIONS"
                if modifications_made
                else "APPROVE"
            ),
            "reason": "A bounded correction is required.",
            "score": 8.0,
            "modifications_made": modifications_made,
            "final_content": final_content,
        }
    return {
        "decision": "APPROVED",
        "reason": "A bounded correction is required.",
        "score": 8.0,
        "modifications_made": modifications_made,
        "final_content": final_content,
    }


async def _run_public_review(
    engine: GranSabioEngine,
    *,
    review_kind: str,
    request: ContentRequest,
    callback: AsyncMock,
    phase_logger: Mock,
    tool_event_callback: AsyncMock | None = None,
):
    if review_kind == "iterations":
        return await engine.review_iterations(
            session_id="nested-json-review",
            iterations=[
                {
                    "iteration": 1,
                    "content": '{"ok":false}',
                    "consensus": {"average_score": 7.0},
                    "qa_results": {},
                    "qa_layers_config": [],
                }
            ],
            original_request=request,
            stream_callback=callback,
            phase_logger=phase_logger,
            tool_event_callback=tool_event_callback,
        )
    return await engine.review_minority_deal_breakers(
        session_id="nested-json-review",
        content='{"ok":false}',
        minority_deal_breakers={
            "details": [],
            "total_evaluations": 0,
            "qa_configuration": {},
        },
        original_request=request,
        stream_callback=callback,
        phase_logger=phase_logger,
        tool_event_callback=tool_event_callback,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("review_kind", ["iterations", "minority"])
@pytest.mark.parametrize("event_binding", ["explicit", "prebound"])
async def test_public_review_invalid_nested_json_never_commits(
    review_kind: str,
    event_binding: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A nested contract violation blocks the commit and is announced.

    The tool-loop telemetry already flowed live as drafts, so the consumer is
    told explicitly that the attempt was rejected instead of being left with
    events that quietly never resolve.
    """
    payload = _review_payload(review_kind, PRIVATE_NESTED_JSON)
    raw_response = json.dumps(payload)
    tool_event_callback = AsyncMock()
    engine = _public_review_engine(
        payload,
        raw_response,
        prebound_tool_event_callback=(
            tool_event_callback if event_binding == "prebound" else None
        ),
    )
    callback = AsyncMock()
    phase_logger = Mock()
    phase_logger.extra_verbose = False
    request = _request(json_output=True).model_copy(
        update={"json_schema": REVIEW_SCHEMA}
    )

    result = await _run_public_review(
        engine,
        review_kind=review_kind,
        request=request,
        callback=callback,
        phase_logger=phase_logger,
        tool_event_callback=(
            tool_event_callback if event_binding == "explicit" else None
        ),
    )

    assert result.approved is False
    assert result.final_content == '{"ok":false}'
    assert PRIVATE_NESTED_JSON not in (result.error or "")
    # No canonical commit: nothing was accepted, so nothing was streamed as
    # accepted either.
    callback.assert_not_awaited()
    # The tool-loop drafts were live, and the rejection closes them out.
    emitted = [awaited.args[0] for awaited in tool_event_callback.await_args_list]
    assert emitted == [
        "tool_call",
        "tool_loop_complete",
        REVIEW_ATTEMPT_REJECTED_EVENT,
    ]
    for awaited in tool_event_callback.await_args_list[:-1]:
        assert awaited.args[1]["content_status"] == ContentStatus.DRAFT
    rejection = tool_event_callback.await_args_list[-1].args[1]
    assert rejection["content_status"] == ContentStatus.REJECTED
    assert rejection["reason"]
    assert PRIVATE_NESTED_JSON not in repr(rejection)
    # Canonicity is untouched: the invalid envelope never reaches the logs.
    phase_logger.log_response.assert_not_called()
    phase_logger.log_decision.assert_not_called()
    assert PRIVATE_NESTED_JSON not in caplog.text
    assert PRIVATE_NESTED_JSON not in repr(phase_logger.mock_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("review_kind", "expected_stage"),
    [
        ("iterations", "iterations_review"),
        ("minority", "deal_breakers_review"),
    ],
)
@pytest.mark.parametrize("event_binding", ["explicit", "prebound"])
async def test_public_review_valid_nested_json_commits_canonical_envelope_once(
    review_kind: str,
    expected_stage: str,
    event_binding: str,
) -> None:
    nested_json = '```json\n { "ok" : true }\n```'
    payload = _review_payload(review_kind, nested_json)
    tool_event_callback = AsyncMock()
    engine = _public_review_engine(
        payload,
        json.dumps(payload),
        prebound_tool_event_callback=(
            tool_event_callback if event_binding == "prebound" else None
        ),
    )
    callback = AsyncMock()
    phase_logger = Mock()
    phase_logger.extra_verbose = False
    request = _request(json_output=True).model_copy(
        update={"json_schema": REVIEW_SCHEMA}
    )

    result = await _run_public_review(
        engine,
        review_kind=review_kind,
        request=request,
        callback=callback,
        phase_logger=phase_logger,
        tool_event_callback=(
            tool_event_callback if event_binding == "explicit" else None
        ),
    )

    assert result.approved is True
    assert result.modifications_made is True
    assert result.final_content == '{"ok":true}'
    callback.assert_awaited_once()
    commit = callback.await_args.args[0]
    assert commit.content_status == ContentStatus.ACCEPTED
    assert commit.subphase == ContentSubphase.ACCEPTED_STREAM
    assert commit.model == MODEL
    assert commit.layer == expected_stage
    committed_response = commit.content
    # Canonicity: the committed envelope is the normalized one, never the
    # fenced draft the model wrote.
    assert json.loads(committed_response)["final_content"] == '{"ok":true}'
    assert nested_json not in committed_response
    # The tool-loop drafts flowed live and verbatim; a committed attempt emits
    # no rejection.
    assert [
        awaited.args[0]
        for awaited in tool_event_callback.await_args_list
    ] == ["tool_call", "tool_loop_complete"]
    tool_call_payload = tool_event_callback.await_args_list[0].args[1]
    assert tool_call_payload["args_preview"] == nested_json
    assert tool_call_payload["content_status"] == ContentStatus.DRAFT
    phase_logger.log_response.assert_called_once_with(
        model=MODEL,
        response=committed_response,
    )
    phase_logger.log_decision.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("review_kind", ["iterations", "minority"])
@pytest.mark.parametrize("event_binding", ["explicit", "prebound"])
async def test_public_review_discards_irrelevant_nested_final_from_events(
    review_kind: str,
    event_binding: str,
) -> None:
    irrelevant_final = "IRRELEVANT_PRIVATE_FINAL_CONTENT_SENTINEL"
    payload = _review_payload(
        review_kind,
        irrelevant_final,
        modifications_made=False,
    )
    tool_event_callback = AsyncMock()
    engine = _public_review_engine(
        payload,
        json.dumps(payload),
        prebound_tool_event_callback=(
            tool_event_callback if event_binding == "prebound" else None
        ),
    )
    callback = AsyncMock()
    phase_logger = Mock()
    phase_logger.extra_verbose = False
    request = _request(json_output=True).model_copy(
        update={"json_schema": REVIEW_SCHEMA}
    )

    result = await _run_public_review(
        engine,
        review_kind=review_kind,
        request=request,
        callback=callback,
        phase_logger=phase_logger,
        tool_event_callback=(
            tool_event_callback if event_binding == "explicit" else None
        ),
    )

    assert result.approved is True
    assert result.modifications_made is False
    assert result.final_content == '{"ok":false}'
    callback.assert_awaited_once()
    commit = callback.await_args.args[0]
    assert commit.content_status == ContentStatus.ACCEPTED
    committed_response = commit.content
    # Canonicity: an irrelevant ``final_content`` is dropped from the envelope
    # that governs the result, even though the draft telemetry showed it.
    assert json.loads(committed_response)["final_content"] is None
    assert irrelevant_final not in committed_response
    assert [call.args[0] for call in tool_event_callback.await_args_list] == [
        "tool_call",
        "tool_loop_complete",
    ]
    assert (
        tool_event_callback.await_args_list[0].args[1]["args_preview"]
        == irrelevant_final
    )
    phase_logger.log_response.assert_called_once_with(
        model=MODEL,
        response=committed_response,
    )


def _configure_scripted_review_engine(
    service: _ScriptedStreamingService,
    monkeypatch: pytest.MonkeyPatch,
) -> GranSabioEngine:
    """Build a public review engine streaming from a scripted provider."""
    engine = GranSabioEngine(ai_service=service)
    engine._get_default_critical_analysis_model = Mock(
        return_value=(MODEL, "high", None, 0.2)
    )
    engine._resolve_gran_sabio_max_tokens = Mock(return_value=128)
    engine._resolve_request_timeout = Mock(return_value=10.0)
    monkeypatch.setattr(
        gran_sabio_module,
        "_resolve_output_truncation_retry_plan",
        lambda *_args, **_kwargs: _retry_plan(),
    )
    return engine


async def _run_scripted_iterations_review(
    engine: GranSabioEngine,
    *,
    timeline: list[tuple[str, Any]],
):
    """Run the public iterations review recording BOTH live surfaces in order.

    The retry telemetry travels on the tool-event surface and the terminal
    verdict on the content one, so only a single ordered timeline can show which
    verdict a consumer reads LAST.
    """

    async def _stream_callback(event: ContentEvent) -> None:
        timeline.append(("content", event))

    async def _tool_event_callback(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        timeline.append((event_type, payload))

    return await engine.review_iterations(
        session_id="retry-then-verdict",
        iterations=[
            {
                "iteration": 1,
                "content": '{"ok":false}',
                "consensus": {"average_score": 7.0},
                "qa_results": {},
                "qa_layers_config": [],
            }
        ],
        original_request=_request(json_output=True),
        stream_callback=_stream_callback,
        tool_event_callback=_tool_event_callback,
    )


@pytest.mark.asyncio
async def test_truncation_retry_then_content_filter_never_closes_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filtered retry stays filtered even when the surfaces differ.

    The first attempt emits ``output_truncation_retry_scheduled`` through the
    review's tool-event recorder; the retry is then stopped by content policy and
    its verdict is published on the CONTENT surface. The recorder only learns
    about the filter from that publication, so without the cross-surface mark its
    ``finally`` announced a contradicting ``review_attempt_rejected`` as the last
    word on the attempt.
    """
    service = _ScriptedStreamingService(
        [
            [StreamChunk(PRIVATE_PARTIAL), _terminal(truncated=True)],
            [
                StreamChunk(PRIVATE_PARTIAL),
                _terminal(category="content_filter", max_tokens=256),
            ],
        ]
    )
    engine = _configure_scripted_review_engine(service, monkeypatch)
    timeline: list[tuple[str, Any]] = []

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await _run_scripted_iterations_review(engine, timeline=timeline)

    assert is_content_filter_stop(raised.value)
    assert service.budgets == [128, 256]
    names = [name for name, _payload in timeline]
    # The retry telemetry DID reach the recorder: that emission is what makes a
    # closing rejection possible at all.
    assert names.count("output_truncation_retry_scheduled") == 1
    assert REVIEW_ATTEMPT_REJECTED_EVENT not in names
    last_name, last_event = timeline[-1]
    assert last_name == "content"
    assert last_event.content_status == ContentStatus.FILTERED
    assert last_event.attempt == 2
    assert last_event.content == ""
    assert last_event.data["error_code"] == "content_filter"


@pytest.mark.asyncio
async def test_truncation_retry_then_invalid_envelope_still_closes_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine non-filter failure keeps its explicit rejection.

    Same retry sequence, no content policy: the attempt died on its own output
    contract, so the drafts the consumer saw must still be superseded by
    ``review_attempt_rejected``.
    """
    service = _ScriptedStreamingService(
        [
            [StreamChunk(PRIVATE_PARTIAL), _terminal(truncated=True)],
            [StreamChunk('{"ok":'), _terminal(max_tokens=256)],
        ]
    )
    engine = _configure_scripted_review_engine(service, monkeypatch)
    timeline: list[tuple[str, Any]] = []

    result = await _run_scripted_iterations_review(engine, timeline=timeline)

    assert result.approved is False
    assert result.error
    names = [name for name, _payload in timeline]
    assert names.count("output_truncation_retry_scheduled") == 1
    assert names[-1] == REVIEW_ATTEMPT_REJECTED_EVENT
    assert timeline[-1][1]["content_status"] == ContentStatus.REJECTED
    assert ContentStatus.FILTERED not in [
        event.content_status for name, event in timeline if name == "content"
    ]
