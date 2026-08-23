"""Offline regressions for content-filter integrity at auxiliary surfaces."""

from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest

import ai_service as ai_service_module
import auto_qa_planner
import core.generation.prestart_flow as prestart_flow
import core.generation_processor as generation_processor
import core.streaming_routes as streaming_routes
from ai_service import (
    AccentGuardError,
    AIRequestError,
    AIService,
    GenerationOutputTruncated,
    GenerationStoppedUnexpectedly,
    StreamChunk,
    is_content_filter_stop,
)
from auto_qa_planner import (
    AUTO_QA_CONTENT_FILTER_MESSAGE,
    AutoQAPlanningError,
    run_auto_qa_planning,
)
from config import config
from content_events import ContentEvent, ContentStatus
from core import app_state
from core.generation.prestart_flow import PrestartFlowDeps, run_prestart_flow
from models import ContentRequest, GenerationStatus
from provider_errors import ProviderErrorKind
from tool_loop_models import LoopScope, ToolLoopEnvelope

RAW_PROVIDER_DETAIL = "RAW_PROVIDER_DETAIL_MUST_NOT_LEAK"


def _content_filter_stop() -> GenerationStoppedUnexpectedly:
    return GenerationStoppedUnexpectedly(
        f"{RAW_PROVIDER_DETAIL}: request refused",
        provider="test-provider",
        model_id="test-model",
        finish_reason="content_filter",
        finish_reason_category="content_filter",
        partial_content="FILTERED_PARTIAL_CONTENT",
        usage={
            "finish_reason": "content_filter",
            "finish_reason_category": "content_filter",
            "finish_unusable": True,
            "output_truncated": False,
        },
    )


def _main_generation_request(**overrides) -> SimpleNamespace:
    values = dict(
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
    values.update(overrides)
    return SimpleNamespace(**values)


def _bounded_retry_plan() -> dict:
    return {
        "requested_max_tokens": 64,
        "effective_max_tokens": 64,
        "retry_max_tokens": 128,
        "model_safe_max_tokens": 1024,
        "retry_ordinal": 1,
        "growth_policy": "bounded-test-policy",
    }


class _FeedbackManager:
    async def initialize_session(self, *_args, **_kwargs) -> dict:
        return {"initial_rules": []}

    async def complete_session(self, *_args, **_kwargs) -> None:
        return None


def _patch_main_generation_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cancel_check,
    debug_record: AsyncMock,
    publish: AsyncMock,
    use_tools: bool = False,
) -> None:
    monkeypatch.setattr(
        generation_processor,
        "_should_use_generation_tools",
        lambda _request: use_tools,
    )
    monkeypatch.setattr(
        generation_processor,
        "has_active_generation_validators",
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
        cancel_check,
    )
    monkeypatch.setattr(
        generation_processor,
        "add_verbose_log",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_record_event",
        debug_record,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        publish,
    )


@pytest.mark.asyncio
async def test_gpt5_empty_content_filter_reaches_typed_finish_guard() -> None:
    service = AIService.__new__(AIService)
    service.openai_client = object()
    service._is_openai_responses_api_model = Mock(return_value=False)
    service._should_inject_json_prompt = Mock(return_value=False)
    service._call_chat_completions_create_with_reasoning_retry = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None),
                    finish_reason="content_filter",
                )
            ],
            usage=None,
        )
    )

    content, usage = await service._generate_openai(
        prompt="Write a short answer.",
        model_id="gpt-5-mini",
        temperature=0.2,
        max_tokens=64,
        system_prompt="System.",
    )

    assert content == ""
    assert usage["finish_reason_category"] == "content_filter"
    assert usage["finish_unusable"] is True
    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        service._raise_if_generation_stopped_unusably(
            content,
            usage,
            provider="openai",
            model_id="gpt-5-mini",
            max_tokens=64,
        )
    assert is_content_filter_stop(raised.value)
    service._call_chat_completions_create_with_reasoning_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_zero_output_stream_emits_filter_metadata_not_runtime_error() -> None:
    service = AIService.__new__(AIService)
    service.openai_client = object()
    service._should_inject_json_prompt = Mock(return_value=False)

    class FilteredStream:
        def __init__(self) -> None:
            self._chunks = iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="content_filter",
                                delta=SimpleNamespace(content=None),
                            )
                        ],
                        usage=None,
                    )
                ]
            )

        def __aiter__(self) -> "FilteredStream":
            return self

        async def __anext__(self) -> object:
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    service._call_chat_completions_create_with_reasoning_retry = AsyncMock(
        return_value=FilteredStream()
    )

    chunks = [
        chunk
        async for chunk in service._stream_openai(
            prompt="Write a short answer.",
            model_id="gpt-4o",
            temperature=0.2,
            max_tokens=64,
            system_prompt="System.",
            extra_verbose=False,
        )
    ]

    assert len(chunks) == 1
    assert isinstance(chunks[0], StreamChunk)
    assert chunks[0].text == ""
    assert chunks[0].metadata["finish_reason_category"] == "content_filter"
    assert chunks[0].metadata["finish_unusable"] is True
    service._call_chat_completions_create_with_reasoning_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_unexplained_zero_output_keeps_runtime_error_path() -> None:
    service = AIService.__new__(AIService)
    service.openai_client = object()
    service._should_inject_json_prompt = Mock(return_value=False)

    class EmptyStopStream:
        def __init__(self) -> None:
            self._chunks = iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                delta=SimpleNamespace(content=None),
                            )
                        ],
                        usage=None,
                    )
                ]
            )

        def __aiter__(self) -> "EmptyStopStream":
            return self

        async def __anext__(self) -> object:
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    service._call_chat_completions_create_with_reasoning_retry = AsyncMock(
        return_value=EmptyStopStream()
    )

    chunks: list[object] = []
    with pytest.raises(RuntimeError, match="no content received"):
        async for chunk in service._stream_openai(
            prompt="Write a short answer.",
            model_id="gpt-4o",
            temperature=0.2,
            max_tokens=64,
            system_prompt="System.",
            extra_verbose=False,
        ):
            chunks.append(chunk)

    assert chunks == []
    service._call_chat_completions_create_with_reasoning_retry.assert_awaited_once()


@pytest.mark.asyncio
async def test_accent_audit_rethrows_content_filter_before_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = _content_filter_stop()
    service = AIService.__new__(AIService)
    service.generate_content = AsyncMock(side_effect=stop)
    service._raise_if_cancelled = AsyncMock()
    service._normalize_tool_loop_provider = Mock(return_value="openai")
    service._audit_model_supports_structured_outputs = Mock(return_value=False)
    monkeypatch.setattr(
        ai_service_module,
        "resolve_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="accent-model",
            params={},
        ),
    )
    monkeypatch.setattr(
        ai_service_module,
        "resolve_temperature",
        lambda _route: 0.1,
    )
    monkeypatch.setattr(
        ai_service_module,
        "config",
        SimpleNamespace(
            get_model_info=lambda _model: {
                "provider": "openai",
                "model_id": "accent-model",
            }
        ),
    )

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await service.audit_accent(
            "Candidate.",
            language="en",
            criteria_block="Avoid generic formulas.",
            min_score=8.0,
            timeout_seconds=5.0,
            max_tokens=128,
            on_error="fail_open",
        )

    assert raised.value is stop
    service.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_accent_audit_keeps_ordinary_failure_wrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = AIService.__new__(AIService)
    service.generate_content = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    service._raise_if_cancelled = AsyncMock()
    service._normalize_tool_loop_provider = Mock(return_value="openai")
    service._audit_model_supports_structured_outputs = Mock(return_value=False)
    monkeypatch.setattr(
        ai_service_module,
        "resolve_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="accent-model",
            params={},
        ),
    )
    monkeypatch.setattr(
        ai_service_module,
        "resolve_temperature",
        lambda _route: 0.1,
    )
    monkeypatch.setattr(
        ai_service_module,
        "config",
        SimpleNamespace(
            get_model_info=lambda _model: {
                "provider": "openai",
                "model_id": "accent-model",
            }
        ),
    )

    with pytest.raises(AccentGuardError) as raised:
        await service.audit_accent(
            "Candidate.",
            language="en",
            criteria_block="Avoid generic formulas.",
            min_score=8.0,
            timeout_seconds=5.0,
            max_tokens=128,
            on_error="fail_open",
        )

    assert raised.value.reason == "error"
    assert "provider unavailable" in str(raised.value)
    service.generate_content.assert_awaited_once()


def _patch_auto_qa_planning_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        auto_qa_planner,
        "build_auto_qa_planning_payload",
        lambda *_args, **_kwargs: {"request": "bounded"},
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "_build_prompt_safety_parts",
        lambda _payload: [],
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "assert_prompt_is_model_blind",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "resolve_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="planner-model",
            params={},
        ),
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "resolve_temperature",
        lambda _route: 0.1,
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "resolve_request_timeout",
        lambda *_args, **_kwargs: 5.0,
    )
    monkeypatch.setattr(
        auto_qa_planner,
        "config",
        SimpleNamespace(
            REQUEST_TIMEOUT=5.0,
            REQUEST_TIMEOUTS={},
            resolve_output_max_tokens=lambda *_args, **_kwargs: {
                "max_tokens": 128
            },
        ),
    )


@pytest.mark.asyncio
async def test_auto_qa_preserves_sanitized_content_filter_contract_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auto_qa_planning_boundary(monkeypatch)
    generate = AsyncMock(side_effect=_content_filter_stop())
    service = SimpleNamespace(generate_content=generate)

    with pytest.raises(AutoQAPlanningError) as raised:
        await run_auto_qa_planning(
            service,
            SimpleNamespace(),
            model_alias_registry=MagicMock(),
        )

    feedback = raised.value.to_feedback()
    assert feedback == {
        "code": "content_filter",
        "message": AUTO_QA_CONTENT_FILTER_MESSAGE,
        "details": {
            "finish_reason_category": "content_filter",
            "retryable": False,
        },
    }
    assert RAW_PROVIDER_DETAIL not in str(feedback)
    generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_qa_keeps_ordinary_failure_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_auto_qa_planning_boundary(monkeypatch)
    generate = AsyncMock(side_effect=RuntimeError("ordinary planner outage"))

    with pytest.raises(AutoQAPlanningError) as raised:
        await run_auto_qa_planning(
            SimpleNamespace(generate_content=generate),
            SimpleNamespace(),
            model_alias_registry=MagicMock(),
        )

    assert raised.value.code == "auto_qa_planner_failed"
    assert raised.value.details == {"error": "ordinary planner outage"}
    generate.assert_awaited_once()


class _PhaseLogger:
    def phase(self, _phase: object) -> object:
        return nullcontext()


class _UsageTracker:
    def span(self, **_kwargs: object) -> object:
        return nullcontext()

    def create_callback(self, **_kwargs: object) -> None:
        return None


@pytest.mark.asyncio
async def test_preflight_retry_replaces_live_snapshot_and_keeps_attempt_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live preflight view contains only attempt 2 after protocol repair."""
    published: list[ContentEvent] = []

    async def _capture(event: ContentEvent) -> None:
        published.append(event)

    async def _preflight(*_args, **kwargs):
        callback = kwargs["stream_callback"]
        await callback(ContentEvent(phase="preflight", content="malformed", attempt=1))
        await callback(ContentEvent(phase="preflight", content="valid", attempt=2))
        return SimpleNamespace(decision="proceed")

    monkeypatch.setattr(prestart_flow, "publish_content_event", _capture)
    monkeypatch.setattr(prestart_flow, "flush_content_events", AsyncMock())
    registry = SimpleNamespace(
        register_session=AsyncMock(),
        register_current_task=AsyncMock(),
        is_cancelled=AsyncMock(return_value=False),
        is_soft_cancelled=AsyncMock(return_value=False),
    )
    deps = PrestartFlowDeps(
        ai_service=SimpleNamespace(),
        cancellation_registry=registry,
        register_session=AsyncMock(),
        mutate_session=AsyncMock(),
        pop_session=AsyncMock(),
        is_terminal_session=Mock(return_value=False),
        apply_session_cancelled_state=Mock(),
        publish_project_session_end=AsyncMock(),
        debug_record_event=AsyncMock(),
        create_phase_logger=Mock(return_value=_PhaseLogger()),
        run_auto_qa_planning=AsyncMock(),
        run_preflight_validation=_preflight,
        apply_auto_qa_plan=Mock(),
        validate_auto_qa_effective_contract=Mock(),
    )
    request = SimpleNamespace(
        max_iterations=1,
        verbose=False,
        extra_verbose=False,
        request_name="preflight retry labels",
        show_query_stats=0,
    )

    result = await run_prestart_flow(
        request=request,
        session_id="session-preflight-retry",
        project_id="project-preflight-retry",
        project_epoch=3,
        auto_qa_requested=False,
        request_fields_set=set(),
        usage_tracker=_UsageTracker(),
        preflight_context=[],
        preflight_image_info=None,
        model_alias_registry=MagicMock(),
        deps=deps,
    )

    assert result.response is None
    assert result.temp_session["preflight_content"] == "valid"
    assert result.temp_session[prestart_flow._PREFLIGHT_ATTEMPT_KEY] == 2
    assert [event.attempt for event in published] == [1, 2]
    assert [event.content for event in published] == ["malformed", "valid"]


@pytest.mark.asyncio
async def test_prestart_content_filter_rejection_removes_temporary_session() -> None:
    register_session = AsyncMock()
    pop_session = AsyncMock()
    debug_record_event = AsyncMock()
    publish_project_session_end = AsyncMock()
    registry = SimpleNamespace(
        register_session=AsyncMock(),
        register_current_task=AsyncMock(),
        is_cancelled=AsyncMock(return_value=False),
        is_soft_cancelled=AsyncMock(return_value=False),
    )
    raw_error = AutoQAPlanningError(
        "content_filter",
        RAW_PROVIDER_DETAIL,
        details={"provider_payload": RAW_PROVIDER_DETAIL},
    )
    deps = PrestartFlowDeps(
        ai_service=SimpleNamespace(),
        cancellation_registry=registry,
        register_session=register_session,
        mutate_session=AsyncMock(),
        pop_session=pop_session,
        is_terminal_session=Mock(return_value=False),
        apply_session_cancelled_state=Mock(),
        publish_project_session_end=publish_project_session_end,
        debug_record_event=debug_record_event,
        create_phase_logger=Mock(return_value=_PhaseLogger()),
        run_auto_qa_planning=AsyncMock(side_effect=raw_error),
        run_preflight_validation=AsyncMock(),
        apply_auto_qa_plan=Mock(),
        validate_auto_qa_effective_contract=Mock(),
    )
    request = SimpleNamespace(
        max_iterations=1,
        verbose=False,
        extra_verbose=False,
        request_name=None,
        show_query_stats=0,
        auto_qa=SimpleNamespace(
            rigor="light",
            manual_layer_policy="replace",
        ),
    )

    result = await run_prestart_flow(
        request=request,
        session_id="session-filtered-auto-qa",
        project_id="project-filtered-auto-qa",
        project_epoch=0,
        auto_qa_requested=True,
        request_fields_set=set(),
        usage_tracker=_UsageTracker(),
        preflight_context=[],
        preflight_image_info=None,
        model_alias_registry=MagicMock(),
        deps=deps,
    )

    assert result.response is not None
    assert result.response.status == "auto_qa_rejected"
    assert result.response.session_id is None
    assert result.response.auto_qa_feedback == {
        "code": "content_filter",
        "message": AUTO_QA_CONTENT_FILTER_MESSAGE,
        "details": {
            "finish_reason_category": "content_filter",
            "retryable": False,
        },
    }
    register_session.assert_awaited_once()
    pop_session.assert_awaited_once_with("session-filtered-auto-qa")
    publish_project_session_end.assert_awaited_once()
    assert RAW_PROVIDER_DETAIL not in str(debug_record_event.await_args_list)


@pytest.mark.asyncio
async def test_auto_qa_summary_is_published_labelled_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Auto-QA planning summary is a validated commit, and says so.

    The planner emits it once, after the QA contract validated, and it used to
    reach the bus raw: no ``content_status``, no attempt, nothing a transversal
    filter could read. It now travels through the central handler labelled
    ``accepted``, and no raw publication is left behind.
    """
    summary = "Auto-QA planning started.\nAuto-QA planned 2 QA layer(s).\n"
    published: list = []

    async def _capture(event) -> None:
        published.append(event)

    monkeypatch.setattr(prestart_flow, "publish_content_event", _capture)
    monkeypatch.setattr(prestart_flow, "flush_content_events", AsyncMock())

    async def _plan(*_args, **kwargs):
        await kwargs["stream_callback"](summary)
        return SimpleNamespace(
            qa_layers=[object(), object()],
            generated_layer_names=["clarity", "coherence"],
            warnings=[],
            public_dict=lambda: {"qa_layers": 2},
        )

    registry = SimpleNamespace(
        register_session=AsyncMock(),
        register_current_task=AsyncMock(),
        is_cancelled=AsyncMock(return_value=False),
        is_soft_cancelled=AsyncMock(return_value=False),
    )
    deps = PrestartFlowDeps(
        ai_service=SimpleNamespace(),
        cancellation_registry=registry,
        register_session=AsyncMock(),
        mutate_session=AsyncMock(),
        pop_session=AsyncMock(),
        is_terminal_session=Mock(return_value=False),
        apply_session_cancelled_state=Mock(),
        publish_project_session_end=AsyncMock(),
        debug_record_event=AsyncMock(),
        create_phase_logger=Mock(return_value=_PhaseLogger()),
        run_auto_qa_planning=_plan,
        run_preflight_validation=AsyncMock(
            return_value=SimpleNamespace(decision="proceed")
        ),
        apply_auto_qa_plan=Mock(),
        validate_auto_qa_effective_contract=Mock(),
    )
    request = SimpleNamespace(
        max_iterations=1,
        verbose=False,
        extra_verbose=False,
        request_name="auto qa labels",
        show_query_stats=0,
        auto_qa=SimpleNamespace(rigor="light", manual_layer_policy="replace"),
    )

    result = await run_prestart_flow(
        request=request,
        session_id="session-auto-qa-labels",
        project_id="project-auto-qa-labels",
        project_epoch=7,
        auto_qa_requested=True,
        request_fields_set=set(),
        usage_tracker=_UsageTracker(),
        preflight_context=[],
        preflight_image_info=None,
        model_alias_registry=MagicMock(),
        deps=deps,
    )

    assert result.response is None
    assert len(published) == 1
    event = published[0]
    assert event.phase == "auto_qa"
    assert event.content == summary
    assert event.content_status == ContentStatus.ACCEPTED
    assert event.session_id == "session-auto-qa-labels"
    assert event.project_id == "project-auto-qa-labels"
    assert event.project_epoch == 7
    assert event.request_name == "auto qa labels"
    assert event.attempt == prestart_flow._AUTO_QA_ATTEMPT_ID
    assert event.is_thinking is False
    # The readable session surface and its attempt id keep being fed.
    assert result.temp_session["auto_qa_content"] == summary
    assert result.temp_session[prestart_flow._AUTO_QA_ATTEMPT_KEY] == (
        prestart_flow._AUTO_QA_ATTEMPT_ID
    )


def _direct_stream_session() -> dict[str, object]:
    return {
        "request": SimpleNamespace(
            prompt="Write a short answer.",
            temperature=0.2,
            max_tokens=64,
            content_type="other",
        ),
        "resolved_context": [],
        "status": GenerationStatus.GENERATING,
        "project_id": "direct-stream-project",
        "project_epoch": 0,
        "cancelled": False,
        "hard_cancelled": False,
        "usage_tracker": None,
    }


async def _run_direct_stream(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[object],
) -> bytes:
    session = _direct_stream_session()

    async def mutate_session(_session_id: str, mutator: object) -> object:
        return mutator(session)

    class FakeAIService:
        async def generate_content_stream(self, **_kwargs: object):
            for chunk in chunks:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

    registry = SimpleNamespace(
        is_hard_cancelled=AsyncMock(return_value=False),
        is_project_cancelled_or_stopping=AsyncMock(return_value=False),
        is_cancelled=AsyncMock(return_value=False),
        register_current_task=AsyncMock(),
        unregister_task=AsyncMock(),
    )
    monkeypatch.setattr(streaming_routes, "mutate_session", mutate_session)
    monkeypatch.setattr(streaming_routes, "cancellation_registry", registry)
    monkeypatch.setattr(streaming_routes, "ai_service", FakeAIService())
    monkeypatch.setattr(streaming_routes, "_ensure_services", lambda: None)
    monkeypatch.setattr(
        streaming_routes,
        "resolve_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="test-model",
            params={},
        ),
    )
    monkeypatch.setattr(
        type(streaming_routes.config),
        "resolve_output_max_tokens",
        lambda _self, *_args, **_kwargs: {"max_tokens": 64},
    )
    monkeypatch.setattr(
        streaming_routes,
        "bind_console_context",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        streaming_routes,
        "reset_console_context",
        lambda _tokens: None,
    )

    response = await streaming_routes.stream_content_direct_v2(
        "direct-stream-session"
    )
    return b"".join([chunk async for chunk in response.body_iterator])


@pytest.mark.asyncio
async def test_direct_stream_marks_filter_after_delivered_content_from_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1: chunks are live, so the terminal marker follows what was delivered."""
    output = await _run_direct_stream(
        monkeypatch,
        [
            "FILTERED_PARTIAL_CONTENT",
            StreamChunk(
                "",
                metadata={
                    "finish_reason": "content_filter",
                    "finish_reason_category": "content_filter",
                    "finish_unusable": True,
                },
            ),
        ],
    )

    assert output == (
        b"FILTERED_PARTIAL_CONTENT"
        + streaming_routes._DIRECT_STREAM_CONTENT_FILTER_TERMINAL
    )


@pytest.mark.asyncio
async def test_direct_stream_marks_filter_after_delivered_content_when_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1: a raised filter stop also terminates an already live transport."""
    output = await _run_direct_stream(
        monkeypatch,
        ["FILTERED_PARTIAL_CONTENT", _content_filter_stop()],
    )

    assert output == (
        b"FILTERED_PARTIAL_CONTENT"
        + streaming_routes._DIRECT_STREAM_CONTENT_FILTER_TERMINAL
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_category", ["stop", "pause_turn"])
async def test_direct_stream_preserves_normal_and_pause_turn_content(
    monkeypatch: pytest.MonkeyPatch,
    finish_category: str,
) -> None:
    output = await _run_direct_stream(
        monkeypatch,
        [
            "alpha ",
            StreamChunk("beta"),
            StreamChunk(
                "",
                metadata={
                    "finish_reason": finish_category,
                    "finish_reason_category": finish_category,
                    "finish_unusable": finish_category == "pause_turn",
                },
            ),
        ],
    )

    assert output == b"alpha beta"


@pytest.mark.asyncio
async def test_main_generation_labels_filtered_draft_and_keeps_it_out_of_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C16/F6: the filtered draft stays visible labelled; canonical stays clean.

    The provider had already delivered those tokens to the engine, so the live
    surfaces keep them (C16) and the attempt is closed with an explicit
    ``filtered`` verdict instead of a bare error event. Canonicity is untouched:
    the terminal exception carries no partial text, the debugger records no
    ``generator_output`` and no accepted commit is published.
    """
    session = {
        "project_id": "project-filtered-main",
        "project_epoch": 1,
        "request_name": "filtered main stream",
    }
    publish = AsyncMock()
    debug_record = AsyncMock()
    live_events: list = []

    async def _capture_content_event(event) -> None:
        live_events.append(event)

    async def _filtered_stream():
        yield "FILTERED_PARTIAL_CONTENT"
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

    ai_service = SimpleNamespace(
        generate_content_stream=Mock(return_value=_filtered_stream()),
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
    monkeypatch.setattr(
        generation_processor,
        "add_verbose_log",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_record_event",
        debug_record,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        publish,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        _capture_content_event,
    )
    monkeypatch.setattr(
        generation_processor,
        "flush_content_events",
        AsyncMock(),
    )

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a short article.",
            request=_main_generation_request(),
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-filtered-main",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    assert raised.value.finish_reason_category == "content_filter"
    assert raised.value.partial_content == ""
    assert len(publish.await_args_list) == 1
    terminal_event = publish.await_args_list[0]
    assert terminal_event.kwargs["event"] == "error"
    assert terminal_event.kwargs["content"] is None
    assert (
        generation_processor._UNUSABLE_FINISH_PUBLIC_MESSAGE
        in terminal_event.kwargs["reason"]
    )

    # F6: the attempt is closed with the filtered verdict, after the draft.
    assert [event.content_status for event in live_events] == [
        ContentStatus.DRAFT,
        ContentStatus.FILTERED,
    ]
    draft_event, filtered_event = live_events
    assert draft_event.content == "FILTERED_PARTIAL_CONTENT"
    # The verdict never re-sends the body it invalidates.
    assert filtered_event.content == ""
    assert filtered_event.attempt == draft_event.attempt
    assert filtered_event.iteration == draft_event.iteration
    assert "content policy" in filtered_event.reason
    assert filtered_event.data["error_code"] == "content_filter"
    assert filtered_event.data["retryable"] is False

    # C16: the live surfaces keep the draft the provider already delivered.
    assert session["generation_content"] == "FILTERED_PARTIAL_CONTENT"
    assert session["partial_content"] == "FILTERED_PARTIAL_CONTENT"
    # Canonicity: nothing accepted, nothing in the debugger result payloads.
    assert session.get("last_generated_content", "") == ""
    assert "FILTERED_PARTIAL_CONTENT" not in repr(debug_record.await_args_list)
    assert not any(
        call.args[1] == "generator_output"
        for call in debug_record.await_args_list
    )
    assert not any(
        call.kwargs.get("content_status") == ContentStatus.ACCEPTED
        for call in publish.await_args_list
    )


@pytest.mark.asyncio
async def test_failing_filtered_verdict_never_replaces_the_terminal_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry cannot corrupt the terminal path.

    ``_publish_content_filter_terminal`` is awaited from inside the ``except``
    block that re-raises the content-filter stop. If publishing raised, the new
    exception would REPLACE that stop and the failure would be classified with
    the wrong category, the wrong ``retryable`` and without the state purge.
    """
    session = {
        "project_id": "project-filtered-publish-fails",
        "project_epoch": 1,
        "request_name": "filtered publish fails",
    }
    live_events: list = []

    async def _capture_content_event(event) -> None:
        if event.content_status == ContentStatus.FILTERED:
            raise RuntimeError("bus is down")
        live_events.append(event)

    async def _filtered_stream():
        yield "FILTERED_PARTIAL_CONTENT"
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

    ai_service = SimpleNamespace(
        generate_content_stream=Mock(return_value=_filtered_stream()),
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
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        _capture_content_event,
    )
    monkeypatch.setattr(generation_processor, "flush_content_events", AsyncMock())

    with pytest.raises(GenerationStoppedUnexpectedly) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a short article.",
            request=_main_generation_request(),
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-filtered-publish-fails",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    # The provider stop survives intact, category included.
    assert raised.value.finish_reason_category == "content_filter"
    assert raised.value.partial_content == ""
    assert is_content_filter_stop(raised.value)
    # The dropped verdict is the only loss; the draft still travelled and the
    # C16 live surfaces are still the ones a content filter leaves behind.
    assert [event.content_status for event in live_events] == [ContentStatus.DRAFT]
    assert session["generation_content"] == "FILTERED_PARTIAL_CONTENT"


@pytest.mark.asyncio
async def test_main_generation_streams_drafts_and_commits_once_after_terminal_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2/C3: the draft is live on the session fields, the commit is single.

    The live fields follow the attempt in progress (sampled, so they hold a
    prefix of it), while the canonical commit event is published exactly once,
    after the terminal finish metadata validated the candidate.
    """
    session = {
        "project_id": "project-normal-main",
        "project_epoch": 1,
        "request_name": "normal main stream",
    }
    publish = AsyncMock()

    async def _normal_stream():
        yield "alpha "
        # First chunk always flushes: the draft is visible immediately.
        assert session.get("generation_content", "") == "alpha "
        assert session.get("partial_content", "") == "alpha "
        publish.assert_not_awaited()
        yield StreamChunk("beta")
        assert "alpha beta".startswith(session.get("generation_content", ""))
        assert "alpha beta".startswith(session.get("partial_content", ""))
        publish.assert_not_awaited()
        yield StreamChunk(
            "",
            metadata={
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
                "output_truncated": False,
            },
        )

    ai_service = SimpleNamespace(
        generate_content_stream=Mock(return_value=_normal_stream()),
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
    monkeypatch.setattr(
        generation_processor,
        "add_verbose_log",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_record_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        publish,
    )

    content = await generation_processor._generate_full_content(
        final_prompt="Write a short article.",
        request=_main_generation_request(),
        ai_service=ai_service,
        usage_tracker=None,
        session_id="session-normal-main",
        session=session,
        iteration=0,
        json_output_requested=False,
    )

    assert content == "alpha beta"
    assert session["generation_content"] == "alpha beta"
    assert session["partial_content"] == "alpha beta"
    assert [call.args[2] for call in publish.await_args_list] == [
        "alpha beta",
    ]
    assert publish.await_args.kwargs["subphase"] == "accepted_stream"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_plan", "expected_budgets"),
    [
        (None, [64]),
        (_bounded_retry_plan(), [64, 128]),
    ],
    ids=["no-retry-plan", "retry-also-truncated"],
)
async def test_main_generation_never_commits_unrecoverable_truncation(
    monkeypatch: pytest.MonkeyPatch,
    retry_plan: dict | None,
    expected_budgets: list[int],
) -> None:
    """No-plan and exhausted-retry truncations fail without draft exposure."""

    session = {
        "project_id": "project-truncated-main",
        "request_name": "truncated main stream",
    }
    budgets: list[int] = []
    private_drafts: list[str] = []
    publish = AsyncMock()
    debug_record = AsyncMock()

    def _truncated_stream(**kwargs):
        budgets.append(kwargs["max_tokens"])
        private_draft = f"TRUNCATED_PRIVATE_DRAFT_{len(budgets)}"
        private_drafts.append(private_draft)

        async def _chunks():
            yield private_draft
            yield StreamChunk(
                "",
                metadata={
                    "provider": "test-provider",
                    "finish_reason": "max_tokens",
                    "provider_stop_reason": "max_tokens",
                    "finish_reason_category": "length",
                    "output_truncated": True,
                    "max_tokens": kwargs["max_tokens"],
                },
            )

        return _chunks()

    _patch_main_generation_runtime(
        monkeypatch,
        cancel_check=AsyncMock(return_value=False),
        debug_record=debug_record,
        publish=publish,
    )
    monkeypatch.setattr(
        type(config),
        "resolve_bounded_output_truncation_retry",
        lambda _self, *_args, **_kwargs: retry_plan,
    )

    with pytest.raises(GenerationOutputTruncated) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a bounded response.",
            request=_main_generation_request(),
            ai_service=SimpleNamespace(
                generate_content_stream=Mock(side_effect=_truncated_stream),
            ),
            usage_tracker=None,
            session_id="session-truncated-main",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    assert budgets == expected_budgets
    assert raised.value.partial_content == ""
    assert session["generation_content"] == ""
    assert session["generation_content_length"] == 0
    assert session["partial_content"] == ""
    for private_draft in private_drafts:
        assert private_draft not in repr(session)
        assert private_draft not in repr(publish.await_args_list)
        assert private_draft not in repr(debug_record.await_args_list)
    assert not any(
        len(call.args) > 2 and call.args[2]
        for call in publish.await_args_list
    )
    assert not any(
        call.args[1] == "generator_output"
        for call in debug_record.await_args_list
    )


@pytest.mark.asyncio
async def test_generation_tool_callback_publishes_labelled_truncated_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool telemetry is visible live and labelled; the verdict stays terminal.

    The streamed ``validate_draft`` argument is the generation in progress, so
    it reaches the stream as a ``draft`` while it happens (C6). Visibility does
    not weaken canonicity: the truncation is still terminal, and the live
    session fields of a failed attempt still end up empty.
    """

    live_draft = "TOOL_LOOP_LIVE_TRUNCATED_DRAFT"
    truncated = ToolLoopEnvelope(
        loop_scope=LoopScope.GENERATOR,
        turns=1,
        accepted=False,
        accepted_via="output_truncated",
        output_schema_valid=False,
        output_truncated=True,
        truncation_reason="output_token_limit",
        finish_reason="max_tokens",
        provider_stop_reason="max_tokens",
        max_tokens=64,
    )
    publish = AsyncMock()
    debug_record = AsyncMock()
    live_events: list[object] = []
    session = {
        "project_id": "project-tool-truncated",
        "project_epoch": 1,
        "request_name": "tool truncated",
        "current_iteration": 1,
    }

    async def _capture_content_event(event: object) -> None:
        live_events.append(event)

    async def _tool_loop(**kwargs):
        await kwargs["tool_event_callback"](
            "tool_call_delta",
            {
                "delta": f'{{"text":"{live_draft}"}}',
                "candidate": live_draft,
                "content_status": ContentStatus.DRAFT,
            },
        )
        return live_draft, truncated

    _patch_main_generation_runtime(
        monkeypatch,
        cancel_check=AsyncMock(return_value=False),
        debug_record=debug_record,
        publish=publish,
        use_tools=True,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        _capture_content_event,
    )
    monkeypatch.setattr(
        type(config),
        "resolve_bounded_output_truncation_retry",
        lambda _self, *_args, **_kwargs: None,
    )

    with pytest.raises(GenerationOutputTruncated) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a bounded response.",
            request=_main_generation_request(generation_tools_mode="auto"),
            ai_service=SimpleNamespace(
                call_ai_with_validation_tools=AsyncMock(side_effect=_tool_loop),
                generate_content_stream=AsyncMock(),
            ),
            usage_tracker=None,
            session_id="session-tool-truncated",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    draft_events = [
        event
        for event in live_events
        if getattr(event, "content_status", None) == ContentStatus.DRAFT
    ]
    assert [event.content for event in draft_events] == [live_draft]
    assert draft_events[0].phase == "generation"
    assert draft_events[0].event == "chunk"
    assert draft_events[0].iteration == 1
    assert draft_events[0].attempt == 1
    assert draft_events[0].data["candidate"] == live_draft

    # Canonicity is untouched: the failed attempt leaves no live content and no
    # accepted publication behind.
    assert raised.value.partial_content == ""
    assert session["generation_content"] == ""
    assert session["partial_content"] == ""
    assert live_draft not in repr(session)
    assert live_draft not in repr(publish.await_args_list)
    assert live_draft not in repr(debug_record.await_args_list)


@pytest.mark.asyncio
async def test_cancel_during_publish_observes_only_full_committed_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation cannot split replay into a first externally visible chunk."""

    first = "FIRST_PRIVATE_CHUNK"
    second = "SECOND_PRIVATE_CHUNK"
    accepted = first + second
    session = {
        "project_id": "project-cancel-at-commit",
        "request_name": "cancel at commit",
    }
    cancellation_state = {"cancelled": False}
    debug_record = AsyncMock()

    async def _cancel_check(_session_id: str) -> bool:
        return cancellation_state["cancelled"]

    async def _publish(*_args, **_kwargs) -> None:
        cancellation_state["cancelled"] = True

    publish = AsyncMock(side_effect=_publish)

    async def _stream():
        yield first
        yield second
        yield StreamChunk(
            "",
            metadata={
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
                "output_truncated": False,
            },
        )

    _patch_main_generation_runtime(
        monkeypatch,
        cancel_check=_cancel_check,
        debug_record=debug_record,
        publish=publish,
    )

    with pytest.raises(asyncio.CancelledError):
        await generation_processor._generate_full_content(
            final_prompt="Write a bounded response.",
            request=_main_generation_request(),
            ai_service=SimpleNamespace(
                generate_content_stream=Mock(return_value=_stream()),
            ),
            usage_tracker=None,
            session_id="session-cancel-at-commit",
            session=session,
            iteration=0,
            json_output_requested=False,
        )

    assert [call.args[2] for call in publish.await_args_list] == [accepted]
    assert session["generation_content"] == accepted
    assert session["partial_content"] == accepted
    assert not any(
        call.args[1] == "generator_output"
        for call in debug_record.await_args_list
    )


@pytest.mark.asyncio
async def test_deadline_during_publish_never_leaves_first_private_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release timeout occurs after the one full-candidate commit point."""

    accepted = "FIRST_PRIVATE_CHUNKSECOND_PRIVATE_CHUNK"
    session = {
        "project_id": "project-deadline-at-commit",
        "request_name": "deadline at commit",
    }
    debug_record = AsyncMock()

    async def _blocked_publish(*_args, **_kwargs) -> None:
        await asyncio.sleep(30)

    publish = AsyncMock(side_effect=_blocked_publish)

    async def _stream():
        yield "FIRST_PRIVATE_CHUNK"
        yield "SECOND_PRIVATE_CHUNK"
        yield StreamChunk(
            "",
            metadata={
                "finish_reason": "stop",
                "finish_reason_category": "stop",
                "finish_unusable": False,
                "output_truncated": False,
            },
        )

    _patch_main_generation_runtime(
        monkeypatch,
        cancel_check=AsyncMock(return_value=False),
        debug_record=debug_record,
        publish=publish,
    )
    monkeypatch.setattr(
        generation_processor,
        "_resolve_generation_stream_timeout",
        lambda _request: 0.03,
    )

    started = time.monotonic()
    with pytest.raises(AIRequestError) as raised:
        await generation_processor._generate_full_content(
            final_prompt="Write a bounded response.",
            request=_main_generation_request(),
            ai_service=SimpleNamespace(
                generate_content_stream=Mock(return_value=_stream()),
            ),
            usage_tracker=None,
            session_id="session-deadline-at-commit",
            session=session,
            iteration=0,
            json_output_requested=False,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert raised.value.provider_failure.kind is ProviderErrorKind.TIMEOUT
    assert [call.args[2] for call in publish.await_args_list] == [accepted]
    assert session["generation_content"] == accepted
    assert session["partial_content"] == accepted
    assert not any(
        call.args[1] == "generator_output"
        for call in debug_record.await_args_list
    )


def _structured_main_request() -> ContentRequest:
    return ContentRequest(
        prompt="Return the requested bounded JSON object.",
        content_type="json",
        json_output=True,
        json_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        json_retry_without_iteration=False,
        generator_model="gpt-4o",
        generation_tools_mode="never",
        max_iterations=1,
        max_tokens=64,
        qa_layers=[],
        qa_models=[],
        auto_qa={"enabled": False},
        gran_sabio_fallback=False,
        long_text_mode="off",
    )


class _StaticMainStreamAI:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate_content_stream(self, **_kwargs):
        async def _stream():
            yield StreamChunk(self.content)
            yield StreamChunk(
                "",
                metadata={
                    "provider": "offline",
                    "finish_reason": "stop",
                    "finish_reason_category": "stop",
                    "finish_unusable": False,
                    "output_truncated": False,
                    "max_tokens": 64,
                },
            )

        return _stream()


def _patch_structured_process_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    debug_record: AsyncMock,
    publish: AsyncMock,
    publish_end: AsyncMock,
    phase_logger: MagicMock,
) -> None:
    monkeypatch.setattr(generation_processor, "_ensure_services", lambda: None)
    monkeypatch.setattr(
        generation_processor,
        "get_feedback_manager",
        lambda: _FeedbackManager(),
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
        "_resolve_generation_stream_timeout",
        lambda _request: 1.0,
    )
    monkeypatch.setattr(
        generation_processor,
        "create_phase_logger",
        Mock(return_value=phase_logger),
    )
    monkeypatch.setattr(
        generation_processor,
        "add_verbose_log",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_record_event",
        debug_record,
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_update_status",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        publish,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_session_end",
        publish_end,
    )


@pytest.mark.asyncio
async def test_initial_invalid_json_candidate_never_commits_from_generation_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_json = '{"broken":'
    session = {
        "project_id": "project-invalid-json-helper",
        "project_epoch": 1,
        "request_name": "invalid json helper",
    }
    debug_record = AsyncMock()
    publish = AsyncMock()
    _patch_main_generation_runtime(
        monkeypatch,
        cancel_check=AsyncMock(return_value=False),
        debug_record=debug_record,
        publish=publish,
    )

    candidate = await generation_processor.generate_iteration_candidate(
        request=_structured_main_request(),
        session=session,
        session_id="session-invalid-json-helper",
        final_prompt="Return JSON.",
        context_prompt="",
        ai_service=_StaticMainStreamAI(private_json),
        usage_tracker=None,
        iteration=0,
        json_output_requested=True,
        phase_logger=None,
        images=None,
    )

    assert candidate.content == private_json
    assert session.get("generation_content", "") == ""
    assert session.get("partial_content", "") == ""
    assert private_json not in repr(session)
    assert private_json not in repr(publish.await_args_list)
    # C13.2: the debugger is the one surface that DOES keep the rejected
    # candidate, verbatim and labelled as unvalidated. That is the evidence
    # needed to explain why a run kept retrying; it is not a commit, and the
    # live/canonical surfaces above stay empty.
    unvalidated_outputs = [
        call.args[2]
        for call in debug_record.await_args_list
        if call.args[1] == "generator_output"
    ]
    assert [payload["content"] for payload in unvalidated_outputs] == [private_json]
    assert all(
        payload["json_output"] is True and payload["json_guard_validated"] is False
        for payload in unvalidated_outputs
    )


@pytest.mark.asyncio
async def test_initial_invalid_json_is_absent_from_every_released_surface(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_json = '{"broken":'
    session_id = f"invalid-json-process-{uuid4()}"
    session = {
        "status": "initialized",
        "project_id": "project-invalid-json-process",
        "project_epoch": 1,
        "request_name": "invalid json process",
    }
    debug_record = AsyncMock()
    publish = AsyncMock()
    publish_end = AsyncMock()
    phase_logger = MagicMock()
    phase_logger.extra_verbose = False
    _patch_structured_process_runtime(
        monkeypatch,
        debug_record=debug_record,
        publish=publish,
        publish_end=publish_end,
        phase_logger=phase_logger,
    )
    app_state.active_sessions[session_id] = session
    app_state.ai_service = _StaticMainStreamAI(private_json)

    try:
        with caplog.at_level("WARNING"):
            await generation_processor.process_content_generation(
                session_id,
                _structured_main_request(),
            )

        assert session["status"] == GenerationStatus.FAILED.value
        assert session["final_result"]["content"] == ""
        assert session.get("generation_content", "") == ""
        assert session.get("last_generated_content", "") == ""
        assert session.get("partial_content", "") == ""
        assert private_json not in repr(session)
        assert private_json not in repr(publish.await_args_list)
        assert private_json not in repr(publish_end.await_args_list)
        assert private_json not in caplog.text
        phase_logger.log_response.assert_not_called()
        # C13.2: the rejected candidate survives in the debugger only, marked
        # as an unvalidated JSON draft. Every released surface stays clean.
        assert [
            call.args[2]["content"]
            for call in debug_record.await_args_list
            if call.args[1] == "generator_output"
        ] == [private_json]
    finally:
        app_state.active_sessions.pop(session_id, None)
        app_state.ai_service = None


@pytest.mark.asyncio
async def test_valid_json_commits_normalized_content_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_json = ' { "ok" : true } '
    normalized_json = '{"ok":true}'
    session_id = f"valid-json-process-{uuid4()}"
    session = {
        "status": "initialized",
        "project_id": "project-valid-json-process",
        "project_epoch": 1,
        "request_name": "valid json process",
    }
    debug_record = AsyncMock()
    publish = AsyncMock()
    publish_end = AsyncMock()
    phase_logger = MagicMock()
    phase_logger.extra_verbose = False
    _patch_structured_process_runtime(
        monkeypatch,
        debug_record=debug_record,
        publish=publish,
        publish_end=publish_end,
        phase_logger=phase_logger,
    )
    app_state.active_sessions[session_id] = session
    app_state.ai_service = _StaticMainStreamAI(raw_json)

    try:
        await generation_processor.process_content_generation(
            session_id,
            _structured_main_request(),
        )

        assert session["status"] == GenerationStatus.COMPLETED.value
        assert session["generation_content"] == normalized_json
        assert session["last_generated_content"] == normalized_json
        assert session["partial_content"] == normalized_json
        assert session["final_result"]["content"] == {"ok": True}
        published_content = [
            call.args[2]
            for call in publish.await_args_list
            if len(call.args) > 2 and call.args[2] is not None
        ]
        assert published_content == [normalized_json]
        # C13.2: the debugger records both the raw provider output and the
        # normalized payload the guard released, in that order. The single
        # canonical commit above is what "exactly once" is about; the raw draft
        # is traceability, and it never reaches the bus.
        generator_outputs = [
            (call.args[2]["content"], call.args[2]["json_guard_validated"])
            for call in debug_record.await_args_list
            if call.args[1] == "generator_output"
        ]
        assert generator_outputs == [
            (raw_json, False),
            (normalized_json, True),
        ]
        assert raw_json not in repr(publish.await_args_list)
    finally:
        app_state.active_sessions.pop(session_id, None)
        app_state.ai_service = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_mode",
    ["content_filter", "cancel", "truncated", "timeout"],
)
async def test_json_retry_terminal_failure_keeps_draft_out_of_canonical_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    terminal_mode: str,
) -> None:
    """C2: the JSON retry feeds the live snapshot, never the canonical surfaces.

    The retry loop now streams through the shared chunk handler, so it flushes
    the live session snapshot like the main loop does. What must never happen
    is the unvalidated draft reaching the terminal result, the debugger result
    payloads or the accepted commit events.
    """

    session_id = f"json-retry-integrity-{terminal_mode}-{uuid4()}"
    private_draft = f"JSON_RETRY_PRIVATE_DRAFT_{terminal_mode}"
    session = {
        "status": "initialized",
        "project_id": f"project-{terminal_mode}",
        "request_name": f"json retry {terminal_mode}",
    }
    debug_record = AsyncMock()
    publish = AsyncMock()
    publish_end = AsyncMock()
    flush_snapshot = Mock()
    live_events: list[object] = []

    async def _capture_content_event(event: object) -> None:
        live_events.append(event)
    initial_candidate = SimpleNamespace(
        content="not valid json",
        mode="standard",
        long_text_state=None,
        controller_summary=None,
        diagnostics_summary=None,
        used_tool_loop=False,
    )

    class _JsonRetryFailureAI:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content_stream(self, **_kwargs):
            self.calls += 1

            async def _stream():
                yield private_draft
                if terminal_mode == "content_filter":
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
                elif terminal_mode == "truncated":
                    yield StreamChunk(
                        "",
                        metadata={
                            "provider": "test-provider",
                            "finish_reason": "max_tokens",
                            "provider_stop_reason": "max_tokens",
                            "finish_reason_category": "length",
                            "output_truncated": True,
                            "max_tokens": 64,
                        },
                    )
                elif terminal_mode == "cancel":
                    session["cancelled"] = True
                else:
                    await asyncio.sleep(30)

            return _stream()

    request = ContentRequest(
        prompt="Generate compact JSON for an integrity test.",
        content_type="json",
        json_output=False,
        json_retry_without_iteration=True,
        generator_model="gpt-4o",
        max_iterations=1,
        max_tokens=64,
        timeout_seconds=0.03 if terminal_mode == "timeout" else 1.0,
        qa_layers=[],
        qa_models=[],
        auto_qa={"enabled": False},
        long_text_mode="off",
    )

    retry_ai = _JsonRetryFailureAI()
    app_state.active_sessions[session_id] = session
    app_state.ai_service = retry_ai
    monkeypatch.setattr(generation_processor, "_ensure_services", lambda: None)
    monkeypatch.setattr(
        generation_processor,
        "get_feedback_manager",
        lambda: _FeedbackManager(),
    )
    monkeypatch.setattr(
        generation_processor,
        "generate_iteration_candidate",
        AsyncMock(return_value=initial_candidate),
    )
    monkeypatch.setattr(config, "MAX_JSON_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(
        generation_processor,
        "resolve_call",
        lambda *_args, **_kwargs: SimpleNamespace(
            call_id="generation.json_retry",
            model="gpt-4o",
            params={"temperature": 0.2},
        ),
    )
    monkeypatch.setattr(
        generation_processor,
        "_resolve_generation_stream_timeout",
        lambda _request: 0.03 if terminal_mode == "timeout" else 1.0,
    )
    monkeypatch.setattr(
        generation_processor,
        "_flush_stream_session_snapshot",
        flush_snapshot,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_content_event",
        _capture_content_event,
    )
    monkeypatch.setattr(
        generation_processor,
        "add_verbose_log",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_record_event",
        debug_record,
    )
    monkeypatch.setattr(
        generation_processor,
        "_debug_update_status",
        AsyncMock(),
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_phase_chunk",
        publish,
    )
    monkeypatch.setattr(
        generation_processor,
        "publish_project_session_end",
        publish_end,
    )

    try:
        if terminal_mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await generation_processor.process_content_generation(
                    session_id,
                    request,
                )
        else:
            await generation_processor.process_content_generation(
                session_id,
                request,
            )

        flush_snapshot.assert_called()
        # The retry draft is visible live, labelled as a draft.
        draft_events = [
            event
            for event in live_events
            if getattr(event, "content_status", None) == "draft"
        ]
        assert [event.content for event in draft_events] == [private_draft]
        assert retry_ai.calls == 1
        assert session.get("generation_content", "") == ""
        assert session.get("last_generated_content", "") == ""
        assert session.get("partial_content", "") == ""
        assert private_draft not in repr(session)
        assert private_draft not in repr(debug_record.await_args_list)
        assert private_draft not in repr(publish.await_args_list)
        assert private_draft not in repr(publish_end.await_args_list)
        if terminal_mode == "cancel":
            assert session["status"] == GenerationStatus.CANCELLED.value
        else:
            expected_error_code = {
                "content_filter": "content_filter",
                "truncated": "output_token_limit",
                "timeout": ProviderErrorKind.TIMEOUT.value,
            }[terminal_mode]
            assert session["final_result"]["error_code"] == expected_error_code
    finally:
        app_state.active_sessions.pop(session_id, None)
        app_state.ai_service = None
