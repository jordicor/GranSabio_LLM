from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from ai_service import GenerationOutputTruncated, StreamChunk
from arbiter import Arbiter, ArbiterContext, LayerEditHistory
from config import config
from core.generation_processor import _generate_full_content
from gran_sabio import GranSabioEngine
from models import ContentRequest
from tool_loop_models import LoopScope, ToolLoopEnvelope


def _generator_request(**overrides):
    request = SimpleNamespace(
        generator_model="claude-opus-4-8",
        generation_tools_mode="never",
        max_iterations=3,
        temperature=0.2,
        max_tokens=4096,
        system_prompt=None,
        extra_verbose=False,
        reasoning_effort=None,
        thinking_budget_tokens=None,
        content_type="json",
        json_output=True,
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
    )
    for key, value in overrides.items():
        setattr(request, key, value)
    request._external_generation_min_tokens_adjustment = {
        "original_tokens": 3072,
        "adjusted_tokens": 4096,
    }
    return request


def _retry_plan(
    *,
    requested: int = 3072,
    effective: int = 4096,
    retry: int = 8192,
    safe: int = 121600,
):
    return {
        "requested_max_tokens": requested,
        "effective_max_tokens": effective,
        "retry_max_tokens": retry,
        "model_safe_max_tokens": safe,
        "retry_ordinal": 1,
        "growth_policy": "double_or_plus_1024_capped_to_model_safe_limit",
    }


def test_bounded_retry_plan_doubles_without_jumping_to_model_ceiling():
    with patch.object(
        type(config),
        "resolve_output_truncation_retry_max_tokens",
        return_value=121600,
    ):
        plan = config.resolve_bounded_output_truncation_retry(
            "claude-opus-4-8",
            4096,
            requested_max_tokens=3072,
        )

    assert plan == _retry_plan()


@pytest.mark.asyncio
async def test_normal_stream_retries_truncation_once_with_budget_telemetry():
    request = _generator_request()
    budgets = []

    def _stream(**kwargs):
        budgets.append(kwargs["max_tokens"])

        async def _chunks():
            if len(budgets) == 1:
                yield StreamChunk("partial", is_thinking=False)
                yield StreamChunk(
                    "",
                    metadata={
                        "provider": "anthropic",
                        "finish_reason": "max_tokens",
                        "provider_stop_reason": "max_tokens",
                        "output_truncated": True,
                        "max_tokens": 4096,
                    },
                )
            else:
                yield StreamChunk('{"ok":true}', is_thinking=False)
                yield StreamChunk(
                    "",
                    metadata={
                        "provider": "anthropic",
                        "finish_reason": "end_turn",
                        "provider_stop_reason": "end_turn",
                        "output_truncated": False,
                        "max_tokens": 8192,
                    },
                )

        return _chunks()

    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(),
        generate_content_stream=Mock(side_effect=_stream),
    )
    session = {}
    debug_event = AsyncMock()

    with patch(
        "core.generation_processor._should_use_generation_tools",
        return_value=False,
    ), patch(
        "core.generation_processor._resolve_generation_reasoning_telemetry",
        return_value={
            "requested_reasoning_effort": None,
            "requested_thinking_budget_tokens": None,
            "reasoning_effort": "high",
            "thinking_budget_tokens": None,
            "is_thinking_model": True,
        },
    ), patch(
        "core.generation_processor.check_session_cancelled",
        new=AsyncMock(return_value=False),
    ), patch(
        "core.generation_processor.add_verbose_log",
        new=AsyncMock(),
    ), patch(
        "core.generation_processor._debug_record_event",
        new=debug_event,
    ), patch.object(
        type(config),
        "resolve_bounded_output_truncation_retry",
        return_value=_retry_plan(),
    ):
        content = await _generate_full_content(
            final_prompt="Return compact JSON.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-stream-retry",
            session=session,
            iteration=0,
            json_output_requested=True,
        )

    assert content == '{"ok":true}'
    assert budgets == [4096, 8192]
    assert len(session["output_truncation_retries"]) == 1
    retry = session["output_truncation_retries"][0]
    assert retry["requested_max_tokens"] == 3072
    assert retry["effective_max_tokens"] == 4096
    assert retry["retry_max_tokens"] == 8192
    retry_events = [
        call for call in debug_event.await_args_list
        if call.args[1] == "output_truncation_retry_scheduled"
    ]
    assert len(retry_events) == 1


@pytest.mark.asyncio
async def test_normal_tool_loop_retries_truncation_only_once():
    request = _generator_request(generation_tools_mode="auto")
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
        max_tokens=4096,
    )
    complete = ToolLoopEnvelope(
        loop_scope=LoopScope.GENERATOR,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        output_schema_valid=True,
        output_truncated=False,
        finish_reason="end_turn",
        provider_stop_reason="end_turn",
        max_tokens=8192,
    )
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(
            side_effect=[("", truncated), ('{"ok":true}', complete)]
        ),
        generate_content_stream=AsyncMock(),
    )

    with patch(
        "core.generation_processor._should_use_generation_tools",
        return_value=True,
    ), patch(
        "core.generation_processor.has_active_generation_validators",
        return_value=True,
    ), patch(
        "core.generation_processor._resolve_generation_reasoning_telemetry",
        return_value={
            "requested_reasoning_effort": None,
            "requested_thinking_budget_tokens": None,
            "reasoning_effort": "high",
            "thinking_budget_tokens": None,
            "is_thinking_model": True,
        },
    ), patch(
        "core.generation_processor.add_verbose_log",
        new=AsyncMock(),
    ), patch(
        "core.generation_processor._debug_record_event",
        new=AsyncMock(),
    ), patch(
        "core.generation_processor.time.monotonic",
        side_effect=[100.0, 103.0],
    ), patch.object(
        type(config),
        "resolve_bounded_output_truncation_retry",
        return_value=_retry_plan(),
    ):
        content = await _generate_full_content(
            final_prompt="Return compact JSON.",
            request=request,
            ai_service=ai_service,
            usage_tracker=None,
            session_id="session-tool-retry",
            session={},
            iteration=0,
            json_output_requested=True,
        )

    assert content == '{"ok":true}'
    assert [
        call.kwargs["max_tokens"]
        for call in ai_service.call_ai_with_validation_tools.await_args_list
    ] == [4096, 8192]
    timeouts = [
        call.kwargs["request_timeout"]
        for call in ai_service.call_ai_with_validation_tools.await_args_list
    ]
    assert timeouts[1] == pytest.approx(timeouts[0] - 3.0)
    ai_service.generate_content_stream.assert_not_called()


@pytest.mark.asyncio
async def test_arbiter_tool_loop_uses_bounded_retry_and_remaining_deadline():
    truncated = ToolLoopEnvelope(
        loop_scope=LoopScope.ARBITER,
        turns=1,
        accepted=False,
        accepted_via="output_truncated",
        output_schema_valid=False,
        output_truncated=True,
        truncation_reason="output_token_limit",
        finish_reason="max_tokens",
        provider_stop_reason="max_tokens",
        max_tokens=4096,
    )
    recovered_payload = {
        "reasoning": "Recovered within the original phase deadline.",
        "decisions": [],
        "conflicts_resolved": [],
    }
    recovered = ToolLoopEnvelope(
        loop_scope=LoopScope.ARBITER,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        output_schema_valid=True,
        output_truncated=False,
        payload=recovered_payload,
        finish_reason="end_turn",
        provider_stop_reason="end_turn",
        max_tokens=8192,
    )
    ai_service = SimpleNamespace(
        call_ai_with_validation_tools=AsyncMock(
            side_effect=[("", truncated), ("", recovered)]
        )
    )
    arbiter = Arbiter(ai_service=ai_service, model="claude-opus-4-8")
    context = ArbiterContext(
        original_prompt="Apply only well-supported edits.",
        content_type="other",
        system_prompt=None,
        layer_name="Accuracy",
        layer_criteria="Preserve supported facts.",
        layer_min_score=8.0,
        current_content="Candidate.",
        content_excerpt=None,
        proposed_edits=[],
        evaluator_scores={},
        layer_history=LayerEditHistory(layer_name="Accuracy"),
        original_request=SimpleNamespace(),
    )

    with patch.object(
        arbiter,
        "_resolve_request_timeout",
        return_value=2.0,
    ), patch.object(
        arbiter,
        "_resolve_output_max_tokens",
        return_value=(4096, {}),
    ), patch.object(
        type(config),
        "resolve_bounded_output_truncation_retry",
        return_value=_retry_plan(requested=4096),
    ), patch(
        "arbiter.time.monotonic",
        side_effect=[100.0, 101.0],
    ):
        result = await arbiter._resolve_with_tool_loop(
            context=context,
            prompt="Resolve the proposed edits.",
            selected_model="claude-opus-4-8",
            prompt_safety_parts=None,
        )

    assert result == recovered_payload
    calls = ai_service.call_ai_with_validation_tools.await_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [4096, 8192]
    assert calls[1].kwargs["request_timeout"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_gransabio_streaming_fallback_retries_truncation_once():
    request = ContentRequest(
        prompt="Return compact JSON.",
        content_type="json",
        json_output=True,
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=4096,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )
    object.__setattr__(
        request,
        "_external_generation_min_tokens_adjustment",
        {"original_tokens": 3072, "adjusted_tokens": 4096},
    )
    budgets = []

    def _stream(**kwargs):
        budgets.append(kwargs["max_tokens"])

        async def _chunks():
            if len(budgets) == 1:
                yield StreamChunk("partial", is_thinking=False)
                yield StreamChunk(
                    "",
                    metadata={
                        "provider": "anthropic",
                        "finish_reason": "max_tokens",
                        "provider_stop_reason": "max_tokens",
                        "output_truncated": True,
                        "max_tokens": 4096,
                    },
                )
            else:
                yield StreamChunk('{"ok":true}', is_thinking=False)
                yield StreamChunk(
                    "",
                    metadata={
                        "provider": "anthropic",
                        "finish_reason": "end_turn",
                        "provider_stop_reason": "end_turn",
                        "output_truncated": False,
                        "max_tokens": 8192,
                    },
                )

        return _chunks()

    service = MagicMock()
    service.generate_content_stream = MagicMock(side_effect=_stream)
    service.call_ai_with_validation_tools = AsyncMock()
    stream_callback = AsyncMock()
    telemetry_callback = AsyncMock()
    engine = GranSabioEngine(ai_service=service)

    with patch.object(
        engine,
        "_get_default_content_generation_model",
        return_value=("claude-opus-4-8", "high", None, 0.2),
    ), patch.object(
        engine,
        "_resolve_gran_sabio_max_tokens",
        return_value=4096,
    ), patch(
        "gran_sabio._resolve_output_truncation_retry_plan",
        return_value=_retry_plan(),
    ), patch.object(
        config,
        "MAX_RETRIES",
        1,
    ), patch.object(
        config,
        "RETRY_STREAMING_AFTER_PARTIAL",
        False,
    ):
        result = await engine.regenerate_content(
            session_id="session-fallback-retry",
            original_request=request,
            stream_callback=stream_callback,
            tool_event_callback=telemetry_callback,
        )

    assert result.approved is True
    assert result.final_content == '{"ok":true}'
    assert budgets == [4096, 8192]
    retry_events = [
        call for call in telemetry_callback.await_args_list
        if call.args[0] == "output_truncation_retry_scheduled"
    ]
    assert len(retry_events) == 1
    assert retry_events[0].args[1]["requested_max_tokens"] == 3072
    assert retry_events[0].args[1]["effective_max_tokens"] == 4096
    assert retry_events[0].args[1]["retry_max_tokens"] == 8192


@pytest.mark.asyncio
async def test_gransabio_streaming_regeneration_exhaustion_stays_typed():
    request = ContentRequest(
        prompt="Return compact JSON.",
        content_type="json",
        json_output=True,
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=4096,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )
    object.__setattr__(
        request,
        "_external_generation_min_tokens_adjustment",
        {"original_tokens": 3072, "adjusted_tokens": 4096},
    )
    budgets = []

    def _stream(**kwargs):
        budgets.append(kwargs["max_tokens"])

        async def _chunks():
            yield StreamChunk("partial", is_thinking=False)
            yield StreamChunk(
                "",
                metadata={
                    "provider": "anthropic",
                    "finish_reason": "max_tokens",
                    "provider_stop_reason": "max_tokens",
                    "output_truncated": True,
                    "max_tokens": kwargs["max_tokens"],
                },
            )

        return _chunks()

    service = MagicMock()
    service.generate_content_stream = MagicMock(side_effect=_stream)
    service.call_ai_with_validation_tools = AsyncMock()
    engine = GranSabioEngine(ai_service=service)

    with patch.object(
        engine,
        "_get_default_content_generation_model",
        return_value=("claude-opus-4-8", "high", None, 0.2),
    ), patch.object(
        engine,
        "_resolve_gran_sabio_max_tokens",
        return_value=4096,
    ), patch(
        "gran_sabio._resolve_output_truncation_retry_plan",
        return_value=_retry_plan(),
    ), patch.object(
        config,
        "MAX_RETRIES",
        1,
    ), patch.object(
        config,
        "RETRY_STREAMING_AFTER_PARTIAL",
        False,
    ):
        with pytest.raises(GenerationOutputTruncated) as exc_info:
            await engine.regenerate_content(
                session_id="session-regeneration-stream-truncated",
                original_request=request,
                stream_callback=AsyncMock(),
                tool_event_callback=AsyncMock(),
            )

    assert budgets == [4096, 8192]
    assert exc_info.value.finish_reason == "max_tokens"
    assert exc_info.value.max_tokens == 8192
    assert exc_info.value.partial_content == "partial"



@pytest.mark.asyncio
async def test_gransabio_tool_loop_fallback_uses_escalated_retry_budget():
    request = ContentRequest(
        prompt="Return compact JSON.",
        content_type="json",
        json_output=True,
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=4096,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="auto",
    )
    object.__setattr__(
        request,
        "_external_generation_min_tokens_adjustment",
        {"original_tokens": 3072, "adjusted_tokens": 4096},
    )
    truncated = ToolLoopEnvelope(
        loop_scope=LoopScope.GRAN_SABIO,
        turns=1,
        accepted=False,
        accepted_via="output_truncated",
        output_schema_valid=False,
        output_truncated=True,
        truncation_reason="output_token_limit",
        finish_reason="max_tokens",
        provider_stop_reason="max_tokens",
        max_tokens=4096,
    )
    complete = ToolLoopEnvelope(
        loop_scope=LoopScope.GRAN_SABIO,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        output_schema_valid=True,
        output_truncated=False,
        finish_reason="end_turn",
        provider_stop_reason="end_turn",
        max_tokens=8192,
    )
    service = MagicMock()
    service.call_ai_with_validation_tools = AsyncMock(
        side_effect=[("", truncated), ('{"ok":true}', complete)]
    )
    service.generate_content = AsyncMock()
    telemetry_callback = AsyncMock()
    engine = GranSabioEngine(ai_service=service)

    with patch(
        "gran_sabio._should_use_gransabio_tools",
        return_value=True,
    ), patch.object(
        engine,
        "_get_default_content_generation_model",
        return_value=("claude-opus-4-8", "high", None, 0.2),
    ), patch.object(
        engine,
        "_resolve_gran_sabio_max_tokens",
        return_value=4096,
    ), patch.object(
        type(config),
        "resolve_bounded_output_truncation_retry",
        return_value=_retry_plan(),
    ):
        result = await engine.regenerate_content(
            session_id="session-fallback-tool-retry",
            original_request=request,
            tool_event_callback=telemetry_callback,
        )

    assert result.approved is True
    assert result.final_content == '{"ok":true}'
    assert [
        call.kwargs["max_tokens"]
        for call in service.call_ai_with_validation_tools.await_args_list
    ] == [4096, 8192]
    timeouts = [
        call.kwargs["request_timeout"]
        for call in service.call_ai_with_validation_tools.await_args_list
    ]
    assert 0 < timeouts[1] <= timeouts[0]
    service.generate_content.assert_not_called()
    retry_events = [
        call for call in telemetry_callback.await_args_list
        if call.args[0] == "output_truncation_retry_scheduled"
    ]
    assert len(retry_events) == 1


@pytest.mark.asyncio
async def test_gransabio_tool_loop_regeneration_exhaustion_stays_typed():
    request = ContentRequest(
        prompt="Return compact JSON.",
        content_type="json",
        json_output=True,
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=4096,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="auto",
    )
    object.__setattr__(
        request,
        "_external_generation_min_tokens_adjustment",
        {"original_tokens": 3072, "adjusted_tokens": 4096},
    )

    def _truncated(max_tokens: int) -> ToolLoopEnvelope:
        return ToolLoopEnvelope(
            loop_scope=LoopScope.GRAN_SABIO,
            turns=1,
            accepted=False,
            accepted_via="output_truncated",
            output_schema_valid=False,
            output_truncated=True,
            truncation_reason="output_token_limit",
            finish_reason="max_tokens",
            provider_stop_reason="max_tokens",
            max_tokens=max_tokens,
        )

    service = MagicMock()
    service.call_ai_with_validation_tools = AsyncMock(
        side_effect=[
            ("first-partial", _truncated(4096)),
            ("retry-partial", _truncated(8192)),
        ]
    )
    service.generate_content = AsyncMock()
    engine = GranSabioEngine(ai_service=service)

    with patch(
        "gran_sabio._should_use_gransabio_tools",
        return_value=True,
    ), patch.object(
        engine,
        "_get_default_content_generation_model",
        return_value=("claude-opus-4-8", "high", None, 0.2),
    ), patch.object(
        engine,
        "_resolve_gran_sabio_max_tokens",
        return_value=4096,
    ), patch.object(
        type(config),
        "resolve_bounded_output_truncation_retry",
        return_value=_retry_plan(),
    ):
        with pytest.raises(GenerationOutputTruncated) as exc_info:
            await engine.regenerate_content(
                session_id="session-regeneration-tool-truncated",
                original_request=request,
                tool_event_callback=AsyncMock(),
            )

    assert [
        call.kwargs["max_tokens"]
        for call in service.call_ai_with_validation_tools.await_args_list
    ] == [4096, 8192]
    assert exc_info.value.finish_reason == "max_tokens"
    assert exc_info.value.max_tokens == 8192
    assert exc_info.value.partial_content == "retry-partial"
    service.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_gransabio_review_stream_exhaustion_stays_typed():
    request = ContentRequest(
        prompt="Return compact JSON.",
        content_type="json",
        json_output=True,
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=4096,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode="never",
    )
    object.__setattr__(
        request,
        "_external_generation_min_tokens_adjustment",
        {"original_tokens": 3072, "adjusted_tokens": 4096},
    )
    budgets = []

    def _stream(**kwargs):
        budgets.append(kwargs["max_tokens"])

        async def _chunks():
            yield StreamChunk("partial", is_thinking=False)
            yield StreamChunk(
                "",
                metadata={
                    "provider": "anthropic",
                    "finish_reason": "max_tokens",
                    "provider_stop_reason": "max_tokens",
                    "output_truncated": True,
                    "max_tokens": kwargs["max_tokens"],
                },
            )

        return _chunks()

    service = MagicMock()
    service.generate_content_stream = MagicMock(side_effect=_stream)
    service.call_ai_with_validation_tools = AsyncMock()
    engine = GranSabioEngine(ai_service=service)

    with patch.object(
        engine,
        "_get_default_critical_analysis_model",
        return_value=("claude-opus-4-8", "high", None, 0.2),
    ), patch.object(
        engine,
        "_resolve_gran_sabio_max_tokens",
        return_value=4096,
    ), patch(
        "gran_sabio._resolve_output_truncation_retry_plan",
        return_value=_retry_plan(),
    ), patch.object(
        config,
        "MAX_RETRIES",
        1,
    ), patch.object(
        config,
        "RETRY_STREAMING_AFTER_PARTIAL",
        False,
    ):
        with pytest.raises(GenerationOutputTruncated) as exc_info:
            await engine.review_iterations(
                session_id="session-review-truncated",
                iterations=[
                    {
                        "content": '{"candidate":true}',
                        "global_score": 7.0,
                        "qa_results": {},
                    }
                ],
                original_request=request,
                stream_callback=AsyncMock(),
            )

    assert budgets == [4096, 8192]
    assert exc_info.value.finish_reason == "max_tokens"
    assert exc_info.value.max_tokens == 8192
    assert exc_info.value.partial_content == "partial"
