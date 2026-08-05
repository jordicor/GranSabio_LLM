"""Offline regressions for Gran Sabio operation-wide retry deadlines."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import gran_sabio as gran_sabio_module
from ai_service import StreamChunk
from config import config
from gran_sabio import GranSabioEngine
from models import ContentRequest
from tool_loop_models import LoopScope, ToolLoopEnvelope


def _request(*, tools_mode: str = "auto") -> ContentRequest:
    return ContentRequest(
        prompt="Produce a bounded generic test response.",
        content_type="other",
        generator_model="claude-opus-4-8",
        gran_sabio_model="claude-opus-4-8",
        max_tokens=128,
        qa_layers=[],
        qa_models=[],
        gransabio_tools_mode=tools_mode,
    )


def _retry_plan() -> dict[str, object]:
    return {
        "requested_max_tokens": 128,
        "effective_max_tokens": 128,
        "retry_max_tokens": 256,
        "model_safe_max_tokens": 4096,
        "retry_ordinal": 1,
        "growth_policy": "bounded_test_growth",
    }


def _truncated_envelope(max_tokens: int = 128) -> ToolLoopEnvelope:
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


def _complete_envelope() -> ToolLoopEnvelope:
    return ToolLoopEnvelope(
        loop_scope=LoopScope.GRAN_SABIO,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        output_schema_valid=True,
        output_truncated=False,
        finish_reason="end_turn",
        provider_stop_reason="end_turn",
        max_tokens=256,
    )


def _no_tool_support_envelope() -> ToolLoopEnvelope:
    return ToolLoopEnvelope(
        loop_scope=LoopScope.GRAN_SABIO,
        tools_skipped_reason="no_tool_support",
        turns=0,
        accepted=False,
        accepted_via="tools_skipped",
        payload=None,
    )


def _engine(ai_service: object) -> GranSabioEngine:
    engine = GranSabioEngine(ai_service=ai_service)
    engine._get_default_content_generation_model = Mock(
        return_value=("claude-opus-4-8", "high", None, 0.2)
    )
    engine._resolve_gran_sabio_max_tokens = Mock(return_value=128)
    engine._resolve_request_timeout = Mock(return_value=10.0)
    return engine


def _review_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }


async def _invoke_review(
    engine: GranSabioEngine,
    request: ContentRequest,
    *,
    stream_callback: object = None,
) -> tuple[dict[str, object], str]:
    return await engine._invoke_gran_sabio_llm(
        request=request,
        prompt="Return strict JSON.",
        model="claude-opus-4-8",
        reasoning_effort="high",
        thinking_tokens=None,
        temperature=0.2,
        system_prompt="Return JSON.",
        response_schema=_review_schema(),
        max_tool_rounds=1,
        initial_measurement_text=None,
        usage_callback=None,
        phase_logger=None,
        alias_registry=None,
        prompt_safety_parts=None,
        stream_callback=stream_callback,
        stream_stage_label="review",
        cancel_callback=None,
        abort_stage="review_stream",
        max_tokens=128,
    )


def test_remaining_timeout_is_bound_to_one_monotonic_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )

    assert gran_sabio_module._remaining_gran_sabio_timeout(
        110.0,
        operation="test operation",
    ) == pytest.approx(10.0)
    clock[0] = 110.0
    with pytest.raises(asyncio.TimeoutError, match="process deadline"):
        gran_sabio_module._remaining_gran_sabio_timeout(
            110.0,
            operation="test operation",
        )


@pytest.mark.asyncio
async def test_regeneration_tool_loop_retry_uses_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    timeouts: list[float] = []

    async def call_tool_loop(**kwargs: object) -> tuple[str, ToolLoopEnvelope]:
        timeouts.append(float(kwargs["request_timeout"]))
        if len(timeouts) == 1:
            clock[0] += 3.0
            return "partial", _truncated_envelope()
        return "regenerated", _complete_envelope()

    service = MagicMock()
    service.call_ai_with_validation_tools = AsyncMock(side_effect=call_tool_loop)
    service.generate_content = AsyncMock()
    engine = _engine(service)
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )

    with (
        patch("gran_sabio._should_use_gransabio_tools", return_value=True),
        patch.object(
            type(config),
            "resolve_bounded_output_truncation_retry",
            return_value=_retry_plan(),
        ),
    ):
        result = await engine.regenerate_content(
            session_id="regeneration-tool-deadline",
            original_request=_request(),
        )

    assert result.approved is True
    assert result.final_content == "regenerated"
    assert timeouts == pytest.approx([10.0, 7.0])
    service.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_regeneration_stream_retry_uses_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [200.0]
    timeouts: list[float] = []

    class StreamingService:
        def generate_content_stream(self, **kwargs: object):
            timeouts.append(float(kwargs["request_timeout"]))

            async def chunks():
                if len(timeouts) == 1:
                    clock[0] += 2.0
                    raise TimeoutError("transient adapter timeout")
                yield "recovered"

            return chunks()

    engine = _engine(StreamingService())
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(config, "MAX_RETRIES", 3)
    monkeypatch.setattr(config, "RETRY_STREAMING_AFTER_PARTIAL", True)
    monkeypatch.setattr(config, "RETRY_DELAY", 0.0)

    with patch(
        "gran_sabio._should_use_gransabio_tools",
        return_value=False,
    ):
        result = await engine.regenerate_content(
            session_id="regeneration-stream-deadline",
            original_request=_request(tools_mode="never"),
            stream_callback=AsyncMock(),
        )

    assert result.approved is True
    assert result.final_content == "recovered"
    assert timeouts == pytest.approx([10.0, 8.0])


@pytest.mark.asyncio
async def test_review_tool_fallback_keeps_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [300.0]
    timeouts: list[float] = []

    async def call_tool_loop(**kwargs: object):
        timeouts.append(float(kwargs["request_timeout"]))
        clock[0] += 4.0
        return "", _no_tool_support_envelope()

    async def generate_content(**kwargs: object) -> str:
        timeouts.append(float(kwargs["request_timeout"]))
        return '{"ok":true}'

    service = MagicMock()
    service.call_ai_with_validation_tools = AsyncMock(side_effect=call_tool_loop)
    service.generate_content = AsyncMock(side_effect=generate_content)
    engine = _engine(service)
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )

    with patch(
        "gran_sabio._should_use_gransabio_tools",
        return_value=True,
    ):
        payload, raw = await _invoke_review(
            engine,
            _request(),
        )

    assert payload == {"ok": True}
    assert raw == '{"ok":true}'
    assert timeouts == pytest.approx([10.0, 6.0])


@pytest.mark.asyncio
async def test_review_stream_truncation_retry_uses_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [400.0]
    timeouts: list[float] = []
    budgets: list[int] = []

    class StreamingService:
        def generate_content_stream(self, **kwargs: object):
            timeouts.append(float(kwargs["request_timeout"]))
            budgets.append(int(kwargs["max_tokens"]))

            async def chunks():
                if len(timeouts) == 1:
                    clock[0] += 3.0
                    yield StreamChunk("partial")
                    yield StreamChunk(
                        "",
                        metadata={
                            "provider": "anthropic",
                            "finish_reason": "max_tokens",
                            "provider_stop_reason": "max_tokens",
                            "output_truncated": True,
                            "max_tokens": 128,
                        },
                    )
                    return
                yield StreamChunk('{"ok":true}')
                yield StreamChunk(
                    "",
                    metadata={
                        "provider": "anthropic",
                        "finish_reason": "end_turn",
                        "provider_stop_reason": "end_turn",
                        "output_truncated": False,
                        "max_tokens": 256,
                    },
                )

            return chunks()

    engine = _engine(StreamingService())
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )

    with (
        patch("gran_sabio._should_use_gransabio_tools", return_value=False),
        patch.object(
            type(config),
            "resolve_bounded_output_truncation_retry",
            return_value=_retry_plan(),
        ),
    ):
        payload, raw = await _invoke_review(
            engine,
            _request(tools_mode="never"),
            stream_callback=AsyncMock(),
        )

    assert payload == {"ok": True}
    assert raw == '{"ok":true}'
    assert budgets == [128, 256]
    assert timeouts == pytest.approx([10.0, 7.0])


@pytest.mark.asyncio
async def test_regeneration_does_not_dispatch_retry_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [500.0]

    async def call_tool_loop(**_kwargs: object) -> tuple[str, ToolLoopEnvelope]:
        clock[0] += 10.0
        return "partial", _truncated_envelope()

    service = MagicMock()
    service.call_ai_with_validation_tools = AsyncMock(side_effect=call_tool_loop)
    service.generate_content = AsyncMock()
    engine = _engine(service)
    monkeypatch.setattr(
        gran_sabio_module,
        "_gran_sabio_monotonic",
        lambda: clock[0],
    )

    with (
        patch("gran_sabio._should_use_gransabio_tools", return_value=True),
        patch.object(
            type(config),
            "resolve_bounded_output_truncation_retry",
            return_value=_retry_plan(),
        ),
    ):
        result = await engine.regenerate_content(
            session_id="regeneration-expired-deadline",
            original_request=_request(),
        )

    assert result.approved is False
    assert result.error is not None
    assert "process deadline" in result.error
    assert service.call_ai_with_validation_tools.await_count == 1
    service.generate_content.assert_not_awaited()
