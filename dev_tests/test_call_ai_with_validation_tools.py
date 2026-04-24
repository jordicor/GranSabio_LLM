"""Phase 1 regression coverage for ``call_ai_with_validation_tools``.

Covers the minimum set of scenarios required by the Phase 1 refactor:

- Module-level shortcut is importable.
- Entry point honours the typed envelope return contract.
- Fallback envelopes surface ``tools_skipped_reason`` cleanly for
  Responses API models, unsupported providers, and ``context_too_large``.
- Generator-path integration with ``DraftValidationResult`` succeeds when
  a tool call returns approved on the first pass.
- Initial measurement injection fires when ``initial_measurement_text``
  is provided.
- Oversize ``validate_draft`` input raises ``ValidationToolInputTooLarge``
  from ``_invoke_validation_callback``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_service import (
    AIRequestError,
    AIService,
    StreamChunk,
    call_ai_with_validation_tools,
    estimate_prompt_overflow,
)
from deterministic_validation import DraftValidationResult
from tool_loop_models import (
    JsonContractError,
    LoopScope,
    OutputContract,
    PayloadScope,
    ToolLoopContractError,
    ToolLoopEnvelope,
    ToolLoopSchemaViolationError,
    ValidationToolInputTooLarge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approved_result(word_count: int = 42) -> DraftValidationResult:
    return DraftValidationResult(
        approved=True,
        hard_failed=False,
        score=10.0,
        word_count=word_count,
        feedback="All deterministic checks passed.",
        issues=[],
        metrics={"word_count": word_count},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


def _rejected_result() -> DraftValidationResult:
    return DraftValidationResult(
        approved=False,
        hard_failed=True,
        score=0.0,
        word_count=0,
        feedback="The draft failed deterministic validation.",
        issues=[{"code": "word_count", "severity": "hard"}],
        metrics={"word_count": 0},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


# ---------------------------------------------------------------------------
# Import / module-level wiring
# ---------------------------------------------------------------------------


def test_module_level_shortcut_is_coroutine():
    """The module-level ``call_ai_with_validation_tools`` must be a coroutine."""
    import inspect

    assert inspect.iscoroutinefunction(call_ai_with_validation_tools)


def test_aiservice_exposes_call_ai_method():
    """The AIService method surface keeps the same name."""
    assert hasattr(AIService, "call_ai_with_validation_tools")
    assert not hasattr(AIService, "generate_content_with_validation_tools")


# ---------------------------------------------------------------------------
# Envelope-returning fallbacks (Responses API / no tool support / context overflow)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_responses_api_envelope():
    """OpenAI Responses API models fall back to single-shot with a typed envelope."""
    service = AIService.__new__(AIService)
    with patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=True), \
         patch("ai_service.config") as mock_config:
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 512,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "o3-pro"},
            "reasoning_timeout_seconds": None,
        }
        content, envelope = await service.call_ai_with_validation_tools(
            prompt="p",
            model="o3-pro",
            validation_callback=lambda _: _approved_result(),
        )
    assert content == ""
    assert isinstance(envelope, ToolLoopEnvelope)
    assert envelope.tools_skipped_reason == "responses_api"
    assert envelope.accepted is False


@pytest.mark.asyncio
async def test_fallback_no_tool_support_envelope():
    """Unknown provider surfaces ``no_tool_support`` rather than raising."""
    service = AIService.__new__(AIService)
    with patch.object(AIService, "_normalize_tool_loop_provider", return_value="unknown"), \
         patch("ai_service.config") as mock_config:
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 512,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "unknown", "model_id": "mystery-model"},
            "reasoning_timeout_seconds": None,
        }
        content, envelope = await service.call_ai_with_validation_tools(
            prompt="p",
            model="mystery-model",
            validation_callback=lambda _: _approved_result(),
        )
    assert content == ""
    assert envelope.tools_skipped_reason == "no_tool_support"


# ---------------------------------------------------------------------------
# estimate_prompt_overflow helper
# ---------------------------------------------------------------------------


def test_estimate_prompt_overflow_unknown_model_returns_none():
    """Unknown models degrade gracefully (no context-window info available)."""
    with patch("ai_service.config") as mock_config:
        mock_config.get_model_info.side_effect = RuntimeError("unknown model")
        assert estimate_prompt_overflow("fake-model", 100_000, 4000) is None


def test_estimate_prompt_overflow_fits_returns_none():
    """When estimated tokens fit comfortably, return ``None``."""
    with patch("ai_service.config") as mock_config:
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}
        # 4000 chars ~= 1000 tokens, well below 128k.
        assert estimate_prompt_overflow("gpt-4o", 4000, 2000) is None


def test_estimate_prompt_overflow_exceeds_returns_flag():
    """Overflow produces the tagged string."""
    with patch("ai_service.config") as mock_config:
        mock_config.get_model_info.return_value = {"input_tokens": 1000}
        # 100_000 chars ~= 25_000 tokens, far above 1000.
        assert estimate_prompt_overflow("tiny-model", 100_000, 100) == "context_too_large"


def test_normalize_usage_marks_length_finish_reason_as_truncated():
    """OpenAI-style ``length`` finish reason must be surfaced as output truncation."""
    service = AIService.__new__(AIService)
    usage = service._normalize_usage({
        "prompt_tokens": 10,
        "completion_tokens": 4000,
        "finish_reason": "length",
    })

    assert usage["output_truncated"] is True
    assert usage["truncation_reason"] == "output_token_limit"
    assert usage["provider_stop_reason"] == "length"


def test_usage_with_finish_metadata_marks_claude_max_tokens_as_truncated():
    """Claude ``stop_reason=max_tokens`` means the text may be incomplete."""
    service = AIService.__new__(AIService)
    response = SimpleNamespace(stop_reason="max_tokens")
    usage_obj = SimpleNamespace(input_tokens=100, output_tokens=4000)

    usage = service._usage_with_finish_metadata(
        usage_obj,
        response,
        provider="claude",
        max_tokens=4000,
    )

    assert usage["provider"] == "claude"
    assert usage["provider_stop_reason"] == "max_tokens"
    assert usage["output_truncated"] is True
    assert usage["truncation_reason"] == "output_token_limit"
    assert usage["max_tokens"] == 4000


def test_stream_chunk_carries_finish_metadata():
    """Final streaming control chunks can carry provider finish metadata."""
    chunk = StreamChunk("", metadata={"output_truncated": True})

    assert chunk.text == ""
    assert chunk.is_thinking is False
    assert chunk.metadata["output_truncated"] is True


# ---------------------------------------------------------------------------
# DraftValidationResult gate / visible_payload semantics
# ---------------------------------------------------------------------------


def test_draft_validation_result_generator_visible_payload_has_gate_fields():
    """Generator scope exposes ``approved`` / ``hard_failed`` to the LLM."""
    result = _approved_result()
    payload = result.build_visible_payload(PayloadScope.GENERATOR)
    assert payload["approved"] is True
    assert payload["hard_failed"] is False
    assert payload["feedback"] == "All deterministic checks passed."


def test_draft_validation_result_measurement_only_strips_gate_fields():
    """Measurement-only scope MUST NOT leak gate verdicts to evaluators."""
    result = _rejected_result()
    payload = result.build_visible_payload(PayloadScope.MEASUREMENT_ONLY)
    assert "approved" not in payload
    assert "hard_failed" not in payload
    assert payload["feedback_neutral"] == "The draft failed deterministic validation."


# ---------------------------------------------------------------------------
# _invoke_validation_callback — oversize enforcement + type check
# ---------------------------------------------------------------------------


def test_invoke_validation_callback_raises_on_oversize_input():
    """Defense-in-depth enforcement of ``VALIDATE_DRAFT_MAX_LENGTH``."""
    service = AIService.__new__(AIService)
    huge_text = "x" * 250_000
    with patch("ai_service.config") as mock_config:
        mock_config.VALIDATE_DRAFT_MAX_LENGTH = 200_000
        with pytest.raises(ValidationToolInputTooLarge) as exc_info:
            service._invoke_validation_callback(
                validation_callback=lambda _: _approved_result(),
                draft=huge_text,
                mode_name="test",
                model_id="gpt-4o",
                turn=1,
                stage="tool_argument",
            )
    assert exc_info.value.actual_length == 250_000
    assert exc_info.value.max_length == 200_000


def test_invoke_validation_callback_rejects_non_draftvalidationresult():
    """Legacy dict-returning callbacks must be rejected fail-fast."""
    service = AIService.__new__(AIService)
    with patch("ai_service.config") as mock_config:
        mock_config.VALIDATE_DRAFT_MAX_LENGTH = 200_000
        with pytest.raises(ValueError, match="expected DraftValidationResult"):
            service._invoke_validation_callback(
                validation_callback=lambda _: {"approved": True},
                draft="hello",
                mode_name="test",
                model_id="gpt-4o",
                turn=1,
                stage="tool_argument",
            )


def test_invoke_validation_callback_returns_result_within_limit():
    """Happy path returns the ``DraftValidationResult`` unchanged."""
    service = AIService.__new__(AIService)
    expected = _approved_result(word_count=123)
    with patch("ai_service.config") as mock_config:
        mock_config.VALIDATE_DRAFT_MAX_LENGTH = 200_000
        result = service._invoke_validation_callback(
            validation_callback=lambda _: expected,
            draft="hello",
            mode_name="test",
            model_id="gpt-4o",
            turn=1,
            stage="tool_argument",
        )
    assert result is expected
    assert result.word_count == 123


def test_claude_text_extractor_ignores_tool_use_blocks_without_text():
    """A bare Claude tool_use block is not model output and must not become JSON."""
    service = AIService.__new__(AIService)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu_123",
                name="validate_draft",
                input={},
            )
        ]
    )

    assert service._extract_text_from_claude_response(response) == ""


def test_claude_text_extractor_keeps_visible_text_and_skips_tool_use_blocks():
    service = AIService.__new__(AIService)
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text="internal"),
            SimpleNamespace(type="text", text='{"ok": true}'),
            SimpleNamespace(type="tool_use", name="validate_draft", input={}),
        ]
    )

    assert service._extract_text_from_claude_response(response) == '{"ok": true}'


# ---------------------------------------------------------------------------
# Envelope build — trace preserved, payload parsed for JSON_STRUCTURED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_ai_builds_envelope_from_loop_metadata():
    """The new entry point wraps per-loop metadata in a typed ``ToolLoopEnvelope``."""
    service = AIService.__new__(AIService)
    fake_content = "final draft"
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 2,
        "accepted": "assistant_final",
        "trace": [
            {"turn": 1, "tool": "validate_draft", "approved": True, "score": 10.0},
            {"turn": 2, "event": "assistant_final", "approved": True, "score": 10.0},
        ],
        "accent_fail_open_delta_count": 0,
        "accent_fail_open_delta_paths": [],
    }

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=False), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(AIService, "_run_openai_compatible_validation_tool_loop", new=AsyncMock(return_value=(fake_content, fake_metadata))):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 1024,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write something.",
            model="gpt-4o",
            validation_callback=lambda _: _approved_result(),
        )

    assert content == "final draft"
    assert isinstance(envelope, ToolLoopEnvelope)
    assert envelope.turns == 2
    assert envelope.accepted is True
    assert envelope.accepted_via == "assistant_final"
    assert envelope.loop_scope == LoopScope.GENERATOR
    assert len(envelope.trace) == 2
    assert envelope.trace[0].tool == "validate_draft"
    assert envelope.payload is None  # FREE_TEXT contract


# ---------------------------------------------------------------------------
# JSON_STRUCTURED contract hardening (findings #4, #5, #6 from review)
# ---------------------------------------------------------------------------


_SIMPLE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {
        "decision": {"type": "string", "enum": ["APPLY", "DISCARD"]},
    },
}


def _accepted_metadata() -> Dict[str, Any]:
    return {
        "mode": "openai_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }


async def _invoke_call_ai_with_fake_loop(
    *,
    fake_content: str,
    fake_metadata: Dict[str, Any],
    response_format: Any = _SIMPLE_SCHEMA,
    json_expectations: Any = None,
    output_contract: OutputContract = OutputContract.JSON_STRUCTURED,
):
    """Invoke ``call_ai_with_validation_tools`` with a mocked provider loop."""
    service = AIService.__new__(AIService)

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=False), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(
             AIService,
             "_run_openai_compatible_validation_tool_loop",
             new=AsyncMock(return_value=(fake_content, fake_metadata)),
         ):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 1024,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

        return await service.call_ai_with_validation_tools(
            prompt="Write something.",
            model="gpt-4o",
            validation_callback=lambda _: _approved_result(),
            output_contract=output_contract,
            response_format=response_format,
            json_expectations=json_expectations,
        )


@pytest.mark.asyncio
async def test_call_ai_emits_tool_loop_error_event_on_provider_failure():
    """Provider/tool-loop failures are visible through the tool event channel."""
    service = AIService.__new__(AIService)
    events = []

    async def tool_event_callback(event_type: str, payload: Dict[str, Any]) -> None:
        events.append((event_type, payload))

    cause = AttributeError("module aiohttp has no attribute ClientConnectorDNSError")
    provider_error = AIRequestError("openai", "gpt-4o", 1, 1, cause)

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=False), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(
             AIService,
             "_run_openai_compatible_validation_tool_loop",
             new=AsyncMock(side_effect=provider_error),
         ):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 1024,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

        with pytest.raises(AIRequestError):
            await service.call_ai_with_validation_tools(
                prompt="Write something.",
                model="gpt-4o",
                validation_callback=lambda _: _approved_result(),
                tool_event_callback=tool_event_callback,
            )

    assert events
    event_type, payload = events[-1]
    assert event_type == "tool_loop_error"
    assert payload["exception_class"] == "AIRequestError"
    assert payload["cause_class"] == "AttributeError"
    assert "ClientConnectorDNSError" in payload["cause_message"]


@pytest.mark.asyncio
async def test_call_ai_raises_on_tool_loop_exhausted():
    """Finding #6: ``accepted_via='tool_loop_exhausted'`` must fail fast."""
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 5,
        "accepted": "tool_loop_exhausted",
        "trace": [],
    }

    with pytest.raises(ToolLoopSchemaViolationError, match="tool loop exhausted"):
        await _invoke_call_ai_with_fake_loop(
            fake_content="partial content",
            fake_metadata=fake_metadata,
            response_format=_SIMPLE_SCHEMA,
        )


@pytest.mark.asyncio
async def test_call_ai_raises_on_empty_content_under_json_structured():
    """Finding #5: empty content under JSON_STRUCTURED raises JsonContractError."""
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }

    with pytest.raises(JsonContractError, match="empty content"):
        await _invoke_call_ai_with_fake_loop(
            fake_content="",
            fake_metadata=fake_metadata,
            response_format=_SIMPLE_SCHEMA,
        )


@pytest.mark.asyncio
async def test_call_ai_raises_when_json_structured_missing_response_format():
    """Finding #4 defensive guard: JSON_STRUCTURED requires response_format."""
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }

    with pytest.raises(ToolLoopContractError, match="requires"):
        await _invoke_call_ai_with_fake_loop(
            fake_content='{"decision": "APPLY"}',
            fake_metadata=fake_metadata,
            response_format=None,
        )


@pytest.mark.asyncio
async def test_call_ai_accepts_valid_json_loose_object_and_array():
    """JSON_LOOSE accepts top-level objects/arrays and carries no payload."""
    for content in ('{"decision": "APPLY"}', '[{"decision": "APPLY"}]'):
        returned_content, envelope = await _invoke_call_ai_with_fake_loop(
            fake_content=content,
            fake_metadata=_accepted_metadata(),
            output_contract=OutputContract.JSON_LOOSE,
            response_format=None,
        )

        assert returned_content == content
        assert envelope.output_schema_valid is True
        assert envelope.payload is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('text before {"decision": "APPLY"}', '{"decision":"APPLY"}'),
        ('```json\n{"decision": "APPLY"}\n```', '{"decision":"APPLY"}'),
        ('{"decision": "APPLY",}', '{"decision":"APPLY"}'),
        ("{decision: 'APPLY'}", '{"decision":"APPLY"}'),
    ],
)
async def test_call_ai_accepts_and_normalizes_ai_json_loose_wrappers(content: str, expected: str):
    """JSON_LOOSE extracts/repairs common AI JSON wrappers."""
    returned_content, envelope = await _invoke_call_ai_with_fake_loop(
        fake_content=content,
        fake_metadata=_accepted_metadata(),
        output_contract=OutputContract.JSON_LOOSE,
        response_format=None,
    )

    assert returned_content == expected
    assert envelope.output_schema_valid is True
    assert envelope.payload is None


@pytest.mark.asyncio
async def test_call_ai_accepts_and_normalizes_curly_quote_delimiters_under_json_loose():
    """JSON_LOOSE treats curly quote delimiters as a recoverable AI slip."""
    curly_open = chr(0x201C)
    curly_close = chr(0x201D)
    content = '{"decision": ' + curly_open + "APPLY" + curly_close + "}"

    returned_content, envelope = await _invoke_call_ai_with_fake_loop(
        fake_content=content,
        fake_metadata=_accepted_metadata(),
        output_contract=OutputContract.JSON_LOOSE,
        response_format=None,
    )

    assert returned_content == '{"decision":"APPLY"}'
    assert envelope.output_schema_valid is True
    assert envelope.payload is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", '"APPLY"', "42", '{"decision": '])
async def test_call_ai_rejects_missing_truncated_or_scalar_json_loose_payloads(content: str):
    """JSON_LOOSE still rejects real failures."""
    with pytest.raises(JsonContractError):
        await _invoke_call_ai_with_fake_loop(
            fake_content=content,
            fake_metadata=_accepted_metadata(),
            output_contract=OutputContract.JSON_LOOSE,
            response_format=None,
        )


@pytest.mark.asyncio
async def test_call_ai_rejects_empty_content_under_json_loose():
    with pytest.raises(JsonContractError, match="empty content"):
        await _invoke_call_ai_with_fake_loop(
            fake_content="",
            fake_metadata=_accepted_metadata(),
            output_contract=OutputContract.JSON_LOOSE,
            response_format=None,
        )


@pytest.mark.asyncio
async def test_call_ai_forwards_expectations_for_json_loose():
    with pytest.raises(JsonContractError):
        await _invoke_call_ai_with_fake_loop(
            fake_content='{"decision": "APPLY"}',
            fake_metadata=_accepted_metadata(),
            output_contract=OutputContract.JSON_LOOSE,
            response_format=None,
            json_expectations=[{"path": "decision", "equals": "DISCARD"}],
        )


@pytest.mark.asyncio
async def test_call_ai_forwards_expectations_for_json_structured():
    with pytest.raises(ToolLoopSchemaViolationError):
        await _invoke_call_ai_with_fake_loop(
            fake_content='{"decision": "APPLY"}',
            fake_metadata=_accepted_metadata(),
            response_format=_SIMPLE_SCHEMA,
            json_expectations=[{"path": "decision", "equals": "DISCARD"}],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_contract", "response_format", "match"),
    [
        (OutputContract.JSON_LOOSE, _SIMPLE_SCHEMA, "forbids"),
        (OutputContract.FREE_TEXT, _SIMPLE_SCHEMA, "forbids"),
        (OutputContract.JSON_STRUCTURED, {}, "non-empty dict"),
        (OutputContract.JSON_STRUCTURED, [], "non-empty dict"),
        (OutputContract.JSON_STRUCTURED, {"type": "array"}, "top-level type"),
    ],
)
async def test_call_ai_rejects_invalid_output_contract_arguments(
    output_contract: OutputContract,
    response_format: Any,
    match: str,
):
    with pytest.raises(ToolLoopContractError, match=match):
        await _invoke_call_ai_with_fake_loop(
            fake_content='{"decision": "APPLY"}',
            fake_metadata=_accepted_metadata(),
            output_contract=output_contract,
            response_format=response_format,
        )


@pytest.mark.asyncio
async def test_call_ai_routes_json_loose_to_provider_without_schema():
    """Provider loops get json_output=True and json_schema=None for JSON_LOOSE."""
    service = AIService.__new__(AIService)
    loop_mock = AsyncMock(return_value=('{"decision": "APPLY"}', _accepted_metadata()))

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=False), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None), \
         patch.object(
             AIService,
             "_run_openai_compatible_validation_tool_loop",
             new=loop_mock,
         ):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 1024,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 200_000
        mock_config.get_model_info.return_value = {"input_tokens": 128_000}

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="Write JSON.",
            model="gpt-4o",
            validation_callback=lambda _: _approved_result(),
            output_contract=OutputContract.JSON_LOOSE,
            response_format=None,
        )

    assert content == '{"decision": "APPLY"}'
    assert envelope.payload is None
    kwargs = loop_mock.await_args.kwargs
    assert kwargs["json_output"] is True
    assert kwargs["json_schema"] is None
    assert kwargs["output_contract"] == OutputContract.JSON_LOOSE


@pytest.mark.asyncio
async def test_call_ai_raises_on_schema_violation():
    """Finding #4: JSON that parses but violates the declared schema fails."""
    # ``decision`` is missing the required ``APPLY|DISCARD`` enum value.
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }

    with pytest.raises(ToolLoopSchemaViolationError, match="schema validation"):
        await _invoke_call_ai_with_fake_loop(
            fake_content='{"decision": "MAYBE"}',
            fake_metadata=fake_metadata,
            response_format=_SIMPLE_SCHEMA,
        )


# ---------------------------------------------------------------------------
# publish_project_phase_chunk ``data=`` kwarg (finding #1)
# ---------------------------------------------------------------------------


def test_project_phase_event_includes_small_data_payload():
    """Finding #1: small ``data`` payloads are embedded verbatim."""
    import json_utils

    from core.app_state import _build_project_phase_event

    payload = {"event": "validate_draft", "turn": 2, "approved": True}
    event = _build_project_phase_event(
        event="tool_event",
        phase="qa",
        data=payload,
    )
    parsed = json_utils.loads(event)
    assert parsed["data"] == payload


def test_project_phase_event_truncates_oversized_data_payload():
    """Finding #1: oversized ``data`` payloads are replaced by a preview."""
    import json_utils

    from core.app_state import _build_project_phase_event

    # Build a payload that exceeds the 8KB guard.
    oversized = {f"key_{i}": "x" * 200 for i in range(80)}
    event = _build_project_phase_event(
        event="tool_event",
        phase="qa",
        data=oversized,
    )
    parsed = json_utils.loads(event)
    assert parsed["data"]["truncated"] is True
    assert parsed["data"]["size_bytes"] > 8192
    assert len(parsed["data"]["preview_keys"]) == 10


def test_project_phase_event_omits_data_when_none():
    """``data=None`` (the default) must not appear in the event JSON."""
    import json_utils

    from core.app_state import _build_project_phase_event

    event = _build_project_phase_event(event="tool_event", phase="qa")
    parsed = json_utils.loads(event)
    assert "data" not in parsed


@pytest.mark.asyncio
async def test_call_ai_accepts_valid_json_structured_payload():
    """JSON_STRUCTURED happy path parses AND validates against the schema."""
    fake_metadata = {
        "mode": "openai_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }

    content, envelope = await _invoke_call_ai_with_fake_loop(
        fake_content='{"decision": "APPLY"}',
        fake_metadata=fake_metadata,
        response_format=_SIMPLE_SCHEMA,
    )

    assert content == '{"decision": "APPLY"}'
    assert isinstance(envelope, ToolLoopEnvelope)
    assert envelope.accepted is True
    assert envelope.output_schema_valid is True
    assert envelope.payload == {"decision": "APPLY"}


@pytest.mark.asyncio
async def test_call_ai_accepts_fenced_json_structured_payload():
    """JSON_STRUCTURED accepts model JSON wrapped in markdown fences."""
    fake_metadata = {
        "mode": "gemini_tool_loop",
        "turns": 1,
        "accepted": "assistant_final",
        "trace": [],
    }

    content, envelope = await _invoke_call_ai_with_fake_loop(
        fake_content='```json\n{"decision": "APPLY"}\n```',
        fake_metadata=fake_metadata,
        response_format=_SIMPLE_SCHEMA,
    )

    assert content == '{"decision":"APPLY"}'
    assert envelope.output_schema_valid is True
    assert envelope.payload == {"decision": "APPLY"}
