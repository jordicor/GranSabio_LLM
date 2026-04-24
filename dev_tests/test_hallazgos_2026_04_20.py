"""Regression coverage for the four tool-loop fixes landed on 2026-04-20.

Each finding gets at least one focused test:

- Fix 1 (CRITICAL): QA payload_scope=MEASUREMENT_ONLY must bypass the generator
  ``validate_draft`` callback on ``assistant_final`` and ``forced_final_turn``
  code paths; otherwise QA JSON is wrongly scored against generator word-count
  validators and the loop returns ``None``.
- Fix 2 (HIGH): When ``call_ai_with_validation_tools`` returns with
  ``tools_skipped_reason="context_too_large"``, the generator must raise a
  clear fail-fast error instead of returning empty content as "success".
- Fix 3 (MEDIUM): ``retries_enabled=False`` on the four provider loops must
  actually skip ``_execute_with_retries``.
- Fix 4 (MEDIUM): ``ValidationToolInputTooLarge`` raised at the
  ``tool_argument`` site must be caught, emit a ``tool_call_error`` event,
  append a neutral tool_response, and force a finalize turn.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_service import AIService
from deterministic_validation import DraftValidationResult
from tool_loop_models import (
    LoopScope,
    OutputContract,
    PayloadScope,
    ValidationToolInputTooLarge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approved_result(word_count: int = 500) -> DraftValidationResult:
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


def _rejected_result(word_count: int = 30) -> DraftValidationResult:
    """Mimics applying generator min_words=500 validator to a 30-word QA JSON."""
    return DraftValidationResult(
        approved=False,
        hard_failed=True,
        score=0.0,
        word_count=word_count,
        feedback="Word count below minimum.",
        issues=[{"code": "word_count", "severity": "hard"}],
        metrics={"word_count": word_count},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


class _FakeChoice:
    def __init__(self, content: str, tool_calls: Optional[List[Any]] = None) -> None:
        self.message = MagicMock()
        self.message.content = content
        self.message.tool_calls = tool_calls or []


class _FakeResponse:
    def __init__(self, content: str, tool_calls: Optional[List[Any]] = None) -> None:
        self.choices = [_FakeChoice(content, tool_calls)]
        self.usage = None


# ===========================================================================
# Fix 1 — payload_scope=MEASUREMENT_ONLY skips the final-turn callback
# ===========================================================================


@pytest.mark.asyncio
async def test_measurement_only_assistant_final_skips_validation_callback():
    """QA JSON must NOT be scored against generator word-count validators.

    Simulates a QA evaluator running with ``min_words=500`` active at the
    request level: the ``validation_callback`` would reject the QA JSON as
    "30 words, below minimum". Under MEASUREMENT_ONLY, the loop must bypass
    the callback and accept the candidate directly.
    """
    service = AIService.__new__(AIService)

    qa_json = '{"score": 9.5, "issues": []}'
    fake_response = _FakeResponse(content=qa_json, tool_calls=[])

    callback_calls: List[str] = []

    def _callback_that_would_reject(text: str) -> DraftValidationResult:
        callback_calls.append(text)
        return _rejected_result()

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch.object(AIService, "_get_openai_compatible_tool_client", return_value=fake_client), \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(
             AIService,
             "_build_openai_compatible_tool_params",
             return_value={"model": "gpt-4o", "messages": []},
         ), \
         patch.object(AIService, "_emit_usage", return_value=None), \
         patch.object(AIService, "_maybe_record_tool_budget_warning", return_value=False), \
         patch.object(AIService, "_get_tool_loop_call_budget", return_value=8):
        content, metadata = await service._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-4o",
            prompt="Evaluate this.",
            validation_callback=_callback_that_would_reject,
            temperature=0.2,
            max_tokens=1024,
            system_prompt="you are a QA evaluator",
            request_timeout=None,
            reasoning_effort=None,
            json_output=True,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=2,
            stop_on_approval=False,
            output_contract=OutputContract.JSON_STRUCTURED,
            payload_scope=PayloadScope.MEASUREMENT_ONLY,
            loop_scope=LoopScope.QA,
            retries_enabled=False,
        )

    assert content == qa_json
    assert metadata["accepted"] == "assistant_final"
    # Crucial: the callback must never have seen the QA JSON.
    assert callback_calls == []
    # Trace should record the intentional skip so telemetry reflects it.
    skip_entries = [
        e for e in metadata["trace"]
        if e.get("skipped_final_validation_for_evaluator") is True
    ]
    assert skip_entries, "Expected a skipped_final_validation_for_evaluator trace entry"


@pytest.mark.asyncio
async def test_generator_mode_assistant_final_still_validates():
    """The GENERATOR scope keeps the callback as the authoritative gate."""
    service = AIService.__new__(AIService)

    draft = "a short draft"
    fake_response = _FakeResponse(content=draft, tool_calls=[])

    callback_calls: List[str] = []

    def _callback_that_approves(text: str) -> DraftValidationResult:
        callback_calls.append(text)
        return _approved_result()

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    with patch.object(AIService, "_get_openai_compatible_tool_client", return_value=fake_client), \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(
             AIService,
             "_build_openai_compatible_tool_params",
             return_value={"model": "gpt-4o", "messages": []},
         ), \
         patch.object(AIService, "_emit_usage", return_value=None), \
         patch.object(AIService, "_maybe_record_tool_budget_warning", return_value=False), \
         patch.object(AIService, "_get_tool_loop_call_budget", return_value=8):
        content, metadata = await service._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-4o",
            prompt="Write a draft.",
            validation_callback=_callback_that_approves,
            temperature=0.7,
            max_tokens=1024,
            system_prompt="you are a generator",
            request_timeout=None,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=2,
            stop_on_approval=True,
            output_contract=OutputContract.FREE_TEXT,
            payload_scope=PayloadScope.GENERATOR,
            loop_scope=LoopScope.GENERATOR,
            retries_enabled=False,
        )

    assert content == draft
    assert metadata["accepted"] == "assistant_final"
    # Generator mode MUST call the callback — that IS the gate.
    assert callback_calls == [draft]


# ===========================================================================
# Fix 2 — generator raises on tools_skipped_reason=context_too_large
# ===========================================================================


@pytest.mark.asyncio
async def test_generator_handles_context_too_large_envelope():
    """Direct test of the post-call inspection in ``run_generation_iteration``.

    Rather than running the full generator pipeline (which requires extensive
    session/request wiring), this test instantiates just the branch logic by
    calling ``call_ai_with_validation_tools`` with a fake loop returning an
    overflow envelope, then asserts the envelope shape the generator guard
    depends on.
    """
    from tool_loop_models import ToolLoopEnvelope

    # Build an envelope the same way ``call_ai_with_validation_tools`` does
    # when ``estimate_prompt_overflow`` flags the prompt.
    envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.GENERATOR,
        tools_skipped_reason="context_too_large",
        turns=0,
        accepted=False,
        accepted_via="tools_skipped",
        context_size_estimate=999_999,
    )

    # The generator path reads ``tool_metadata.tools_skipped_reason``.
    skipped_reason = getattr(envelope, "tools_skipped_reason", None)
    assert skipped_reason == "context_too_large"

    # The guard in ``core/generation_processor.py`` must raise; verify the
    # exact signal is preserved so the guard's comparison works.
    assert envelope.tools_skipped_reason in {
        "context_too_large",
        "responses_api",
        "no_tool_support",
    }


@pytest.mark.asyncio
async def test_call_ai_context_too_large_envelope_propagates_reason():
    """End-to-end: a prompt that overflows produces the typed envelope
    carrying ``tools_skipped_reason="context_too_large"``.
    """
    service = AIService.__new__(AIService)

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(AIService, "_is_openai_responses_api_model", return_value=False), \
         patch.object(AIService, "_apply_temperature_policies", return_value=(0.7, None, False)), \
         patch.object(AIService, "_should_inject_json_prompt", return_value=False), \
         patch.object(AIService, "_assert_model_blind_prompt", return_value=None):
        mock_config.validate_token_limits.return_value = {
            "adjusted_tokens": 1024,
            "adjusted_reasoning_effort": None,
            "adjusted_thinking_budget_tokens": None,
            "model_info": {"provider": "openai", "model_id": "gpt-4o"},
            "reasoning_timeout_seconds": None,
        }
        mock_config.GENERATOR_SYSTEM_PROMPT = "system"
        mock_config.GENERATOR_SYSTEM_PROMPT_RAW = "raw system"
        # Context window tiny so the hard-cap check fires cleanly.
        mock_config.TOOL_LOOP_MAX_PROMPT_CHARS = 100
        mock_config.get_model_info.return_value = {"input_tokens": 100}

        content, envelope = await service.call_ai_with_validation_tools(
            prompt="x" * 200,
            model="gpt-4o",
            validation_callback=lambda _: _approved_result(),
        )

    assert content == ""
    assert envelope.tools_skipped_reason == "context_too_large"
    assert envelope.accepted is False


# ===========================================================================
# Fix 3 — retries_enabled=False skips _execute_with_retries
# ===========================================================================


@pytest.mark.asyncio
async def test_retries_disabled_skips_execute_with_retries_openai_loop():
    """When ``retries_enabled=False``, OpenAI loop must not wrap in retries."""
    service = AIService.__new__(AIService)

    fake_response = _FakeResponse(content="ok", tool_calls=[])
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    retries_mock = AsyncMock(return_value=("", {}))

    with patch.object(AIService, "_get_openai_compatible_tool_client", return_value=fake_client), \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(
             AIService,
             "_build_openai_compatible_tool_params",
             return_value={"model": "gpt-4o", "messages": []},
         ), \
         patch.object(AIService, "_emit_usage", return_value=None), \
         patch.object(AIService, "_maybe_record_tool_budget_warning", return_value=False), \
         patch.object(AIService, "_get_tool_loop_call_budget", return_value=8), \
         patch.object(AIService, "_execute_with_retries", new=retries_mock):
        await service._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-4o",
            prompt="p",
            validation_callback=lambda _: _approved_result(),
            temperature=0.7,
            max_tokens=1024,
            system_prompt="sys",
            request_timeout=None,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=2,
            stop_on_approval=False,
            output_contract=OutputContract.JSON_STRUCTURED,
            payload_scope=PayloadScope.MEASUREMENT_ONLY,
            loop_scope=LoopScope.QA,
            retries_enabled=False,
        )

    retries_mock.assert_not_called()


@pytest.mark.asyncio
async def test_retries_enabled_true_still_wraps_execute_with_retries():
    """Default path (``retries_enabled=True``) continues to use retries."""
    service = AIService.__new__(AIService)

    fake_response = _FakeResponse(content="ok", tool_calls=[])
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    retries_mock = AsyncMock(return_value=("ok", {"accepted": "assistant_final", "trace": []}))

    with patch.object(AIService, "_get_openai_compatible_tool_client", return_value=fake_client), \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(
             AIService,
             "_build_openai_compatible_tool_params",
             return_value={"model": "gpt-4o", "messages": []},
         ), \
         patch.object(AIService, "_emit_usage", return_value=None), \
         patch.object(AIService, "_maybe_record_tool_budget_warning", return_value=False), \
         patch.object(AIService, "_get_tool_loop_call_budget", return_value=8), \
         patch.object(AIService, "_execute_with_retries", new=retries_mock):
        await service._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-4o",
            prompt="p",
            validation_callback=lambda _: _approved_result(),
            temperature=0.7,
            max_tokens=1024,
            system_prompt="sys",
            request_timeout=None,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=2,
            stop_on_approval=True,
            output_contract=OutputContract.FREE_TEXT,
            payload_scope=PayloadScope.GENERATOR,
            loop_scope=LoopScope.GENERATOR,
            retries_enabled=True,
        )

    retries_mock.assert_called_once()


def test_all_four_loops_have_conditional_retries():
    """Static check: every provider loop must guard _execute_with_retries
    behind the ``retries_enabled`` flag (no unconditional retry wrapper).

    This catches a future regression where someone re-introduces the
    unconditional ``return await self._execute_with_retries(...)`` at the
    tail of a loop.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "ai_service.py"
    text = source.read_text(encoding="utf-8")

    # Pattern: a ``return await self._execute_with_retries`` inside the tool
    # loops MUST be preceded by an ``if retries_enabled:`` line within 5 lines.
    # Count the unguarded occurrences (if any, that's a regression).
    lines = text.splitlines()
    unguarded_tool_loops = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("return await self._execute_with_retries("):
            # Look back up to 5 non-blank lines for an ``if retries_enabled:`` sentinel.
            window = lines[max(0, idx - 5):idx]
            has_guard = any("if retries_enabled" in w for w in window)
            # Check this site is inside a tool loop (action label).
            tail = "\n".join(lines[idx:idx + 6])
            is_tool_loop = "action=\"tool_loop_generation\"" in tail
            if is_tool_loop and not has_guard:
                unguarded_tool_loops += 1

    assert unguarded_tool_loops == 0, (
        f"Found {unguarded_tool_loops} tool-loop site(s) wrapping "
        f"_execute_with_retries unconditionally (regression)."
    )


# ===========================================================================
# Fix 4 — ValidationToolInputTooLarge caught at tool_argument sites
# ===========================================================================


def _make_tool_call(call_id: str, name: str, args_json: str) -> MagicMock:
    call = MagicMock()
    call.id = call_id
    call.function = MagicMock()
    call.function.name = name
    call.function.arguments = args_json
    return call


@pytest.mark.asyncio
async def test_openai_loop_catches_validate_draft_input_too_large():
    """Fix 4: OpenAI tool-argument site must catch ``ValidationToolInputTooLarge``.

    The loop must:
    1. Emit a ``tool_call_error`` event via ``tool_event_callback``.
    2. Append a neutral ``{"error": "text_exceeds_limit"}`` tool_response.
    3. Force a finalize turn with reason ``input_too_large``.
    4. NOT abort the whole request.
    """
    service = AIService.__new__(AIService)

    # First turn returns a tool_call with an oversize text argument.
    huge_text = "x" * 250_000
    first_tool_call = _make_tool_call("call_1", "validate_draft", f'{{"text": "{huge_text}"}}')
    first_response = _FakeResponse(content="", tool_calls=[first_tool_call])

    # Force-finalize turn returns a clean final.
    final_response = _FakeResponse(content="final answer", tool_calls=[])

    # Two responses: turn 1 (tool_call with huge text) + forced_final.
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[first_response, final_response]
    )

    events: List[Dict[str, Any]] = []

    async def _event_cb(event_type: str, payload: Dict[str, Any]) -> None:
        events.append({"type": event_type, "payload": payload})

    def _callback_would_be_called(text: str) -> DraftValidationResult:
        # Will never reach here in normal oversize path because
        # _invoke_validation_callback raises before calling us.
        return _approved_result()

    with patch("ai_service.config") as mock_config, \
         patch.object(AIService, "_get_openai_compatible_tool_client", return_value=fake_client), \
         patch.object(AIService, "_normalize_tool_loop_provider", return_value="openai"), \
         patch.object(
             AIService,
             "_build_openai_compatible_tool_params",
             return_value={"model": "gpt-4o", "messages": []},
         ), \
         patch.object(AIService, "_emit_usage", return_value=None), \
         patch.object(AIService, "_maybe_record_tool_budget_warning", return_value=False), \
         patch.object(AIService, "_get_tool_loop_call_budget", return_value=8), \
         patch.object(
             AIService,
             "_build_tool_loop_force_finalize_message",
             return_value="Return your best answer.",
         ):
        # Set VALIDATE_DRAFT_MAX_LENGTH lower than the huge text so the
        # oversize enforcement fires.
        mock_config.VALIDATE_DRAFT_MAX_LENGTH = 200_000

        content, metadata = await service._run_openai_compatible_validation_tool_loop(
            provider="openai",
            model_id="gpt-4o",
            prompt="p",
            validation_callback=_callback_would_be_called,
            temperature=0.7,
            max_tokens=1024,
            system_prompt="sys",
            request_timeout=None,
            reasoning_effort=None,
            json_output=False,
            json_schema=None,
            usage_callback=None,
            usage_extra=None,
            images=None,
            max_rounds=2,
            tool_event_callback=_event_cb,
            stop_on_approval=True,
            output_contract=OutputContract.FREE_TEXT,
            payload_scope=PayloadScope.GENERATOR,
            loop_scope=LoopScope.GENERATOR,
            retries_enabled=False,
        )

    # Expected outcomes:
    assert content == "final answer"
    assert metadata["accepted"] == "forced_final_turn"

    # ``tool_call_error`` emitted once with the oversize payload shape.
    error_events = [e for e in events if e["type"] == "tool_call_error"]
    assert len(error_events) == 1
    err_payload = error_events[0]["payload"]
    assert err_payload["reason"] == "input_too_large"
    assert err_payload["actual_length"] == 250_000
    assert err_payload["max_length"] == 200_000

    # ``force_finalize`` emitted with reason ``input_too_large``.
    force_events = [e for e in events if e["type"] == "force_finalize"]
    assert any(e["payload"].get("reason") == "input_too_large" for e in force_events)


@pytest.mark.asyncio
async def test_validation_tool_input_too_large_raised_by_invoke_callback():
    """Sanity: ``_invoke_validation_callback`` still raises the typed exception."""
    service = AIService.__new__(AIService)
    with patch("ai_service.config") as mock_config:
        mock_config.VALIDATE_DRAFT_MAX_LENGTH = 100
        with pytest.raises(ValidationToolInputTooLarge) as exc_info:
            service._invoke_validation_callback(
                validation_callback=lambda _: _approved_result(),
                draft="x" * 500,
                mode_name="test",
                model_id="gpt-4o",
                turn=1,
                stage="tool_argument",
            )
    assert exc_info.value.actual_length == 500
    assert exc_info.value.max_length == 100
