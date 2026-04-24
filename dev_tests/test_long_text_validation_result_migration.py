"""Phase 1 migration coverage for ``long_text/controller.py._validation_callback``.

The long-text section-draft callback used to return a dict with ``{"valid",
"word_count", "summary_words"}``. Phase 1 migrates it to return a
``DraftValidationResult`` (Option A, full migration, no shim).

This test covers the exact mapping documented in §4.8a:

| Legacy dict key      | DraftValidationResult field |
|----------------------|-----------------------------|
| ``"valid": bool``    | ``approved`` + ``hard_failed = not valid`` |
| ``"word_count"``     | ``word_count`` + ``metrics["word_count"]`` |
| ``"summary_words"``  | ``metrics["summary_words"]`` |
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from deterministic_validation import DraftValidationResult
from tool_loop_models import PayloadScope


@pytest.mark.asyncio
async def test_section_tools_validation_callback_returns_draft_validation_result():
    """The rewritten callback must emit a ``DraftValidationResult`` with the
    canonical mapping from the legacy dict keys.
    """
    from long_text.controller import _LongTextController as LongTextController

    # Capture the validation_callback kwarg passed to the AI service.
    captured: Dict[str, Any] = {}

    async def fake_call_ai_with_validation_tools(*args: Any, **kwargs: Any):
        captured["validation_callback"] = kwargs["validation_callback"]
        # Return a payload that parses back as valid section JSON.
        payload = (
            '{"section_text": "Hello world from the section.",'
            ' "summary_anchor": "A brief anchor."}'
        )
        return payload, {"rounds": 1, "accepted": "assistant_final"}

    controller = LongTextController.__new__(LongTextController)
    controller.ai_service = MagicMock()
    controller.ai_service.call_ai_with_validation_tools = fake_call_ai_with_validation_tools
    controller.request = MagicMock()
    controller.request.generator_model = "gpt-4o"
    controller.request.temperature = 0.7
    controller.request.system_prompt = "system"
    controller.request.extra_verbose = False
    controller.request.content_type = "article"
    controller.request._model_alias_registry = None
    controller.phase_logger = None
    controller.used_tool_loop = False
    controller._reserve_generator_call = MagicMock(return_value=None)
    controller._usage_callback = MagicMock(return_value=None)
    controller._ensure_not_cancelled = AsyncMock(return_value=None)
    controller._ensure_wall_clock_budget = MagicMock(return_value=None)

    await controller._run_section_tools(
        prompt="draft section",
        schema={"type": "object"},
        profile="balanced",
        max_tokens=1024,
    )

    validation_callback = captured["validation_callback"]
    assert validation_callback is not None

    # Happy path: both fields populated.
    valid_candidate = (
        '{"section_text": "Hello world from the section.",'
        ' "summary_anchor": "A brief anchor."}'
    )
    good = validation_callback(valid_candidate)
    assert isinstance(good, DraftValidationResult)
    assert good.approved is True
    assert good.hard_failed is False
    assert good.word_count == 5  # "Hello world from the section."
    assert good.metrics["word_count"] == 5
    assert good.metrics["summary_words"] == 3  # "A brief anchor."

    # Missing summary anchor: hard-failed, approved=False.
    missing_anchor = (
        '{"section_text": "Some text here.", "summary_anchor": ""}'
    )
    bad = validation_callback(missing_anchor)
    assert isinstance(bad, DraftValidationResult)
    assert bad.approved is False
    assert bad.hard_failed is True
    assert bad.metrics["summary_words"] == 0
    assert any(issue["code"] == "missing_section_fields" for issue in bad.issues)


def test_draft_validation_result_generator_payload_round_trips_fields():
    """Covers the migration assertion from
    ``dev_tests/test_long_text_generation_pipeline.py:133`` — the callback
    result exposes ``.approved`` (was ``validation_result["valid"]``).
    """
    result = DraftValidationResult(
        approved=True,
        hard_failed=False,
        score=10.0,
        word_count=5,
        feedback="",
        issues=[],
        metrics={"word_count": 5, "summary_words": 3},
        checks={"has_both_fields": True},
        stylistic_metrics=None,
        visible_payload={},
    )
    # Attribute access (new contract) replaces dict subscript (legacy contract).
    assert result.approved is True
    payload = result.build_visible_payload(PayloadScope.GENERATOR)
    assert payload["approved"] is True
    assert payload["word_count"] == 5
    assert payload["metrics"]["summary_words"] == 3
