"""Phase 2 regression coverage for the QA tool-loop adapter.

Covers the scope described in PROPOSAL_TOOLS_FOR_QA_ARBITER_GRANSABIO.md
§3.4.1, §4.3, §4.11 and §4.14:

- ``_should_use_qa_tools`` activation matrix (structured validators present,
  ``qa_tools_mode="never"``, bypassable layer, Responses API models,
  unsupported providers, no validators).
- ``build_measurement_request_for_layer`` whitelist (carries only the
  allowed fields; returns ``None`` when nothing applies).
- Integration: ``QAEvaluationService.evaluate_content`` routes through
  ``call_ai_with_validation_tools`` when eligible, with the correct
  ``loop_scope``, ``payload_scope``, ``retries_enabled=False`` and the
  ``MEASUREMENT_ONLY`` visible payload contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deterministic_validation import DraftValidationResult
from qa_evaluation_service import (
    QAEvaluationService,
    _has_structured_request_validators,
    _should_use_qa_tools,
)
from tool_loop_models import LoopScope, OutputContract, PayloadScope, ToolLoopEnvelope
from validation_context_factory import (
    MeasurementRequest,
    build_measurement_request_for_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer(name: str = "Clarity", criteria: str = "Writing should be clear.") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description="test layer",
        criteria=criteria,
        min_score=7.0,
        deal_breaker_criteria=None,
        concise_on_pass=True,
    )


def _make_request(**overrides: Any) -> SimpleNamespace:
    base: Dict[str, Any] = {
        "qa_tools_mode": "auto",
        "min_words": 500,
        "max_words": 800,
        "phrase_frequency": None,
        "lexical_diversity": None,
        "json_output": False,
        "json_schema": None,
        "json_expectations": None,
        "target_field": None,
        "word_count_enforcement": None,
        "content_type": "biography",
        "prompt": "Write about X.",
        "smart_editing_mode": "never",
        "extra_verbose": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_model_info(provider: str = "openai", model_id: str = "gpt-4o"):
    """Return a patch context for ``config.get_model_info`` used by the helper.

    Also pins ``QA_MAX_TOOL_ROUNDS=3`` (the default from ``config.py``) so the
    integration tests can assert the budget propagation through the adapter
    without relying on module state leaking through the ``Mock`` replacement.
    """
    return patch(
        "qa_evaluation_service.config",
        Mock(
            get_model_info=Mock(return_value={"provider": provider, "model_id": model_id}),
            QA_MAX_TOOL_ROUNDS=3,
            QA_SYSTEM_PROMPT="qa system",
            QA_SYSTEM_PROMPT_RAW="qa system raw",
        ),
    )


def _approved_measurement_result(word_count: int = 600) -> DraftValidationResult:
    return DraftValidationResult(
        approved=True,
        hard_failed=False,
        score=10.0,
        word_count=word_count,
        feedback="All checks passed.",
        issues=[],
        metrics={"word_count": word_count},
        checks={},
        stylistic_metrics=None,
        visible_payload={},
    )


# ---------------------------------------------------------------------------
# _has_structured_request_validators
# ---------------------------------------------------------------------------


def test_structured_validators_detected_by_min_words():
    request = _make_request(min_words=200, max_words=None)
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_max_words():
    request = _make_request(min_words=None, max_words=1000)
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_phrase_frequency():
    request = _make_request(
        min_words=None,
        max_words=None,
        phrase_frequency=SimpleNamespace(enabled=True, rules=[object()]),
    )
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_lexical_diversity():
    request = _make_request(
        min_words=None,
        max_words=None,
        lexical_diversity=SimpleNamespace(enabled=True),
    )
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_json_output():
    request = _make_request(min_words=None, max_words=None, json_output=True)
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_content_type_json_alias():
    request = _make_request(
        min_words=None,
        max_words=None,
        json_output=False,
        content_type="json",
    )
    assert _has_structured_request_validators(request) is True


def test_structured_validators_detected_by_target_field():
    request = _make_request(min_words=None, max_words=None, target_field="generated_text")
    assert _has_structured_request_validators(request) is True


def test_structured_validators_false_when_all_off():
    request = _make_request(
        min_words=None,
        max_words=None,
        phrase_frequency=SimpleNamespace(enabled=False),
        lexical_diversity=SimpleNamespace(enabled=False),
        json_output=False,
        target_field=None,
    )
    assert _has_structured_request_validators(request) is False


def test_structured_validators_false_for_none_request():
    assert _has_structured_request_validators(None) is False


# ---------------------------------------------------------------------------
# _should_use_qa_tools
# ---------------------------------------------------------------------------


def test_should_use_qa_tools_happy_path_with_structured_validators():
    request = _make_request()
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info():
        assert (
            _should_use_qa_tools(
                request, layer, "gpt-4o", bypass_engine=bypass_engine
            )
            is True
        )


def test_should_use_qa_tools_false_when_mode_never():
    request = _make_request(qa_tools_mode="never")
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info():
        assert (
            _should_use_qa_tools(
                request, layer, "gpt-4o", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_false_when_request_is_none():
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info():
        assert (
            _should_use_qa_tools(
                None, layer, "gpt-4o", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_false_when_layer_is_bypassable():
    """Bypass wins over tools (algorithmic always beats LLM tool rounds)."""
    request = _make_request(
        phrase_frequency=SimpleNamespace(enabled=True, rules=[object()]),
    )
    layer = _make_layer(name="Phrase Frequency Guard")
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=True)
    with _patch_model_info():
        assert (
            _should_use_qa_tools(
                request, layer, "gpt-4o", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_false_when_no_structured_validators():
    """Free-text layer criteria must NOT activate tools on their own."""
    request = _make_request(
        min_words=None,
        max_words=None,
        phrase_frequency=None,
        lexical_diversity=None,
        json_output=False,
        target_field=None,
    )
    layer = _make_layer(criteria="Write between 500 and 700 words.")
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info():
        assert (
            _should_use_qa_tools(
                request, layer, "gpt-4o", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_false_for_responses_api_model():
    request = _make_request()
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info(provider="openai", model_id="o3-pro"), \
         patch(
             "qa_evaluation_service.AIService._is_openai_responses_api_model",
             return_value=True,
         ):
        assert (
            _should_use_qa_tools(
                request, layer, "o3-pro", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_false_for_unsupported_provider():
    request = _make_request()
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info(provider="custom", model_id="mystery-model"):
        assert (
            _should_use_qa_tools(
                request, layer, "mystery-model", bypass_engine=bypass_engine
            )
            is False
        )


def test_should_use_qa_tools_true_for_claude():
    request = _make_request()
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with _patch_model_info(provider="claude", model_id="claude-sonnet-4-5"):
        assert (
            _should_use_qa_tools(
                request,
                layer,
                "claude-sonnet-4-5",
                bypass_engine=bypass_engine,
            )
            is True
        )


def test_should_use_qa_tools_false_when_model_info_raises():
    """Unknown models fail-closed (config lookup raising must not activate tools)."""
    request = _make_request()
    layer = _make_layer()
    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)
    with patch(
        "qa_evaluation_service.config",
        Mock(get_model_info=Mock(side_effect=RuntimeError("unknown model"))),
    ):
        assert (
            _should_use_qa_tools(
                request, layer, "fake-model", bypass_engine=bypass_engine
            )
            is False
        )


# ---------------------------------------------------------------------------
# build_measurement_request_for_layer (whitelist fail-closed)
# ---------------------------------------------------------------------------


def test_build_measurement_request_carries_only_whitelisted_fields():
    phrase_frequency = SimpleNamespace(enabled=True, rules=[object()])
    lexical_diversity = SimpleNamespace(enabled=True)
    word_count_enforcement = SimpleNamespace(enabled=True)
    schema = {"type": "object"}
    expectations = [{"path": "status", "equals": "ok"}]

    request = SimpleNamespace(
        min_words=500,
        max_words=800,
        word_count_enforcement=word_count_enforcement,
        phrase_frequency=phrase_frequency,
        lexical_diversity=lexical_diversity,
        json_output=True,
        json_schema=schema,
        json_expectations=expectations,
        target_field="generated_text",
        # Deliberately-forbidden fields:
        cumulative_text="history about X",
        llm_accent_guard=SimpleNamespace(mode="inline"),
        include_stylistic_metrics=True,
        prompt="...",
    )
    layer = _make_layer(name="Clarity")

    result = build_measurement_request_for_layer(request, layer)

    assert isinstance(result, MeasurementRequest)
    assert result.min_words == 500
    assert result.max_words == 800
    assert result.word_count_enforcement is word_count_enforcement
    assert result.phrase_frequency is phrase_frequency
    assert result.lexical_diversity is lexical_diversity
    assert result.content_type == "other"
    assert result.json_output is True
    assert result.json_schema is schema
    assert result.json_expectations is expectations
    assert result.target_field == "generated_text"

    # Whitelist: forbidden fields must NOT cross over even if present on the
    # original request. ``MeasurementRequest`` exposes them only with neutral
    # ("off") defaults.
    assert result.cumulative_text is None
    assert result.include_stylistic_metrics is False
    assert not hasattr(result, "llm_accent_guard")
    assert not hasattr(result, "prompt")

    # Layer name is tracked for observability but kept private (repr=False).
    assert result._source_layer_name == "Clarity"


def test_build_measurement_request_returns_none_when_nothing_applies():
    request = SimpleNamespace(
        min_words=None,
        max_words=None,
        word_count_enforcement=None,
        phrase_frequency=SimpleNamespace(enabled=False),
        lexical_diversity=SimpleNamespace(enabled=False),
        json_output=False,
        json_schema=None,
        target_field=None,
    )
    layer = _make_layer()
    assert build_measurement_request_for_layer(request, layer) is None


def test_build_measurement_request_returns_none_when_request_is_none():
    layer = _make_layer()
    assert build_measurement_request_for_layer(None, layer) is None


def test_build_measurement_request_detects_single_active_field():
    """A single active validator is enough to produce a MeasurementRequest."""
    request = SimpleNamespace(
        min_words=None,
        max_words=None,
        word_count_enforcement=None,
        phrase_frequency=SimpleNamespace(enabled=True, rules=[object()]),
        lexical_diversity=None,
        json_output=False,
        json_schema=None,
        target_field=None,
    )
    layer = _make_layer()
    result = build_measurement_request_for_layer(request, layer)
    assert result is not None
    assert result.phrase_frequency is request.phrase_frequency
    assert result.min_words is None
    assert result.max_words is None


def test_build_measurement_request_preserves_content_type_json_alias():
    request = SimpleNamespace(
        min_words=None,
        max_words=None,
        word_count_enforcement=None,
        phrase_frequency=None,
        lexical_diversity=None,
        content_type="json",
        json_output=False,
        json_schema=None,
        target_field=None,
    )
    layer = _make_layer()

    result = build_measurement_request_for_layer(request, layer)

    assert result is not None
    assert result.content_type == "json"
    assert result.json_output is True


# ---------------------------------------------------------------------------
# DraftValidationResult.build_visible_payload MEASUREMENT_ONLY invariant
# ---------------------------------------------------------------------------


def test_measurement_only_payload_omits_gate_fields():
    """Integration guard: MEASUREMENT_ONLY scope must NOT leak approved/hard_failed."""
    result = _approved_measurement_result()
    payload = result.build_visible_payload(PayloadScope.MEASUREMENT_ONLY)
    assert "approved" not in payload
    assert "hard_failed" not in payload


# ---------------------------------------------------------------------------
# Integration: evaluate_content routes through call_ai_with_validation_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_content_uses_tool_loop_when_eligible():
    """Eligible request+layer triggers ``call_ai_with_validation_tools``
    with the expected scoping arguments."""

    request = _make_request(
        json_output=True,
        json_schema={"type": "object"},
        target_field="generated_text",
    )
    layer = _make_layer()

    approved = _approved_measurement_result()
    envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.QA,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        payload={
            "score": 9.5,
            "feedback": "Passed",
            "deal_breaker": False,
            "deal_breaker_reason": None,
        },
    )

    ai_service = Mock()
    ai_service.call_ai_with_validation_tools = AsyncMock(return_value=("{}", envelope))

    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)

    service = QAEvaluationService(ai_service)

    captured_calls: Dict[str, Any] = {}

    async def _callback(event_type: str, payload: Dict[str, Any]) -> None:
        captured_calls.setdefault("events", []).append((event_type, payload))

    with _patch_model_info(), \
         patch("qa_evaluation_service.validate_generation_candidate", return_value=approved):
        evaluation = await service.evaluate_content(
            content="Generated body.",
            criteria=layer.criteria,
            model="gpt-4o",
            layer_name=layer.name,
            min_score=layer.min_score,
            original_request=request,
            layer=layer,
            bypass_engine=bypass_engine,
            session_id="sess-1",
            project_id="proj-1",
            tool_event_callback=_callback,
            request_edit_info=False,
        )

        # NOTE: validate the ``validation_callback`` wiring while the patch
        # context is still live; the closure resolves the factory function
        # from the module namespace at call time.
        assert ai_service.call_ai_with_validation_tools.await_count == 1
        call_kwargs = ai_service.call_ai_with_validation_tools.await_args.kwargs
        assert call_kwargs["loop_scope"] is LoopScope.QA
        assert call_kwargs["payload_scope"] is PayloadScope.MEASUREMENT_ONLY
        assert call_kwargs["output_contract"] is OutputContract.JSON_STRUCTURED
        assert call_kwargs["stop_on_approval"] is False
        assert call_kwargs["retries_enabled"] is False
        assert call_kwargs["initial_measurement_text"] == "Generated body."
        assert call_kwargs["tool_event_callback"] is _callback
        # QA_MAX_TOOL_ROUNDS default is 3 (see config.py).
        assert call_kwargs["max_tool_rounds"] == 3

        # Validate the callback was wired to return a DraftValidationResult.
        vc = call_kwargs["validation_callback"]
        cb_result = vc("any candidate string")
        assert isinstance(cb_result, DraftValidationResult)
        assert cb_result.approved is True

    # QAEvaluation should be built from the parsed tool-loop payload.
    assert evaluation.score == 9.5
    assert evaluation.feedback == "Passed"


@pytest.mark.asyncio
async def test_evaluate_content_validation_callback_fails_unmet_json_expectations():
    expectations = [{"path": "status", "equals": "ok"}]
    request = _make_request(
        min_words=None,
        max_words=None,
        json_output=True,
        json_schema=None,
        json_expectations=expectations,
    )
    layer = _make_layer()

    envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.QA,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        payload={
            "score": 9.0,
            "feedback": "Passed",
            "deal_breaker": False,
            "deal_breaker_reason": None,
        },
    )

    ai_service = Mock()
    ai_service.call_ai_with_validation_tools = AsyncMock(return_value=("{}", envelope))

    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)

    service = QAEvaluationService(ai_service)

    with _patch_model_info():
        await service.evaluate_content(
            content="Generated body.",
            criteria=layer.criteria,
            model="gpt-4o",
            layer_name=layer.name,
            min_score=layer.min_score,
            original_request=request,
            layer=layer,
            bypass_engine=bypass_engine,
            request_edit_info=False,
        )

        call_kwargs = ai_service.call_ai_with_validation_tools.await_args.kwargs
        validation_callback = call_kwargs["validation_callback"]

        failed = validation_callback('{"status": "bad"}')
        passed = validation_callback('{"status": "ok"}')

    assert failed.approved is False
    assert failed.metrics["json_output"]["passed"] is False
    assert "Expected value" in failed.feedback
    assert passed.approved is True


@pytest.mark.asyncio
async def test_evaluate_content_validation_callback_uses_strict_loose_json_options():
    request = _make_request(
        min_words=None,
        max_words=None,
        json_output=True,
        json_schema=None,
    )
    layer = _make_layer()

    envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.QA,
        turns=1,
        accepted=True,
        accepted_via="assistant_final",
        payload={
            "score": 9.0,
            "feedback": "Passed",
            "deal_breaker": False,
            "deal_breaker_reason": None,
        },
    )

    ai_service = Mock()
    ai_service.call_ai_with_validation_tools = AsyncMock(return_value=("{}", envelope))

    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)

    service = QAEvaluationService(ai_service)

    with _patch_model_info():
        await service.evaluate_content(
            content="Generated body.",
            criteria=layer.criteria,
            model="gpt-4o",
            layer_name=layer.name,
            min_score=layer.min_score,
            original_request=request,
            layer=layer,
            bypass_engine=bypass_engine,
            request_edit_info=False,
        )

        call_kwargs = ai_service.call_ai_with_validation_tools.await_args.kwargs
        validation_callback = call_kwargs["validation_callback"]

        pure_json = validation_callback('{"status": "ok"}')
        prose_wrapped = validation_callback('Sure, here is JSON: {"status": "ok"}')
        fenced = validation_callback('```json\n{"status": "ok"}\n```')
        scalar = validation_callback('"ok"')

    assert pure_json.approved is True
    assert prose_wrapped.approved is True
    assert prose_wrapped.metrics["json_output"]["passed"] is True
    assert fenced.approved is True
    assert fenced.metrics["json_output"]["passed"] is True
    assert scalar.approved is False
    assert scalar.metrics["json_output"]["passed"] is False


@pytest.mark.asyncio
async def test_evaluate_content_single_shot_when_not_eligible():
    """``qa_tools_mode="never"`` must route through the legacy single-shot path."""

    request = _make_request(qa_tools_mode="never")
    layer = _make_layer()

    ai_service = Mock()
    ai_service.call_ai_with_validation_tools = AsyncMock()
    # Single-shot path ends up calling ``generate_content`` returning JSON.
    ai_service.generate_content = AsyncMock(
        return_value='{"score": 8.0, "feedback": "Passed", "deal_breaker": false, "deal_breaker_reason": null}'
    )

    bypass_engine = Mock()
    bypass_engine.can_bypass_layer = Mock(return_value=False)

    service = QAEvaluationService(ai_service)

    with _patch_model_info():
        evaluation = await service.evaluate_content(
            content="Body.",
            criteria=layer.criteria,
            model="gpt-4o",
            layer_name=layer.name,
            min_score=layer.min_score,
            original_request=request,
            layer=layer,
            bypass_engine=bypass_engine,
            request_edit_info=False,
        )

    ai_service.call_ai_with_validation_tools.assert_not_awaited()
    assert ai_service.generate_content.await_count == 1
    assert evaluation.score == 8.0
