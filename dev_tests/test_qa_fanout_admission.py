"""Batch 3 coverage: barrier 2 — atomic QA fan-out admission (spec §6).

Covers:
- strict_all all-or-nothing: one non-admittable evaluator blocks the WHOLE layer
  before any evaluator is dispatched (``evaluate_content`` never called).
- heterogeneous per-receptor capacities + the public per-receptor table (mirror
  rule labels + numbers).
- a context error is NEVER absorbed by quorum / reclassified as model_unavailable;
  the scheduler aborts the layer and cancels in-flight sibling slots.
- QA tool-loop ``context_too_large`` surfaces as ``ContextBudgetExceeded`` (not a
  bare RuntimeError absorbed as an "unexpected" technical failure).
- final-verification path reuses the same admission.
- observe mode proceeds.
- the session error surface (``error_type="context_admission"`` + payload).

Enforce-mode tests set ``CONTEXT_ADMISSION_MODE`` explicitly (the suite defaults
to observe via ``conftest._context_admission_observe_default``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import context_budget
from context_budget import (
    CapacityConfidence,
    ContextAdmissionError,
    ContextBudgetExceeded,
    FanoutAdmissionError,
    TokenLimitsResolution,
)
from model_aliasing import ModelAliasRegistry
from models import ContentRequest, QALayer
from qa_engine import QAEngine
from qa_scheduler import QAScheduler, QASchedulerPolicy, QASchedulerSlot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _limits(
    model: str,
    *,
    max_input=None,
    max_total=None,
    output_accounting="within_total",
    confidence=CapacityConfidence.VERIFIED,
    provenance="manual_curated",
) -> TokenLimitsResolution:
    return TokenLimitsResolution(
        model=model,
        provider="claude",
        max_input_tokens=max_input,
        max_output_tokens=None,
        max_total_context_tokens=max_total,
        output_accounting=output_accounting,
        reasoning_accounting="within_output",
        provenance=provenance,
        confidence=confidence,
    )


def _make_engine() -> QAEngine:
    service = MagicMock()
    service._native_token_count = AsyncMock(return_value=None)
    service.model_call_concurrency_slot = None  # -> use the no-op async context
    bypass = MagicMock()
    bypass.should_bypass_qa_layer = Mock(return_value=False)
    return QAEngine(ai_service=service, bypass_engine=bypass)


def _layer(name="Accuracy", criteria="Be accurate and complete.") -> QALayer:
    return QALayer(name=name, description="d", criteria=criteria, min_score=7.0, order=1)


def _request(qa_models, *, client_named=True) -> ContentRequest:
    req = ContentRequest(
        prompt="Evaluate the content.",
        content_type="biography",
        generator_model="gen-model",
        qa_models=list(qa_models),
    )
    req._client_named_qa_models = client_named
    return req


def _slots(model_names) -> list[QASchedulerSlot]:
    registry = None
    return [
        QASchedulerSlot(
            index=i,
            model=name,
            model_name=name,
            result_key=name,
            evaluator_label=f"Evaluator {chr(ord('A') + i)}",
            timeout_seconds=5.0,
            slot_id=f"qa:{i}",
        )
        for i, name in enumerate(model_names)
    ]


# Big content -> ~213k conservative tokens (680k chars / 3.2).
BIG_CONTENT = "x " * 340000


# ---------------------------------------------------------------------------
# strict_all all-or-nothing: zero dispatch on one non-admittable evaluator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_all_blocks_all_evaluators_before_dispatch(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    engine = _make_engine()
    engine.evaluate_content = AsyncMock()  # must never be called
    models = ["qa-a", "qa-b", "qa-c"]
    request = _request(models)
    registry = ModelAliasRegistry.from_request(request)

    def _limits_by_model(model):
        if model == "qa-b":
            return _limits(model, max_total=1000)  # tiny -> overflow
        return _limits(model, max_total=1_000_000)

    with patch.object(context_budget, "resolve_token_limits", side_effect=_limits_by_model), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(FanoutAdmissionError):
            await engine._evaluate_single_semantic_layer(
                content="some generated content",
                layer=_layer(),
                qa_models=models,
                qa_model_names=models,
                original_request=request,
                model_alias_registry=registry,
            )

    engine.evaluate_content.assert_not_awaited()


# ---------------------------------------------------------------------------
# Final-verification path reuses the same admission (spec §6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_verification_layer_gets_fanout_admission(monkeypatch):
    # Final verification calls _evaluate_single_semantic_layer, so proving the
    # admission fires there proves the final-verification path is covered.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    engine = _make_engine()
    engine.evaluate_content = AsyncMock()
    models = ["qa-a", "qa-b"]
    request = _request(models)
    registry = ModelAliasRegistry.from_request(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_total=500)), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(FanoutAdmissionError):
            await engine._evaluate_single_semantic_layer(
                content="content under final verification",
                layer=_layer(),
                qa_models=models,
                qa_model_names=models,
                original_request=request,
                model_alias_registry=registry,
            )

    engine.evaluate_content.assert_not_awaited()


# ---------------------------------------------------------------------------
# Heterogeneous capacities + per-receptor public table (mirror rule)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heterogeneous_capacities_and_public_table(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    engine = _make_engine()
    models = ["qa-272k", "qa-200k", "qa-1m"]
    request = _request(models, client_named=True)
    registry = ModelAliasRegistry.from_request(request)
    slots = _slots(models)

    def _limits_by_model(model):
        if model == "qa-272k":
            return _limits(model, max_total=272_000, output_accounting="within_total")
        if model == "qa-200k":
            return _limits(model, max_total=200_000, output_accounting="within_total")
        return _limits(model, max_input=1_000_000, output_accounting="separate")

    with patch.object(context_budget, "resolve_token_limits", side_effect=_limits_by_model), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(FanoutAdmissionError) as exc_info:
            await engine._admit_qa_fanout(
                content=BIG_CONTENT,
                layer=_layer(),
                slots=slots,
                original_request=request,
                request_edit_info=False,
                marker_mode="phrase",
                marker_length=None,
                word_map_formatted=None,
                draft_map_formatted=None,
                input_images=None,
                edit_history=None,
                model_alias_registry=registry,
            )

    table = exc_info.value.receptor_table()
    assert len(table) == 3
    labels = [row["receptor_label"] for row in table]
    # client_named=True -> verbatim identifiers.
    assert labels == models
    # numbers are always public.
    for row in table:
        assert isinstance(row["required_tokens"], int)
        assert row["available_tokens"] is not None
        assert "decision" in row
    # Only the 200k evaluator fails; the 272k and 1m ones fit.
    failing_labels = [label for label, _ in exc_info.value.failing]
    assert failing_labels == ["qa-200k"]


@pytest.mark.asyncio
async def test_mirror_rule_routing_chosen_uses_positional_labels(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    engine = _make_engine()
    models = ["qa-a", "qa-b", "qa-c"]
    request = _request(models, client_named=False)  # routing-chosen
    registry = ModelAliasRegistry.from_request(request)
    slots = _slots(models)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_total=500)), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(FanoutAdmissionError) as exc_info:
            await engine._admit_qa_fanout(
                content="content",
                layer=_layer(),
                slots=slots,
                original_request=request,
                request_edit_info=False,
                marker_mode="phrase",
                marker_length=None,
                word_map_formatted=None,
                draft_map_formatted=None,
                input_images=None,
                edit_history=None,
                model_alias_registry=registry,
            )

    labels = [row["receptor_label"] for row in exc_info.value.receptor_table()]
    assert labels == ["Evaluator A", "Evaluator B", "Evaluator C"]
    for leaked in models:
        assert leaked not in labels


# ---------------------------------------------------------------------------
# Observe mode proceeds (no raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observe_mode_fanout_proceeds(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "observe")
    engine = _make_engine()
    models = ["qa-a", "qa-b"]
    request = _request(models)
    registry = ModelAliasRegistry.from_request(request)
    slots = _slots(models)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_total=500)), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        # Must NOT raise in observe mode even though the payload overflows.
        await engine._admit_qa_fanout(
            content=BIG_CONTENT,
            layer=_layer(),
            slots=slots,
            original_request=request,
            request_edit_info=False,
            marker_mode="phrase",
            marker_length=None,
            word_map_formatted=None,
            draft_map_formatted=None,
            input_images=None,
            edit_history=None,
            model_alias_registry=registry,
        )


# ---------------------------------------------------------------------------
# Context error never absorbed by quorum + sibling cancellation (scheduler)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_error_aborts_layer_and_cancels_siblings():
    cancelled = {"sibling": False}

    async def evaluate_slot(slot, attempt):
        if slot.index == 0:
            raise ContextBudgetExceeded(
                "mid-flight provider overflow",
                call_id="qa.evaluate_layer",
                public_label=slot.evaluator_label,
            )
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled["sibling"] = True
            raise
        return None

    scheduler = QAScheduler(QASchedulerPolicy(execution_mode="parallel"))
    slots = _slots(["qa-a", "qa-b"])

    with pytest.raises(ContextAdmissionError):  # NOT QASchedulerUnavailableError
        await scheduler.evaluate_layer(
            layer_name="Accuracy",
            has_deal_breaker_criteria=False,
            slots=slots,
            evaluate_slot=evaluate_slot,
        )

    assert cancelled["sibling"] is True


@pytest.mark.asyncio
async def test_midflight_context_error_not_reclassified_as_model_unavailable(monkeypatch):
    # Admission passes (large limits); a context error surfaces mid-flight from
    # evaluate_content and must propagate as ContextAdmissionError, never as a
    # QAModelUnavailableError absorbed by quorum.
    monkeypatch.setenv("CONTEXT_ADMISSION_MODE", "enforce")
    engine = _make_engine()
    engine.evaluate_content = AsyncMock(
        side_effect=ContextBudgetExceeded(
            "provider overflow", call_id="qa.evaluate_layer", public_label="Evaluator A"
        )
    )
    models = ["qa-a"]
    request = _request(models)
    registry = ModelAliasRegistry.from_request(request)

    with patch.object(context_budget, "resolve_token_limits",
                      side_effect=lambda m: _limits(m, max_total=1_000_000)), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(ContextBudgetExceeded):
            await engine._evaluate_single_semantic_layer(
                content="short content",
                layer=_layer(),
                qa_models=models,
                qa_model_names=models,
                original_request=request,
                model_alias_registry=registry,
            )

    engine.evaluate_content.assert_awaited()


# ---------------------------------------------------------------------------
# QA tool-loop context_too_large -> ContextBudgetExceeded (not RuntimeError)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qa_tool_loop_context_too_large_raises_context_budget_exceeded():
    from qa_evaluation_service import QAEvaluationService
    from tool_loop_models import LoopScope, ToolLoopEnvelope

    service_ai = MagicMock()
    skip_envelope = ToolLoopEnvelope(
        loop_scope=LoopScope.QA,
        tools_skipped_reason="context_too_large",
        context_overflow_kind="model_context_overflow",
    )
    service_ai.call_ai_with_validation_tools = AsyncMock(return_value=("", skip_envelope))
    qa_service = QAEvaluationService(service_ai)

    request = SimpleNamespace(
        content_type="biography",
        prompt="Write about X.",
        min_words=500,
        max_words=800,
    )
    layer = SimpleNamespace(
        name="Clarity",
        criteria="Be clear.",
        min_score=7.0,
        deal_breaker_criteria=None,
        concise_on_pass=True,
        include_input_images=False,
    )

    with patch("qa_evaluation_service._should_use_qa_tools", return_value=True), \
         patch("qa_evaluation_service.build_measurement_request_for_layer",
               return_value=SimpleNamespace(json_output=False, json_schema=None, target_field=None)), \
         patch.object(__import__("config").config.__class__, "resolve_qa_max_tokens", return_value=6000):
        with pytest.raises(ContextBudgetExceeded):
            await qa_service.evaluate_content(
                content="content to evaluate",
                criteria="Be clear.",
                model="qa-model",
                layer_name="Clarity",
                min_score=7.0,
                original_request=request,
                layer=layer,
            )

    assert service_ai.call_ai_with_validation_tools.await_count == 1


# ---------------------------------------------------------------------------
# Session error surface (error_type context_admission + public payload)
# ---------------------------------------------------------------------------


def test_generation_error_result_surfaces_context_admission():
    from core.generation_processor import (
        _build_generation_error_result,
        _context_admission_public_payload,
    )

    exc = FanoutAdmissionError(reports=[], failing=[])
    payload = _context_admission_public_payload(exc)
    assert payload["type"] == "context_admission"
    assert payload["error_code"] == "context_budget_exceeded"
    assert payload["receptors"] == []

    session: dict = {"current_iteration": 2}
    request = ContentRequest(prompt="Write something", generator_model="gen-model")
    final_result = _build_generation_error_result(session, request, exc)

    assert session["error_type"] == "context_admission"
    assert session["error_code"] == "context_budget_exceeded"
    assert isinstance(session["context_admission"], dict)
    assert final_result["error_type"] == "context_admission"
    assert final_result["error_code"] == "context_budget_exceeded"
    assert final_result["context_admission"]["type"] == "context_admission"
    assert final_result["status"] == "failed"


def test_non_fanout_context_error_public_payload_has_label():
    from core.generation_processor import _context_admission_public_payload

    exc = ContextBudgetExceeded(
        "boom", call_id="qa.evaluate_layer", public_label="Evaluator A"
    )
    payload = _context_admission_public_payload(exc)
    assert payload["error_code"] == "context_budget_exceeded"
    assert payload["receptors"][0]["receptor_label"] == "Evaluator A"


# ---------------------------------------------------------------------------
# Gran Sabio re-QA path re-raises context admission errors (does not swallow)
# ---------------------------------------------------------------------------


def test_gran_sabio_reqa_reraises_context_admission_before_generic_handler():
    """A ContextAdmissionError from the Gran Sabio regenerated-content re-QA /
    final-verification path must re-raise so it reaches the outer handler and
    surfaces error_type='context_admission', instead of being swallowed into
    fallback_notes and continuing to standard Gran Sabio review.

    The re-QA block is inline inside the ~3000-line
    ``process_content_generation`` coroutine and cannot be driven in isolation
    with the existing fixtures (it requires a full generation run with every
    iteration rejected, then a Gran Sabio fallback regeneration). This
    regression guard anchors on the real source: the re-QA except chain must
    place ``except ContextAdmissionError: raise`` before the generic
    ``except Exception as e:``. FanoutAdmissionError being a ContextAdmissionError
    subclass proves the clause catches it; the "surfaces as context_admission at
    the outer handler" half is covered by
    test_generation_error_result_surfaces_context_admission.
    """
    import inspect

    from core import generation_processor

    # The re-QA fan-out error is a ContextAdmissionError, so it is caught by the
    # re-raise clause rather than the generic handler.
    assert issubclass(FanoutAdmissionError, ContextAdmissionError)

    src = inspect.getsource(generation_processor.process_content_generation)
    # Anchor on the generic-handler log line unique to the Gran Sabio re-QA loop.
    anchor = "Error evaluating Gran Sabio content"
    assert anchor in src
    region = src[: src.index(anchor)]
    ctx_idx = region.rindex("except ContextAdmissionError:")
    generic_idx = region.rindex("except Exception as e:")
    # The ContextAdmissionError re-raise is ordered before the generic handler...
    assert ctx_idx < generic_idx
    # ...and its body re-raises rather than swallowing.
    assert "raise" in region[ctx_idx:generic_idx]
