"""Regression coverage for Gran Sabio false-positive feedback handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from consensus_engine import ConsensusEngine
from core.generation_processor import (
    _feedback_memory_text,
    build_iteration_feedback_prompt,
)
from models import QAEvaluation
from qa_result_utils import apply_gran_sabio_false_positive_override


def _evaluation(
    *,
    model: str,
    feedback: str,
    score: float,
    deal_breaker: bool,
    deal_breaker_reason: str | None,
    passes_score: bool,
) -> QAEvaluation:
    return QAEvaluation(
        model=model,
        layer="Craft",
        score=score,
        feedback=feedback,
        deal_breaker=deal_breaker,
        deal_breaker_reason=deal_breaker_reason,
        passes_score=passes_score,
    )


def _resolved_false_positive() -> QAEvaluation:
    evaluation = _evaluation(
        model="qa-a",
        feedback="The draft has no rhyme anywhere.",
        score=3.0,
        deal_breaker=True,
        deal_breaker_reason="No real rhyme despite a declared scheme.",
        passes_score=False,
    )
    apply_gran_sabio_false_positive_override(
        evaluation,
        final_score=8.5,
        layer_min_score=8.0,
        original_reason=evaluation.deal_breaker_reason or "",
    )
    return evaluation


def test_resolved_false_positive_is_auditable_but_not_actionable() -> None:
    evaluation = _resolved_false_positive()
    engine = ConsensusEngine(ai_service=MagicMock())

    feedback_by_layer, actionable = engine._collect_feedback_details(
        {"Craft": {"qa-a": evaluation}},
        {"Craft": 8.5},
    )

    assert evaluation.feedback == "The draft has no rhyme anywhere."
    assert evaluation.metadata["gran_sabio_original_deal_breaker_reason"]
    assert feedback_by_layer[0]["deal_breakers"] == []
    assert feedback_by_layer[0]["model_feedback"] == [
        {
            "model": "qa-a",
            "evaluator": "Evaluator",
            "score": 8.5,
            "deal_breaker": False,
            "feedback": "",
            "resolution": "gran_sabio_false_positive",
            "actionable": False,
        }
    ]
    assert actionable == []


def test_iteration_prompt_excludes_override_but_keeps_real_failure() -> None:
    overridden = _resolved_false_positive()
    real_failure = _evaluation(
        model="qa-b",
        feedback="The evidence for the closing claim is missing.",
        score=5.0,
        deal_breaker=False,
        deal_breaker_reason=None,
        passes_score=False,
    )
    qa_results = {"Craft": {"qa-a": overridden, "qa-b": real_failure}}
    engine = ConsensusEngine(ai_service=MagicMock())
    feedback_by_layer, actionable = engine._collect_feedback_details(
        qa_results,
        {"Craft": 6.75},
    )

    prompt = build_iteration_feedback_prompt(
        {
            "deal_breaker_found": False,
            "qa_results": qa_results,
            "consensus": SimpleNamespace(
                feedback_by_layer=feedback_by_layer,
                actionable_feedback=actionable,
            ),
        }
    )

    assert "no rhyme anywhere" not in prompt
    assert "No real rhyme despite" not in prompt
    assert "closing claim is missing" in prompt


def test_fallback_prompt_excludes_override_but_keeps_real_deal_breaker() -> None:
    overridden = _resolved_false_positive()
    real_rejection = _evaluation(
        model="qa-b",
        feedback="The draft invents a date.",
        score=2.0,
        deal_breaker=True,
        deal_breaker_reason="Invented evidence is a critical failure.",
        passes_score=False,
    )

    prompt = build_iteration_feedback_prompt(
        {
            "deal_breaker_found": True,
            "qa_results": {"Craft": {"qa-a": overridden, "qa-b": real_rejection}},
            "consensus": None,
        }
    )

    assert "no rhyme anywhere" not in prompt
    assert "No real rhyme despite" not in prompt
    assert "Invented evidence is a critical failure" in prompt


def test_feedback_memory_fallback_excludes_override_but_keeps_real_failure() -> None:
    overridden = _resolved_false_positive()
    real_failure = _evaluation(
        model="qa-b",
        feedback="The final paragraph contradicts the supplied evidence.",
        score=4.0,
        deal_breaker=False,
        deal_breaker_reason=None,
        passes_score=False,
    )

    feedback_text = _feedback_memory_text(
        SimpleNamespace(actionable_feedback=[]),
        {"Craft": {"qa-a": overridden, "qa-b": real_failure}},
    )

    assert "no rhyme anywhere" not in feedback_text
    assert "No real rhyme despite" not in feedback_text
    assert "final paragraph contradicts" in feedback_text


def test_feedback_memory_prefers_consensus_actionable_feedback() -> None:
    feedback_text = _feedback_memory_text(
        SimpleNamespace(actionable_feedback=["Fix the verified chronology issue."]),
        {"Craft": {"qa-a": _resolved_false_positive()}},
    )

    assert feedback_text == "Fix the verified chronology issue."
