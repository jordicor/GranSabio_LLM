"""Regression tests for security fix AUX-01 (completion package).

AUX-01: evidence grounding used to fail open on any verification error,
returning a conservative pass
indistinguishable from a real one. Wave 1 made the engine apply the on_flag
policy and made BudgetScorer propagate errors. This package completes the fix:

- EvidenceGroundingResult carries explicit ``degraded``/``verification_error``
  fields, populated by GroundingEngine whenever verification could not run.
- The QA pipeline (QAEngine._evaluate_evidence_grounding) treats a degraded
  result according to the configured on_flag policy: fail-closed for
  deal_breaker/regenerate, visible (but non-blocking) for warn, instead of a
  pass indistinguishable from a real verification.

Each test reproduces the finding scenario and would have failed before the
fix (the degraded field did not exist and the pipeline scored a verification
error as a clean 10.0 pass).

No network access or API keys required: AI calls are mocked.
"""

from __future__ import annotations

from typing import Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from evidence_grounding.grounding_engine import GroundingEngine
from models import (
    ClaimBudgetResult,
    EvidenceGroundingConfig,
    EvidenceGroundingResult,
    EvidenceSpan,
    ExtractedClaim,
    QAEvaluation,
    SpanType,
)
from qa_engine import QAEngine

# =============================================================================
# Shared helpers
# =============================================================================


def _grounding_config(on_flag: str) -> EvidenceGroundingConfig:
    """Return an enabled grounding config with the given on_flag policy."""

    return EvidenceGroundingConfig(enabled=True, on_flag=on_flag, max_flagged_claims=2)


async def _run_engine_with_failing_extraction(on_flag: str) -> EvidenceGroundingResult:
    """Run the real engine with claim extraction failing (finding scenario)."""

    engine = GroundingEngine(MagicMock())
    engine.claim_extractor.extract_claims = AsyncMock(side_effect=Exception("API down"))
    return await engine.run_grounding_check(
        content="Test content",
        context="Test context",
        grounding_config=_grounding_config(on_flag),
    )


async def _run_engine_healthy() -> EvidenceGroundingResult:
    """Run the real engine over a fully mocked healthy verification path."""

    claim = ExtractedClaim(
        idx=0,
        claim="The sky is blue.",
        kind="factual",
        importance=0.9,
        cited_spans=["S0"],
        source_text="The sky is blue.",
    )
    span = EvidenceSpan(
        id="S0",
        text="The sky is blue.",
        span_type=SpanType.ASSERTION,
        start_char=0,
        end_char=17,
    )
    scored = ClaimBudgetResult(
        idx=0,
        claim="The sky is blue.",
        cited_spans=["S0"],
        posterior_yes=0.95,
        prior_yes=0.6,
        required_bits=0.5,
        observed_bits=1.2,
        budget_gap=-0.7,
        flagged=False,
        confidence_delta=0.35,
    )
    engine = GroundingEngine(MagicMock())
    engine.claim_extractor.extract_claims = AsyncMock(return_value=[claim])
    engine.evidence_matcher.classify_spans = AsyncMock(return_value=[span])
    engine.budget_scorer.score_claims = AsyncMock(return_value=[scored])
    return await engine.run_grounding_check(
        content="Test content",
        context="Test context",
        grounding_config=_grounding_config("deal_breaker"),
    )


async def _evaluate_in_pipeline(
    grounding_result: EvidenceGroundingResult,
    on_flag: str,
) -> Tuple[QAEvaluation, EvidenceGroundingResult]:
    """Feed a grounding result through the QA pipeline integration point."""

    qa_engine = QAEngine(MagicMock())
    qa_engine.grounding_engine.run_grounding_check = AsyncMock(return_value=grounding_result)
    return await qa_engine._evaluate_evidence_grounding(
        content="Test content",
        context="Test context",
        grounding_config=_grounding_config(on_flag),
    )


# =============================================================================
# AUX-01 (a): error results are explicitly degraded
# =============================================================================


@pytest.mark.asyncio
async def test_aux01_extraction_error_marks_result_degraded() -> None:
    """A claim-extraction error must produce degraded=True with the error
    message, so callers can never confuse it with a real verification."""

    result = await _run_engine_with_failing_extraction("deal_breaker")

    assert result.degraded is True
    assert result.verification_error is not None
    assert "API down" in result.verification_error
    assert result.passed is False
    assert result.triggered_action == "deal_breaker"


@pytest.mark.asyncio
async def test_aux01_warn_error_result_is_degraded_but_unblocked() -> None:
    """Under on_flag=warn the error keeps the non-blocking conservative pass,
    but the result must still be explicitly degraded and carry the error."""

    result = await _run_engine_with_failing_extraction("warn")

    assert result.degraded is True
    assert result.verification_error is not None
    assert "API down" in result.verification_error
    assert result.passed is True
    assert result.triggered_action is None
    assert result.claims_verified == 0


@pytest.mark.asyncio
async def test_aux01_healthy_result_is_not_degraded() -> None:
    """A real verification has degraded absent/False and no error message."""

    result = await _run_engine_healthy()

    assert result.passed is True
    assert result.degraded is False
    assert result.verification_error is None
    assert result.claims_verified == 1


# =============================================================================
# AUX-01 (b): the QA pipeline applies the on_flag policy to degraded results
# =============================================================================


@pytest.mark.asyncio
async def test_aux01_pipeline_degraded_deal_breaker_fails_closed() -> None:
    """With on_flag=deal_breaker a verification error must surface as a
    deal-breaker failure (fail-closed), not as delivered 'verified' content."""

    grounding_result = await _run_engine_with_failing_extraction("deal_breaker")
    qa_eval, _ = await _evaluate_in_pipeline(grounding_result, "deal_breaker")

    assert qa_eval.deal_breaker is True
    assert qa_eval.score == 0.0
    assert qa_eval.passes_score is False
    assert "verification could not run" in qa_eval.feedback
    assert "API down" in qa_eval.feedback
    assert qa_eval.deal_breaker_reason is not None
    assert "Verification could not run" in qa_eval.deal_breaker_reason
    assert "API down" in qa_eval.deal_breaker_reason
    assert qa_eval.metadata["grounding_result"]["degraded"] is True


@pytest.mark.asyncio
async def test_aux01_pipeline_degraded_regenerate_fails_closed() -> None:
    """With on_flag=regenerate a verification error must also fail closed so
    the content is regenerated instead of shipped unverified."""

    grounding_result = await _run_engine_with_failing_extraction("regenerate")
    qa_eval, _ = await _evaluate_in_pipeline(grounding_result, "regenerate")

    assert qa_eval.deal_breaker is True
    assert qa_eval.score == 0.0
    assert qa_eval.passes_score is False
    assert "'regenerate'" in qa_eval.feedback


@pytest.mark.asyncio
async def test_aux01_pipeline_degraded_warn_visible_not_blocking() -> None:
    """With on_flag=warn a verification error must stay non-blocking but
    visible: no clean 10.0 pass, degraded flag and error in the metadata."""

    grounding_result = await _run_engine_with_failing_extraction("warn")
    qa_eval, _ = await _evaluate_in_pipeline(grounding_result, "warn")

    # Non-blocking: no deal-breaker and passes_score stays True.
    assert qa_eval.deal_breaker is False
    assert qa_eval.deal_breaker_reason is None
    assert qa_eval.passes_score is True
    # Visible: warning score instead of the 10.0 of a real pass, explicit
    # feedback, and the degraded state propagated to the report metadata.
    assert qa_eval.score == 5.0
    assert "DEGRADED" in qa_eval.feedback
    assert "API down" in qa_eval.feedback
    assert qa_eval.metadata["grounding_result"]["degraded"] is True
    assert "API down" in qa_eval.metadata["grounding_result"]["verification_error"]
    assert qa_eval.metadata["prompt_facing_grounding_result"]["degraded"] is True


@pytest.mark.asyncio
async def test_aux01_pipeline_healthy_pass_unchanged() -> None:
    """A real verification still scores a clean 10.0 pass with no degradation."""

    grounding_result = await _run_engine_healthy()
    qa_eval, _ = await _evaluate_in_pipeline(grounding_result, "deal_breaker")

    assert qa_eval.score == 10.0
    assert qa_eval.deal_breaker is False
    assert qa_eval.passes_score is True
    assert "PASSED" in qa_eval.feedback
    assert qa_eval.metadata["grounding_result"]["degraded"] is False
