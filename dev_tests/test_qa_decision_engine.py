"""Tests for layer approval decisions with evidence grounding."""

from types import SimpleNamespace

from core.qa_decision_engine import _evaluate_layer_based_approval


def _make_request(*, max_iterations: int = 3, gran_sabio_fallback: bool = False):
    return SimpleNamespace(
        qa_layers=[],
        qa_models=[],
        max_iterations=max_iterations,
        min_global_score=8.0,
        gran_sabio_fallback=gran_sabio_fallback,
    )


def _make_consensus_result(*, average_score: float = 5.0):
    return SimpleNamespace(
        average_score=average_score,
        layer_averages={},
    )


def test_grounding_deal_breaker_forces_retry_before_last_iteration():
    """Grounding-only requests must not auto-approve when grounding blocks the content."""
    result = _evaluate_layer_based_approval(
        qa_results={},
        consensus_result=_make_consensus_result(),
        request=_make_request(max_iterations=3),
        session_id="sess-1",
        iteration=1,
        evaluated_layers=[],
        qa_summary={
            "evidence_grounding": {
                "passed": False,
                "flagged_claims": 2,
                "claims_verified": 3,
                "max_budget_gap": 0.91,
                "triggered_action": "deal_breaker",
            }
        },
    )

    assert result["approved"] is False
    assert result["final_rejection"] is False
    assert result["deal_breaker_type"] == "evidence_grounding_retry"


def test_grounding_deal_breaker_final_rejection_at_max_iterations():
    """Grounding-only requests should reject cleanly once iterations are exhausted."""
    result = _evaluate_layer_based_approval(
        qa_results={},
        consensus_result=_make_consensus_result(),
        request=_make_request(max_iterations=2),
        session_id="sess-1",
        iteration=2,
        evaluated_layers=[],
        qa_summary={
            "evidence_grounding": {
                "passed": False,
                "flagged_claims": 2,
                "claims_verified": 3,
                "max_budget_gap": 0.91,
                "triggered_action": "deal_breaker",
            }
        },
    )

    assert result["approved"] is False
    assert result["final_rejection"] is True
    assert result["deal_breaker_type"] == "evidence_grounding"


def test_grounding_warn_does_not_block_grounding_only_requests():
    """Warn-level grounding should remain non-blocking when there are no semantic layers."""
    result = _evaluate_layer_based_approval(
        qa_results={},
        consensus_result=_make_consensus_result(average_score=5.0),
        request=_make_request(max_iterations=2),
        session_id="sess-1",
        iteration=1,
        evaluated_layers=[],
        qa_summary={
            "evidence_grounding": {
                "passed": False,
                "flagged_claims": 1,
                "claims_verified": 3,
                "max_budget_gap": 0.62,
                "triggered_action": "warn",
            }
        },
    )

    assert result["approved"] is True
    assert result["final_rejection"] is False
