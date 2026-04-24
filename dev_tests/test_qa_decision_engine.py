"""Tests for layer approval decisions with evidence grounding."""

from types import SimpleNamespace

from core.qa_decision_engine import _check_minority_deal_breakers, _evaluate_layer_based_approval
from models import QAEvaluation, QALayer


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


def _qa_eval(
    *,
    model: str = "model_a",
    layer: str = "Final Global QA Verification",
    score: float = 8.5,
    deal_breaker: bool = False,
):
    return QAEvaluation(
        model=model,
        layer=layer,
        score=score,
        feedback="ok" if not deal_breaker else "blocked",
        deal_breaker=deal_breaker,
        deal_breaker_reason="real issue" if deal_breaker else None,
        passes_score=score >= 8.0,
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


def test_fast_global_uses_global_contract_without_original_layer_results():
    """Fast-global approval must not expect every original QA layer result."""
    request = _make_request(max_iterations=3)
    request.qa_models = ["model_a"]
    request.qa_layers = [
        QALayer(name="Original A", description="A", criteria="A", min_score=10.0, is_mandatory=True),
        QALayer(name="Original B", description="B", criteria="B", min_score=10.0, is_mandatory=True),
    ]
    synthetic_layer = QALayer(
        name="Final Global QA Verification",
        description="global",
        criteria="global",
        min_score=8.0,
        is_mandatory=True,
    )
    qa_results = {
        "Final Global QA Verification": {
            "model_a": _qa_eval(score=8.2),
        }
    }
    consensus = SimpleNamespace(
        average_score=8.2,
        layer_averages={"Final Global QA Verification": 8.2},
    )

    result = _evaluate_layer_based_approval(
        qa_results=qa_results,
        consensus_result=consensus,
        request=request,
        session_id="sess-1",
        iteration=1,
        evaluated_layers=[synthetic_layer],
        qa_summary={"summary": {"approval_contract": "fast_global"}},
    )

    assert result["approved"] is True
    assert result["final_rejection"] is False
    assert "Fast global" in result["reason"]


def test_fast_global_fails_below_min_global_score():
    """Fast-global semantic approval is gated by request.min_global_score."""
    request = _make_request(max_iterations=3)
    request.qa_models = ["model_a"]
    request.min_global_score = 8.0
    synthetic_layer = QALayer(
        name="Final Global QA Verification",
        description="global",
        criteria="global",
        min_score=8.0,
        is_mandatory=True,
    )
    qa_results = {
        "Final Global QA Verification": {
            "model_a": _qa_eval(score=7.9),
        }
    }
    consensus = SimpleNamespace(
        average_score=7.9,
        layer_averages={"Final Global QA Verification": 7.9},
    )

    result = _evaluate_layer_based_approval(
        qa_results=qa_results,
        consensus_result=consensus,
        request=request,
        session_id="sess-1",
        iteration=1,
        evaluated_layers=[synthetic_layer],
        qa_summary={"summary": {"approval_contract": "fast_global"}},
    )

    assert result["approved"] is False
    assert result["final_rejection"] is False
    assert "global score 7.90 < 8.0" in result["reason"]


def test_minority_deal_breaker_checker_excludes_50_50_ties():
    """A 2/4 deal-breaker split belongs to tie handling, not minority handling."""
    qa_results = {
        "Accuracy": {
            "model_a": _qa_eval(model="model_a", layer="Accuracy", deal_breaker=True),
            "model_b": _qa_eval(model="model_b", layer="Accuracy", deal_breaker=True),
            "model_c": _qa_eval(model="model_c", layer="Accuracy", deal_breaker=False),
            "model_d": _qa_eval(model="model_d", layer="Accuracy", deal_breaker=False),
        }
    }

    result = _check_minority_deal_breakers(
        qa_results,
        ["model_a", "model_b", "model_c", "model_d"],
    )

    assert result["has_minority_deal_breakers"] is False
    assert result["deal_breaker_count"] == 0


def test_majority_deal_breaker_blocks_when_collection_did_not_early_stop():
    """Collected final verification results with a majority deal-breaker cannot approve."""
    request = _make_request(max_iterations=3)
    request.qa_models = ["model_a", "model_b", "model_c"]
    layer = QALayer(
        name="Accuracy",
        description="accuracy",
        criteria="accuracy",
        min_score=7.0,
        is_mandatory=True,
    )
    qa_results = {
        "Accuracy": {
            "model_a": _qa_eval(model="model_a", layer="Accuracy", score=9.0, deal_breaker=True),
            "model_b": _qa_eval(model="model_b", layer="Accuracy", score=9.0, deal_breaker=True),
            "model_c": _qa_eval(model="model_c", layer="Accuracy", score=9.0, deal_breaker=False),
        }
    }
    consensus = SimpleNamespace(
        average_score=9.0,
        layer_averages={"Accuracy": 9.0},
    )

    result = _evaluate_layer_based_approval(
        qa_results=qa_results,
        consensus_result=consensus,
        request=request,
        session_id="sess-1",
        iteration=1,
        evaluated_layers=[layer],
        qa_summary={"summary": {"layers_summary": {"Accuracy": {"passed": True}}}},
    )

    assert result["approved"] is False
    assert result["deal_breaker_type"] == "majority_consensus_retry"
