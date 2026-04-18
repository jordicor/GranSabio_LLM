"""Shared helpers for QA result health and deal-breaker quorum decisions."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from model_aliasing import get_evaluator_alias


TECHNICAL_QA_ERROR_TYPES = frozenset(
    {
        "api_failure",
        "timeout",
        "unexpected",
        "technical_failure",
        "model_unavailable",
    }
)

TECHNICAL_QA_FAILURE_REASONS = frozenset(
    {
        "technical error during evaluation",
        "timeout during evaluation",
    }
)


def required_valid_qa_models(
    configured_count: int,
    absolute: Optional[int] = None,
    ratio: Optional[float] = None,
) -> int:
    """Return the minimum valid semantic QA responses required for decisions."""

    if configured_count <= 0:
        return 0
    if absolute is not None:
        return max(1, min(int(absolute), configured_count))
    if ratio is not None:
        return max(1, min(math.ceil(configured_count * float(ratio)), configured_count))
    return math.floor(configured_count / 2) + 1


def is_technical_qa_failure(evaluation: Any) -> bool:
    """Return True when an evaluation represents provider/local failure."""

    metadata = getattr(evaluation, "metadata", None)
    if isinstance(metadata, dict):
        if metadata.get("technical_failure") is True:
            return True
        error_type = metadata.get("error_type")
        if isinstance(error_type, str) and error_type in TECHNICAL_QA_ERROR_TYPES:
            return True

    reason = (
        getattr(evaluation, "deal_breaker_reason", None)
        or getattr(evaluation, "reason", None)
        or ""
    )
    return str(reason).strip().lower() in TECHNICAL_QA_FAILURE_REASONS


def is_valid_semantic_qa_result(evaluation: Any) -> bool:
    """Return True for usable semantic QA decisions."""

    if is_technical_qa_failure(evaluation):
        return False
    return getattr(evaluation, "score", None) is not None or bool(
        getattr(evaluation, "deal_breaker", False)
    )


def semantic_deal_breakers(layer_results: Dict[str, Any]) -> List[Any]:
    """Return semantic deal-breaker evaluations, excluding technical failures."""

    return [
        evaluation
        for evaluation in layer_results.values()
        if is_valid_semantic_qa_result(evaluation)
        and bool(getattr(evaluation, "deal_breaker", False))
    ]


def apply_gran_sabio_false_positive_override(
    evaluation: Any,
    *,
    final_score: Optional[float],
    layer_min_score: Optional[float],
    original_reason: str,
) -> Dict[str, Any]:
    """
    Mark a Gran Sabio-approved minority deal-breaker as passed for this layer.

    In the minority deal-breaker flow, Gran Sabio's APPROVED decision means the
    flagged issue does not block the current QA layer. Keep the raw arbiter
    score in metadata for audit, but clamp the effective score to the layer's
    minimum so an approval cannot immediately re-fail the same layer.
    """

    def _score_or_default(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed

    previous_score = getattr(evaluation, "score", None)
    bounded_min_score = min(10.0, max(0.0, _score_or_default(layer_min_score, 0.0)))
    raw_score = _score_or_default(
        final_score,
        _score_or_default(previous_score, bounded_min_score),
    )
    effective_score = min(10.0, max(raw_score, bounded_min_score))

    evaluation.deal_breaker = False
    evaluation.deal_breaker_reason = None
    evaluation.reason = (
        f"[Gran Sabio Override] Originally flagged as deal-breaker but "
        f"Gran Sabio determined it to be false positive. Original reason: {original_reason}"
    )
    evaluation.score = effective_score
    evaluation.passes_score = True

    metadata = getattr(evaluation, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["gran_sabio_override"] = "false_positive"
    metadata["gran_sabio_raw_score"] = raw_score
    metadata["gran_sabio_effective_score"] = effective_score
    metadata["gran_sabio_layer_min_score"] = bounded_min_score
    metadata["gran_sabio_score_clamped_to_min"] = raw_score < bounded_min_score
    metadata["gran_sabio_score_capped_to_max"] = raw_score > 10.0
    if original_reason:
        metadata["gran_sabio_original_deal_breaker_reason"] = original_reason
    evaluation.metadata = metadata

    return {
        "previous_score": previous_score,
        "raw_score": raw_score,
        "effective_score": effective_score,
        "clamped_to_min": raw_score < bounded_min_score,
    }


def build_qa_counts(
    layer_results: Dict[str, Any],
    configured_count: int,
    *,
    skipped: int = 0,
    required_valid: Optional[int] = None,
) -> Dict[str, int]:
    """Build scheduler/quorum counters with separate technical and semantic counts."""

    attempted = len(layer_results)
    valid = sum(
        1 for evaluation in layer_results.values() if is_valid_semantic_qa_result(evaluation)
    )
    technical_failed = sum(
        1 for evaluation in layer_results.values() if is_technical_qa_failure(evaluation)
    )
    semantic_db = len(semantic_deal_breakers(layer_results))
    invalid = max(0, attempted - valid - technical_failed)
    required = required_valid if required_valid is not None else required_valid_qa_models(configured_count)
    required_majority = math.floor(valid / 2) + 1 if valid > 0 else 1

    return {
        "configured": configured_count,
        "attempted": attempted,
        "valid": valid,
        "technical_failed": technical_failed,
        "invalid": invalid,
        "skipped": max(0, skipped),
        "semantic_deal_breakers": semantic_db,
        "required_valid": required,
        "required_majority": required_majority,
    }


def build_deal_breaker_consensus(
    layer_results: Dict[str, Any],
    total_models: Iterable[str],
    *,
    skipped: int = 0,
    required_valid: Optional[int] = None,
) -> Dict[str, Any]:
    """Build majority deal-breaker consensus using valid semantic results only."""

    total_model_list = list(total_models or [])
    configured_count = len(total_model_list) or len(layer_results)
    counts = build_qa_counts(
        layer_results,
        configured_count,
        skipped=skipped,
        required_valid=required_valid,
    )
    deal_breaker_details = []

    for model, evaluation in layer_results.items():
        if is_valid_semantic_qa_result(evaluation) and getattr(evaluation, "deal_breaker", False):
            deal_breaker_details.append(
                {
                    "model": model,
                    "evaluator": get_evaluator_alias(evaluation, fallback=model),
                    "reason": getattr(evaluation, "deal_breaker_reason", None),
                }
            )

    valid = counts["valid"]
    deal_breaker_count = counts["semantic_deal_breakers"]
    immediate_stop = (
        valid >= counts["required_valid"]
        and deal_breaker_count >= counts["required_majority"]
    )

    return {
        "immediate_stop": immediate_stop,
        "deal_breaker_count": deal_breaker_count,
        "total_evaluated": valid,
        "total_models": configured_count,
        "deal_breaker_details": deal_breaker_details,
        "majority_threshold": valid / 2 if valid else 0.0,
        "configured": counts["configured"],
        "attempted": counts["attempted"],
        "valid": valid,
        "technical_failed": counts["technical_failed"],
        "invalid": counts["invalid"],
        "skipped": counts["skipped"],
        "semantic_deal_breakers": deal_breaker_count,
        "required_valid": counts["required_valid"],
        "required_majority": counts["required_majority"],
    }


def guaranteed_deal_breaker_majority(
    layer_results: Dict[str, Any],
    configured_count: int,
    remaining_possible_valid: int,
    *,
    required_valid: Optional[int] = None,
) -> bool:
    """Return True if pending responses cannot overturn a DB majority."""

    counts = build_qa_counts(
        layer_results,
        configured_count,
        required_valid=required_valid,
    )
    if counts["valid"] < counts["required_valid"]:
        return False
    max_final_valid = counts["valid"] + max(0, remaining_possible_valid)
    if max_final_valid <= 0:
        return False
    return counts["semantic_deal_breakers"] > (max_final_valid / 2)
