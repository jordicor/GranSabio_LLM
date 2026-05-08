"""Shared helpers for QA result health and deal-breaker quorum decisions."""

from __future__ import annotations

import math
import hashlib
from typing import Any, Dict, Iterable, List, Mapping, Optional

from model_aliasing import get_evaluator_alias

TECHNICAL_QA_ERROR_TYPES = frozenset(
    {
        "api_failure",
        "parse_error",
        "timeout",
        "unexpected",
        "technical_failure",
        "model_unavailable",
    }
)

RETRYABLE_PROVIDER_FAILURE_KINDS = frozenset(
    {
        "transient_network",
        "timeout",
        "rate_limited",
        "provider_overloaded",
        "provider_down",
    }
)

NON_RETRYABLE_PROVIDER_FAILURE_KINDS = frozenset(
    {
        "auth_invalid",
        "permission_denied",
        "billing_required",
        "quota_exhausted",
        "invalid_request",
        "unsupported_parameter",
        "unsupported_model",
        "schema_invalid",
        "context_overflow",
        "content_policy",
        "malformed_response",
        "parse_failed",
        "no_content",
        "cancelled",
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


def _stringify_metadata_value(value: Any) -> Optional[str]:
    """Return a compact stable string for provider/debug metadata fields."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_mapping_or_attr(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _provider_failure_from_exception(exc: Any) -> Any:
    """Best-effort discovery of the future ProviderFailure shape without importing it."""

    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        provider_failure = getattr(current, "provider_failure", None)
        if provider_failure is not None:
            return provider_failure
        if getattr(current, "kind", None) is not None:
            return current
        current = getattr(current, "cause", None) or getattr(current, "__cause__", None)
    return None


def provider_failure_kind(exc: Any) -> Optional[str]:
    """Return normalized provider failure kind when the new taxonomy is present."""

    provider_failure = _provider_failure_from_exception(exc)
    kind = _read_mapping_or_attr(provider_failure, "kind")
    if kind is None:
        kind = _read_mapping_or_attr(exc, "kind")
    value = getattr(kind, "value", kind)
    return _stringify_metadata_value(value)


def _extract_provider_request_id(exc: Any) -> Optional[str]:
    for attr in ("request_id", "response_id", "id", "correlation_id"):
        value = _stringify_metadata_value(getattr(exc, attr, None))
        if value:
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("request_id", "id", "correlation_id"):
            value = _stringify_metadata_value(getattr(response, attr, None))
            if value:
                return value
    return None


def extract_provider_request_id(exc: Any) -> Optional[str]:
    """Find an upstream request/correlation id from wrappers, causes, or ProviderFailure."""

    provider_failure = _provider_failure_from_exception(exc)
    for attr in ("request_id", "correlation_id"):
        value = _stringify_metadata_value(_read_mapping_or_attr(provider_failure, attr))
        if value:
            return value

    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = _extract_provider_request_id(current)
        if value:
            return value
        current = getattr(current, "cause", None) or getattr(current, "__cause__", None)
    return None


def derive_qa_correlation_id(
    *,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    layer_name: Optional[str] = None,
    slot_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    error_type: Optional[str] = None,
    original_exception_class: Optional[str] = None,
    upstream_request_id: Optional[str] = None,
) -> str:
    """Build a deterministic id when no provider request id is available."""

    if upstream_request_id:
        return upstream_request_id
    parts = [
        session_id,
        project_id,
        layer_name,
        slot_id,
        provider,
        model,
        error_type,
        original_exception_class,
    ]
    digest = hashlib.sha256("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()
    return f"qa-{digest[:16]}"


def normalize_qa_technical_failure_metadata(
    *,
    error_type: str,
    message: Optional[str] = None,
    retryable: bool = False,
    attempts: Optional[int] = None,
    max_attempts: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    layer_name: Optional[str] = None,
    slot_id: Optional[str] = None,
    slot_index: Optional[int] = None,
    evaluator_alias: Optional[str] = None,
    configured_count: Optional[int] = None,
    required_valid: Optional[int] = None,
    policy: Optional[Any] = None,
    original_exception: Optional[BaseException] = None,
    extra: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the canonical QA technical-failure/debug metadata payload."""

    provider_failure = _provider_failure_from_exception(original_exception)
    provider_kind = provider_failure_kind(original_exception)
    provider = provider or _stringify_metadata_value(
        _read_mapping_or_attr(provider_failure, "provider")
        or getattr(original_exception, "provider", None)
    )
    model = model or _stringify_metadata_value(
        _read_mapping_or_attr(provider_failure, "model_id")
        or _read_mapping_or_attr(provider_failure, "model")
        or getattr(original_exception, "model", None)
    )
    upstream_request_id = extract_provider_request_id(original_exception)
    original_class = type(original_exception).__name__ if original_exception is not None else None

    if provider_kind in RETRYABLE_PROVIDER_FAILURE_KINDS:
        retryable = True
    elif provider_kind in NON_RETRYABLE_PROVIDER_FAILURE_KINDS:
        retryable = False

    metadata: Dict[str, Any] = {
        "technical_failure": True,
        "technical_failure_category": "qa_evaluator_failed",
        "error_category": "technical",
        "error_type": error_type,
        "retryable": bool(retryable),
        "attempts": attempts,
        "max_attempts": max_attempts,
        "provider": provider,
        "model": model,
        "layer": layer_name,
        "slot_id": slot_id,
        "slot_index": slot_index,
        "evaluator_alias": evaluator_alias,
        "configured_count": configured_count,
        "required_valid": required_valid,
        "provider_failure_kind": provider_kind,
        "provider_error_type": _read_mapping_or_attr(provider_failure, "provider_error_type"),
        "provider_error_code": _read_mapping_or_attr(provider_failure, "provider_error_code"),
        "provider_error_param": _read_mapping_or_attr(provider_failure, "provider_error_param"),
        "original_exception_class": original_class,
        "correlation_id": derive_qa_correlation_id(
            session_id=session_id,
            project_id=project_id,
            layer_name=layer_name,
            slot_id=slot_id,
            provider=provider,
            model=model,
            error_type=error_type,
            original_exception_class=original_class,
            upstream_request_id=upstream_request_id,
        ),
        "provider_request_id": upstream_request_id,
        "debug_event": "qa_evaluator_failed",
    }
    if message:
        metadata["message"] = str(message)[:500]
    if policy is not None:
        metadata["scheduler_policy"] = {
            "execution_mode": getattr(policy, "execution_mode", None),
            "on_model_unavailable": getattr(policy, "on_model_unavailable", None),
            "on_timeout": getattr(policy, "on_timeout", None),
            "min_valid_models": getattr(policy, "min_valid_models", None),
            "min_valid_model_ratio": getattr(policy, "min_valid_model_ratio", None),
            "max_concurrency": getattr(policy, "max_concurrency", None),
            "timeout_retries": getattr(policy, "timeout_retries", None),
        }
    if extra:
        metadata.update(extra)

    return {key: value for key, value in metadata.items() if value is not None}


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
                    "evaluator": get_evaluator_alias(evaluation, fallback=None),
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
