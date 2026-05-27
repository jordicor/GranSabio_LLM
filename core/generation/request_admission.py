"""Request-admission helpers for the /generate route."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from model_capability_registry import (
    model_supports_long_text_section_tool_loop,
    resolve_model_capability_context,
)
from models import ContentRequest, PreflightIssue

MIN_ITERATION_TIMEOUT_SECONDS = 900
QA_LAYER_PADDING_SECONDS = 120
GRAN_SABIO_PADDING_SECONDS = 600
SESSION_TIMEOUT_CAP_SECONDS = 8 * 3600
DEFAULT_WORD_LIMIT_TOKEN_FLOOR = 8000
DEFAULT_MODEL_MAX_TOKENS_FALLBACK = 8192
ESTIMATED_TOKENS_PER_WORD = 2.2
LONG_FORM_TOKEN_BUFFER = 1024


def get_request_fields_set(request: Any) -> set:
    """Return the set of fields explicitly provided in the incoming request."""

    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(request, "__fields_set__", set())
    return set(fields_set or [])


def estimate_session_timeout(
    request: ContentRequest,
    reasoning_timeout_hint: Optional[int],
) -> int:
    """Estimate a realistic client-side timeout considering iterations and QA."""

    per_iteration = reasoning_timeout_hint or 0
    per_iteration = max(per_iteration, MIN_ITERATION_TIMEOUT_SECONDS)

    iteration_budget = per_iteration * max(1, request.max_iterations)
    qa_layers = len(request.qa_layers) if request.qa_layers else 0
    qa_budget = max(1, qa_layers) * QA_LAYER_PADDING_SECONDS

    total = iteration_budget + qa_budget
    if getattr(request, "gran_sabio_fallback", False):
        total += GRAN_SABIO_PADDING_SECONDS

    return int(min(max(total, MIN_ITERATION_TIMEOUT_SECONDS), SESSION_TIMEOUT_CAP_SECONDS))


def estimate_tokens_for_word_target(request: ContentRequest) -> Optional[int]:
    """Estimate a generation token budget from the requested word target."""

    target_words: Optional[int] = None
    if request.max_words:
        target_words = request.max_words
    elif request.min_words:
        target_words = request.min_words

    if not target_words:
        return None

    estimated_tokens = int(math.ceil(target_words * ESTIMATED_TOKENS_PER_WORD))
    estimated_tokens += LONG_FORM_TOKEN_BUFFER
    return max(DEFAULT_WORD_LIMIT_TOKEN_FLOOR, estimated_tokens)


def model_default_max_tokens(model_name: str, config_obj: Any) -> int:
    """Return model max output tokens from specs, falling back to 8192."""

    try:
        model_info = config_obj.get_model_info(model_name)
    except Exception:
        return DEFAULT_MODEL_MAX_TOKENS_FALLBACK

    value = model_info.get("output_tokens")
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MODEL_MAX_TOKENS_FALLBACK
    return tokens if tokens > 0 else DEFAULT_MODEL_MAX_TOKENS_FALLBACK


def apply_external_generation_min_tokens(
    request: ContentRequest,
    *,
    config_obj: Any,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    """Apply the public-generation min token floor to a request, if configured."""

    adjustment = config_obj.apply_external_generation_min_tokens(
        request.generator_model,
        request.max_tokens,
        getattr(request, "reasoning_effort", None),
        getattr(request, "thinking_budget_tokens", None),
    )
    if not adjustment.get("was_adjusted"):
        return None

    request.max_tokens = int(adjustment["adjusted_tokens"])
    request._external_generation_min_tokens_adjustment = adjustment
    logger.info(
        "Raised external generation max_tokens for %s: %s -> %s "
        "(floor=%s, source=%s, safe_limit=%s)",
        request.generator_model,
        adjustment.get("original_tokens"),
        adjustment.get("adjusted_tokens"),
        adjustment.get("min_tokens"),
        adjustment.get("source"),
        adjustment.get("safe_limit"),
    )
    return adjustment


def build_advisory(*, code: str, message: str, severity: str = "warning") -> PreflightIssue:
    """Create a non-blocking accepted-path advisory."""

    return PreflightIssue(
        code=code,
        severity=severity,
        message=message,
        blockers=False,
    )


def supports_long_text_generation_tools(model_name: str, config_obj: Any) -> bool:
    """Return whether the generator supports Long Text section-draft tool loops."""

    specs = getattr(config_obj, "model_specs", {}) or {}
    context = resolve_model_capability_context(model_name, specs)
    if context.status != "resolved":
        return False

    return model_supports_long_text_section_tool_loop(
        context.provider,
        context.model_id,
        specs,
        model_data=context.model_data,
    )


def clip_long_text_bound(
    value: int,
    *,
    request: ContentRequest,
    hard_cap_words: int,
    lower: bool,
) -> int:
    """Clip Long Text target bands against request bounds and the hard cap."""

    if lower:
        if request.min_words is not None:
            value = max(value, request.min_words)
        value = min(value, hard_cap_words)
        return max(1, value)

    if request.max_words is not None:
        value = min(value, request.max_words)
    value = min(value, hard_cap_words)
    if request.min_words is not None:
        value = max(value, request.min_words)
    return max(1, value)


def resolve_long_text_mode(
    request: ContentRequest,
    request_fields_set: set,
    *,
    config_obj: Any,
) -> tuple[Dict[str, Any], List[PreflightIssue]]:
    """Resolve request-level Long Text activation and derived word bands."""

    requested_mode = getattr(request, "long_text_mode", "auto")
    hard_cap_words = int(config_obj.LONG_TEXT_HARD_CAP_WORDS)
    user_set_max_iterations = "max_iterations" in request_fields_set
    explicit_min_words = "min_words" in request_fields_set and request.min_words is not None
    explicit_max_words = "max_words" in request_fields_set and request.max_words is not None
    advisories: List[PreflightIssue] = []

    resolved: Dict[str, Any] = {
        "enabled": False,
        "requested_mode": requested_mode,
        "activation_reason": "Long Text Mode disabled.",
        "derived_target_words": None,
        "target_min_words": None,
        "target_max_words": None,
        "emergency_min_words": None,
        "emergency_max_words": None,
        "hard_cap_words": hard_cap_words,
        "user_set_max_iterations": user_set_max_iterations,
    }

    if requested_mode == "off":
        resolved["activation_reason"] = "Long Text Mode explicitly disabled by the caller."
        return resolved, advisories

    derived_target_words: Optional[int] = None
    if request.min_words is not None and request.max_words is not None:
        derived_target_words = int(round((request.min_words + request.max_words) / 2.0))
    elif requested_mode == "on" and request.min_words is not None:
        derived_target_words = request.min_words
    elif requested_mode == "on" and request.max_words is not None:
        derived_target_words = request.max_words

    resolved["derived_target_words"] = derived_target_words

    if requested_mode == "auto":
        if not (explicit_min_words and explicit_max_words):
            reason = "Long Text Mode stayed off because auto mode requires both min_words and max_words."
            resolved["activation_reason"] = reason
            if request.min_words is not None or request.max_words is not None:
                advisories.append(
                    build_advisory(
                        code="long_text_auto_declined_one_sided_bounds",
                        message=reason,
                    )
                )
            return resolved, advisories

        if derived_target_words is None:
            resolved["activation_reason"] = "Long Text Mode stayed off because no derived target could be resolved."
            return resolved, advisories

        if derived_target_words < config_obj.LONG_TEXT_AUTO_MIN_WORDS:
            reason = (
                f"Long Text Mode stayed off because the derived target ({derived_target_words} words) "
                f"is below the auto-activation threshold ({config_obj.LONG_TEXT_AUTO_MIN_WORDS})."
            )
            resolved["activation_reason"] = reason
            advisories.append(
                build_advisory(
                    code="long_text_auto_declined_below_threshold",
                    message=reason,
                    severity="info",
                )
            )
            return resolved, advisories

    if requested_mode == "on" and derived_target_words is None:
        raise HTTPException(
            status_code=400,
            detail="long_text_mode='on' requires min_words, max_words, or both so a target can be derived.",
        )

    if derived_target_words is not None:
        if derived_target_words < config_obj.LONG_TEXT_AUTO_MIN_WORDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode requires a derived target between {config_obj.LONG_TEXT_AUTO_MIN_WORDS} "
                    f"and {hard_cap_words} words; got {derived_target_words}."
                ),
            )
        if derived_target_words > hard_cap_words:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode V1 is capped at {hard_cap_words} words. Split the request across "
                    "multiple runs instead of requesting a larger single document."
                ),
            )

    resolved["enabled"] = True
    resolved["activation_reason"] = (
        f"Long Text Mode enabled ({requested_mode}) with a derived target of {derived_target_words} words."
    )

    band_delta = max(
        1,
        int(round((derived_target_words or 0) * (config_obj.LONG_TEXT_TARGET_BAND_PERCENT / 100.0))),
    )
    emergency_delta = max(
        1,
        int(round((derived_target_words or 0) * (config_obj.LONG_TEXT_EMERGENCY_BAND_PERCENT / 100.0))),
    )

    resolved["target_min_words"] = clip_long_text_bound(
        (derived_target_words or 0) - band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["target_max_words"] = clip_long_text_bound(
        (derived_target_words or 0) + band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )
    resolved["emergency_min_words"] = clip_long_text_bound(
        (derived_target_words or 0) - emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["emergency_max_words"] = clip_long_text_bound(
        (derived_target_words or 0) + emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )

    return resolved, advisories
