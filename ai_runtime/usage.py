"""Provider usage and finish-reason normalization helpers."""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# ``cost_basis`` value of a usage payload whose ``cost`` is the amount the
# provider reported as billed, as opposed to a catalog estimate.
PROVIDER_BILLED_COST_BASIS = "provider_billed"

TOKEN_LIMIT_REASONS_BY_PROVIDER = {
    "openai": {"length", "max_output_tokens", "max_completion_tokens"},
    "openrouter": {"length"},
    "xai": {"length", "max_output_tokens"},
    "moonshot": {"length"},
    "minimax": {"length"},
    "claude": {"max_tokens"},
    "gemini": {"max_tokens"},
    "ollama": {"length"},
}

SUCCESS_REASONS_BY_PROVIDER = {
    "openai": {"stop", "completed"},
    "openrouter": {"stop"},
    "xai": {"stop", "completed"},
    "moonshot": {"stop"},
    "minimax": {"stop"},
    "claude": {"end_turn", "stop_sequence"},
    "gemini": {"stop"},
    "ollama": {"stop"},
    "fake": {"stop"},
}

TOOL_CALL_REASONS_BY_PROVIDER = {
    "openai": {"tool_calls", "function_call"},
    "openrouter": {"tool_calls"},
    "xai": {"tool_calls", "function_call"},
    "moonshot": {"tool_calls"},
    "minimax": {"tool_calls"},
    "claude": {"tool_use"},
    "gemini": {"malformed_function_call", "unexpected_tool_call"},
}

CONTENT_FILTER_REASONS_BY_PROVIDER = {
    "openai": {"content_filter"},
    "openrouter": {"content_filter"},
    "xai": {"content_filter"},
    "moonshot": {"content_filter"},
    "minimax": {"content_filter"},
    "claude": {"refusal"},
    "gemini": {
        "safety",
        "recitation",
        "language",
        "blocklist",
        "prohibited_content",
        "spii",
        "image_safety",
    },
}

CONTEXT_LIMIT_REASONS_BY_PROVIDER = {
    "openai": {"max_prompt_tokens"},
    "claude": {"model_context_window_exceeded"},
}

PAUSE_REASONS_BY_PROVIDER = {
    "claude": {"pause_turn"},
}

ERROR_REASONS_BY_PROVIDER = {
    "openai": {"incomplete", "failed", "cancelled"},
    "openrouter": {"error"},
    "xai": {"incomplete"},
    "gemini": {"other", "finish_reason_unspecified"},
}

PROVIDER_ALIASES = {
    "anthropic": "claude",
    "google": "gemini",
    "grok": "xai",
    "kimi": "moonshot",
    "kimi_api": "moonshot",
}


def stringify_finish_reason(value: Any) -> Optional[str]:
    """Return provider finish/stop reasons as stable strings."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    value_attr = getattr(value, "value", None)
    if isinstance(value_attr, str):
        return value_attr
    return str(value)


def normalize_provider_key(provider: Optional[str]) -> Optional[str]:
    """Return the provider family used for finish-reason classification."""

    if provider is None:
        return None
    key = str(provider or "").strip().lower().replace("-", "_")
    return PROVIDER_ALIASES.get(key, key)


def normalize_finish_reason(value: Any) -> Optional[str]:
    """Normalize provider finish reasons for exact table lookup."""

    reason_text = (stringify_finish_reason(value) or "").strip().lower()
    if not reason_text:
        return None
    return reason_text.replace("-", "_").replace(" ", "_")


def _provider_reason_set(
    table: dict[str, set[str]],
    provider: Optional[str],
) -> set[str]:
    provider_key = normalize_provider_key(provider)
    if provider_key and provider_key in table:
        return table[provider_key]
    return set()


def is_token_limit_finish_reason(reason: Any, provider: Optional[str] = None) -> bool:
    """Detect provider stop reasons that mean output was cut by token budget."""

    normalized = normalize_finish_reason(reason)
    if not normalized:
        return False
    return normalized in _provider_reason_set(TOKEN_LIMIT_REASONS_BY_PROVIDER, provider)


def classify_finish_reason(
    *,
    provider: Optional[str],
    finish_reason: Any,
) -> dict[str, Any]:
    """Classify provider finish/stop reasons using explicit provider tables."""

    finish_reason_text = stringify_finish_reason(finish_reason)
    normalized = normalize_finish_reason(finish_reason)
    provider_key = normalize_provider_key(provider)
    result: dict[str, Any] = {
        "finish_reason": finish_reason_text,
        "finish_reason_normalized": normalized,
        "finish_reason_category": "missing" if normalized is None else "unknown",
        "finish_reason_known": False,
        "finish_unusable": False,
        "output_truncated": False,
    }
    if normalized is None:
        return result

    if normalized in _provider_reason_set(TOKEN_LIMIT_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "output_token_limit",
                "finish_reason_known": True,
                "finish_unusable": True,
                "output_truncated": True,
                "truncation_reason": "output_token_limit",
            }
        )
        return result

    if normalized in _provider_reason_set(SUCCESS_REASONS_BY_PROVIDER, provider_key):
        result.update({"finish_reason_category": "stop", "finish_reason_known": True})
        return result

    if normalized in _provider_reason_set(TOOL_CALL_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "tool_calls",
                "finish_reason_known": True,
                "finish_unusable": True,
            }
        )
        return result

    if normalized in _provider_reason_set(CONTENT_FILTER_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "content_filter",
                "finish_reason_known": True,
                "finish_unusable": True,
            }
        )
        return result

    if normalized in _provider_reason_set(CONTEXT_LIMIT_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "context_limit",
                "finish_reason_known": True,
                "finish_unusable": True,
            }
        )
        return result

    if normalized in _provider_reason_set(PAUSE_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "pause_turn",
                "finish_reason_known": True,
                "finish_unusable": True,
            }
        )
        return result

    if normalized in _provider_reason_set(ERROR_REASONS_BY_PROVIDER, provider_key):
        result.update(
            {
                "finish_reason_category": "provider_error",
                "finish_reason_known": True,
                "finish_unusable": True,
            }
        )
        return result

    return result


def is_unusable_openai_stream_finish(reason: Any) -> bool:
    """Return True when an OpenAI streamed turn ended with unusable partial output."""

    if normalize_finish_reason(reason) is None:
        return True
    classification = classify_finish_reason(provider="openai", finish_reason=reason)
    return classification.get("finish_reason_category") not in {"stop", "tool_calls"}


def build_finish_metadata(
    *,
    provider: str,
    finish_reason: Any,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Build lightweight streaming finish metadata."""

    finish_reason_text = stringify_finish_reason(finish_reason)
    classification = classify_finish_reason(
        provider=provider,
        finish_reason=finish_reason_text,
    )
    metadata: dict[str, Any] = {
        "provider": provider,
        "output_truncated": bool(classification.get("output_truncated")),
    }
    if finish_reason_text is not None:
        metadata["finish_reason"] = finish_reason_text
        metadata["provider_stop_reason"] = finish_reason_text
    metadata["finish_reason_category"] = classification["finish_reason_category"]
    metadata["finish_reason_known"] = classification["finish_reason_known"]
    metadata["finish_unusable"] = classification["finish_unusable"]
    if max_tokens is not None:
        metadata["max_tokens"] = max_tokens
    if metadata["output_truncated"]:
        metadata["truncation_reason"] = "output_token_limit"
    return metadata


def _prompt_block_metadata(reason: Any) -> dict[str, Any]:
    """Classify Gemini's prompt verdict without inventing a candidate finish."""
    normalized = normalize_finish_reason(reason)
    if not normalized or normalized in {"block_reason_unspecified", "blocked_reason_unspecified"}:
        return {}
    classification = classify_finish_reason(provider="gemini", finish_reason=reason)
    return {
        "prompt_block_reason": stringify_finish_reason(reason),
        "finish_reason_category": classification["finish_reason_category"],
        "finish_unusable": True,
    }


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce a provider token count to ``int`` without ever raising.

    Providers occasionally report token fields as numeric strings
    (``"12.5"``) or unexpected structures; usage telemetry must never
    break an otherwise successful generation, so malformed values
    degrade to ``None`` with a warning instead of propagating
    ``ValueError``/``TypeError`` (CFG-02).
    """

    if value is None:
        return None
    try:
        if isinstance(value, str):
            return int(float(value.strip()))
        if isinstance(value, (int, float)):
            return int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    logger.warning("normalize_usage: dropping malformed token count %r", value)
    return None


def _usage_field(obj: Any, name: str) -> Any:
    """Read one provider usage field from a mapping or an SDK object."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _money_amount(value: Any) -> Optional[Decimal]:
    """Finite non-negative money amount, or ``None`` (mirrors provider_budget._decimal).

    Amounts stay ``Decimal`` until the final result so that summing the account
    and upstream charges never carries binary float error into the billed cost.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return amount


# Providers whose usage object carries the amount billed to the account under
# the accounting contract ``provider_budget.ProviderBudget.settle`` trusts.
# TypeSafe usage is forwarded verbatim from its API and settle prices it from
# the verified tariff, so a ``cost`` field there is never a billed amount; the
# OpenAI-compatible and local adapters report no billing at all.
PROVIDER_BILLED_COST_PROVIDERS = frozenset({"openrouter"})


def provider_billed_cost(usage_obj: Any, billing_envelope: Any = None) -> Optional[float]:
    """Return the USD amount billed to the account for one call, when reported.

    This follows ``provider_budget.ProviderBudget.settle`` exactly. OpenRouter
    reports ``usage.cost`` (the credits it charged) and, for BYOK requests,
    ``usage.cost_details.upstream_inference_cost`` (what the upstream provider
    charged the account's own key on top). The billed amount is known only when
    ``cost`` is reported and either ``is_byok`` is explicitly false or the
    upstream charge is reported too (OpenRouter sends ``0``/``null`` for
    non-BYOK calls, so the sum is exact either way). ``is_byok`` is read from
    the usage first and then from ``billing_envelope``, the object settle
    received as ``response``: the completion for a plain call, the usage-bearing
    chunk for a stream. Any other shape is not a billed amount and the catalog
    estimate stays in charge.
    """
    account_cost = _money_amount(_usage_field(usage_obj, "cost"))
    if account_cost is None:
        return None
    byok = _usage_field(usage_obj, "is_byok")
    if type(byok) is not bool:
        byok = _usage_field(billing_envelope, "is_byok")
    if byok is False:
        return float(account_cost)
    upstream_cost = _money_amount(
        _usage_field(_usage_field(usage_obj, "cost_details"), "upstream_inference_cost")
    )
    if upstream_cost is None:
        return None
    return float(account_cost + upstream_cost)


def attach_provider_billed_cost(
    usage_payload: dict[str, Any],
    usage_obj: Any,
    *,
    provider: str,
    billing_envelope: Any = None,
) -> None:
    """Add ``cost``/``cost_basis`` to a normalized payload the provider billed.

    Only a provider in ``PROVIDER_BILLED_COST_PROVIDERS`` can report a billed
    amount; every other provider keeps the catalog estimate whatever fields its
    raw usage carries. A payload that already resolved its billed cost (a
    canonical payload normalized a second time) is left alone.
    """
    if "cost" in usage_payload or provider not in PROVIDER_BILLED_COST_PROVIDERS:
        return
    billed_cost = provider_billed_cost(usage_obj, billing_envelope)
    if billed_cost is not None:
        usage_payload["cost"] = billed_cost
        usage_payload["cost_basis"] = PROVIDER_BILLED_COST_BASIS


def normalize_usage(usage_obj: Any) -> Optional[dict[str, Any]]:
    """Extract token metrics, retaining ``None`` for unreported measurements."""

    if usage_obj is None:
        return None

    def _pluck(obj: Any, *names: str) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                value = obj[name]
                if value is not None:
                    return value
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    def _sum_native_counts(*values: Any) -> Optional[int]:
        # Gemini's optional tool/reasoning components augment its base counts.
        # No reported component means no measurement, rather than zero tokens.
        counts = [_coerce_int(value) for value in values]
        if all(count is None for count in counts):
            return None
        return sum(count or 0 for count in counts)

    input_tokens = _pluck(usage_obj, "prompt_tokens", "input_tokens", "input_token_count")
    if input_tokens is None:
        input_tokens = _sum_native_counts(
            _pluck(usage_obj, "prompt_token_count"),
            _pluck(usage_obj, "tool_use_prompt_token_count"),
        )
    output_tokens = _pluck(usage_obj, "completion_tokens", "output_tokens", "output_token_count")
    if output_tokens is None:
        # Gemini reports candidates and thoughts separately. UsageTracker bills
        # output_tokens, with reasoning_tokens retained only as a breakdown.
        # Combine native fields once; normalized payloads already include them.
        output_tokens = _sum_native_counts(
            _pluck(usage_obj, "candidates_token_count"),
            _pluck(usage_obj, "thoughts_token_count"),
        )
    total_tokens = _pluck(usage_obj, "total_tokens", "total_token_count")
    reasoning_tokens = _pluck(usage_obj, "reasoning_tokens", "thinking_tokens", "thoughts_token_count")
    finish_reason = None
    for name in ("finish_reason", "stop_reason", "provider_stop_reason"):
        if isinstance(usage_obj, dict) and usage_obj.get(name) is not None:
            finish_reason = usage_obj.get(name)
            break
        if hasattr(usage_obj, name):
            finish_reason = getattr(usage_obj, name)
            break
    output_truncated = None
    if isinstance(usage_obj, dict) and "output_truncated" in usage_obj:
        output_truncated = bool(usage_obj.get("output_truncated"))

    if hasattr(usage_obj, "model_dump"):
        try:
            dumped = usage_obj.model_dump()
            if isinstance(dumped, dict):
                if dumped.get("input_tokens") is not None:
                    input_tokens = dumped["input_tokens"]
                if dumped.get("output_tokens") is not None:
                    output_tokens = dumped["output_tokens"]
                total_tokens = dumped.get("total_tokens", total_tokens)
                finish_reason = finish_reason or dumped.get("finish_reason") or dumped.get("stop_reason")
        except Exception:
            pass

    normalized: dict[str, Any] = {
        "input_tokens": _coerce_int(input_tokens),
        "output_tokens": _coerce_int(output_tokens),
        "total_tokens": _coerce_int(total_tokens),
        "reasoning_tokens": _coerce_int(reasoning_tokens),
    }
    input_details = _pluck(usage_obj, "input_tokens_details", "prompt_tokens_details") or {}
    for field, native_name, anthropic_name in (
        ("cached_input_tokens", "cached_tokens", "cache_read_input_tokens"),
        ("cache_write_input_tokens", "cache_write_tokens", "cache_creation_input_tokens"),
    ):
        count = _pluck(usage_obj, field)
        if count is None:
            count = _pluck(input_details, native_name)
        if count is None:
            count = _pluck(usage_obj, anthropic_name)
        if count is not None:
            normalized[field] = _coerce_int(count)
    # Anthropic's native input_tokens excludes both cache categories. Canonical
    # payloads retain only our names, so a second normalization cannot add them
    # again. OpenAI's nested cache details already belong to its input total.
    native_cache_counts = [
        _coerce_int(count)
        for name in ("cache_read_input_tokens", "cache_creation_input_tokens")
        if (count := _pluck(usage_obj, name)) is not None
    ]
    if native_cache_counts:
        if normalized["input_tokens"] is None or any(count is None for count in native_cache_counts):
            normalized["input_tokens"] = None
        else:
            normalized["input_tokens"] += sum(native_cache_counts)
    finish_reason_text = stringify_finish_reason(finish_reason)
    if finish_reason_text is not None:
        normalized["finish_reason"] = finish_reason_text
        normalized["provider_stop_reason"] = finish_reason_text
    provider = None
    if isinstance(usage_obj, dict):
        provider = usage_obj.get("provider")
    elif hasattr(usage_obj, "provider"):
        provider = getattr(usage_obj, "provider")

    classification = classify_finish_reason(
        provider=provider,
        finish_reason=finish_reason_text,
    )
    output_truncated = bool(output_truncated) or bool(classification.get("output_truncated"))
    normalized["output_truncated"] = bool(output_truncated)
    normalized["finish_reason_category"] = classification["finish_reason_category"]
    normalized["finish_reason_known"] = classification["finish_reason_known"]
    normalized["finish_unusable"] = classification["finish_unusable"]
    if normalized["output_truncated"]:
        normalized["truncation_reason"] = "output_token_limit"
    if isinstance(usage_obj, dict):
        normalized.update(_prompt_block_metadata(usage_obj.get("prompt_block_reason")))
    native_finish = (usage_obj.get("native_finish_reason") if isinstance(usage_obj, dict)
                     else getattr(usage_obj, "native_finish_reason", None))
    if native_finish is not None:
        normalized["native_finish_reason"] = stringify_finish_reason(native_finish)
    if _usage_field(usage_obj, "cost_basis") == PROVIDER_BILLED_COST_BASIS:
        # A canonical payload already resolved its billed amount for its
        # provider (attach_provider_billed_cost); the second normalization on
        # the usage-callback path keeps it. A raw provider usage never carries
        # cost_basis, so its cost is never adopted here without the provider.
        canonical_cost = _money_amount(_usage_field(usage_obj, "cost"))
        if canonical_cost is not None:
            normalized["cost"] = float(canonical_cost)
            normalized["cost_basis"] = PROVIDER_BILLED_COST_BASIS
    return normalized


def merge_anthropic_stream_usage(
    captured: dict[str, Any], usage_obj: Any
) -> None:
    """Retain native cumulative counters from stream start and delta events.

    Deltas usually report only output; they must not replace the input/cache
    measurements from message_start. Missing measurements stay unavailable,
    and reported zero remains a measurement. Normalize only after merging.
    """
    for field in (
        "input_tokens", "output_tokens", "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        value = _usage_field(usage_obj, field)
        if value is not None:
            captured[field] = value


def usage_with_finish_metadata(
    usage_obj: Any,
    response: Any,
    *,
    provider: str,
    max_tokens: Optional[int] = None,
    fallback_finish_reason: Any = None,
    fallback_native_finish_reason: Any = None,
    billing_envelope: Any = None,
) -> dict[str, Any]:
    """Combine token usage with provider stop/finish reason metadata.

    ``response`` only feeds the finish-reason derivation. ``billing_envelope``
    is the object ``provider_budget.settle`` received for the same call (the
    completion, or a stream's usage-bearing chunk) and only feeds the billed
    cost, so both ledgers read the BYOK flag from the same place.
    """

    usage_payload = normalize_usage(usage_obj) or {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
    }
    attach_provider_billed_cost(
        usage_payload, usage_obj, provider=provider, billing_envelope=billing_envelope,
    )
    if isinstance(usage_obj, dict):
        completion_details = usage_obj.get("completion_tokens_details")
    else:
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
    if isinstance(completion_details, dict):
        native_reasoning = completion_details.get("reasoning_tokens")
    else:
        native_reasoning = getattr(completion_details, "reasoning_tokens", None)
    native_reasoning = _coerce_int(native_reasoning)
    if native_reasoning is not None:
        usage_payload["reasoning_tokens"] = native_reasoning
        # xAI excludes reasoning from completion_tokens; OpenAI includes it.
        # Canonical payloads have no native details, so repeated normalization
        # preserves the billable output count without adding reasoning again.
        if provider == "xai" and usage_payload["output_tokens"] is not None:
            usage_payload["output_tokens"] += native_reasoning
    native_finish = fallback_native_finish_reason
    if native_finish is None:
        choices = getattr(response, "choices", None) or []
        if choices:
            native_finish = getattr(choices[0], "native_finish_reason", None)
    if native_finish is not None:
        usage_payload["native_finish_reason"] = stringify_finish_reason(native_finish)
    finish_reason = fallback_finish_reason

    if finish_reason is None:
        finish_reason = getattr(response, "stop_reason", None)
    if finish_reason is None:
        finish_reason = getattr(response, "finish_reason", None)
    if finish_reason is None:
        choices = getattr(response, "choices", None) or []
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
    if finish_reason is None:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        incomplete_details = getattr(response, "incomplete_details", None)
        if incomplete_details is not None:
            finish_reason = getattr(incomplete_details, "reason", None)
            if finish_reason is None and isinstance(incomplete_details, dict):
                finish_reason = incomplete_details.get("reason")
    if finish_reason is None:
        status = getattr(response, "status", None)
        status_text = normalize_finish_reason(status)
        if status_text and status_text not in {"completed", "complete", "stop"}:
            finish_reason = status

    finish_reason_text = stringify_finish_reason(finish_reason)
    if finish_reason_text:
        usage_payload["finish_reason"] = finish_reason_text
        usage_payload["provider_stop_reason"] = finish_reason_text
    usage_payload["provider"] = provider
    if max_tokens is not None:
        usage_payload["max_tokens"] = max_tokens
    classification = classify_finish_reason(
        provider=provider,
        finish_reason=finish_reason_text,
    )
    usage_payload["finish_reason_category"] = classification["finish_reason_category"]
    usage_payload["finish_reason_known"] = classification["finish_reason_known"]
    usage_payload["finish_unusable"] = classification["finish_unusable"]
    if classification.get("output_truncated"):
        usage_payload["output_truncated"] = True
        usage_payload["truncation_reason"] = "output_token_limit"
    else:
        usage_payload.setdefault("output_truncated", False)
    block_reason = usage_payload.get("prompt_block_reason")
    if normalize_provider_key(provider) == "gemini":
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None:
            block_reason = getattr(feedback, "block_reason", None) or block_reason
    usage_payload.update(_prompt_block_metadata(block_reason))
    return usage_payload
