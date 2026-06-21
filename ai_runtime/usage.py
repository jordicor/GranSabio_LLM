"""Provider usage and finish-reason normalization helpers."""

from __future__ import annotations

from typing import Any, Optional

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
    "openai": {"incomplete"},
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


def normalize_usage(usage_obj: Any) -> Optional[dict[str, Any]]:
    """Extract token metrics from provider-specific usage objects."""

    if usage_obj is None:
        return None

    def _pluck(obj: Any, *names: str) -> Optional[int]:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    input_tokens = _pluck(usage_obj, "prompt_tokens", "input_tokens", "input_token_count") or 0
    output_tokens = _pluck(usage_obj, "completion_tokens", "output_tokens", "output_token_count") or 0
    total_tokens = _pluck(usage_obj, "total_tokens", "total_token_count")
    reasoning_tokens = _pluck(usage_obj, "reasoning_tokens", "thinking_tokens")
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
                input_tokens = dumped.get("input_tokens", input_tokens) or input_tokens
                output_tokens = dumped.get("output_tokens", output_tokens) or output_tokens
                total_tokens = dumped.get("total_tokens", total_tokens)
                finish_reason = finish_reason or dumped.get("finish_reason") or dumped.get("stop_reason")
        except Exception:
            pass

    normalized: dict[str, Any] = {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
        "reasoning_tokens": int(reasoning_tokens) if reasoning_tokens is not None else None,
    }
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
    return normalized


def usage_with_finish_metadata(
    usage_obj: Any,
    response: Any,
    *,
    provider: str,
    max_tokens: Optional[int] = None,
    fallback_finish_reason: Any = None,
) -> dict[str, Any]:
    """Combine token usage with provider stop/finish reason metadata."""

    usage_payload = normalize_usage(usage_obj) or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": None,
        "reasoning_tokens": None,
    }
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
    return usage_payload
