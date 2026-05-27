"""Provider usage and finish-reason normalization helpers."""

from __future__ import annotations

from typing import Any, Optional


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


def is_token_limit_finish_reason(reason: Any) -> bool:
    """Detect provider stop reasons that mean output was cut by token budget."""

    reason_text = (stringify_finish_reason(reason) or "").strip().lower()
    if not reason_text:
        return False
    normalized = reason_text.replace("-", "_").replace(" ", "_")
    if normalized in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "max_token",
        "token_limit",
        "output_token_limit",
        "max_tokens_exceeded",
        "max_output_tokens_exceeded",
    }:
        return True
    return "max" in normalized and "token" in normalized


def is_unusable_openai_stream_finish(reason: Any) -> bool:
    """Return True when an OpenAI streamed turn ended with unusable partial output."""

    reason_text = (stringify_finish_reason(reason) or "").strip().lower()
    if not reason_text:
        return True
    normalized = reason_text.replace("-", "_").replace(" ", "_")
    if normalized in {"stop", "tool_calls", "function_call", "completed"}:
        return False
    return True


def build_finish_metadata(
    *,
    provider: str,
    finish_reason: Any,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Build lightweight streaming finish metadata."""

    finish_reason_text = stringify_finish_reason(finish_reason)
    metadata: dict[str, Any] = {
        "provider": provider,
        "output_truncated": is_token_limit_finish_reason(finish_reason_text),
    }
    if finish_reason_text is not None:
        metadata["finish_reason"] = finish_reason_text
        metadata["provider_stop_reason"] = finish_reason_text
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
    if output_truncated is None:
        output_truncated = is_token_limit_finish_reason(finish_reason_text)
    normalized["output_truncated"] = bool(output_truncated)
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

    finish_reason_text = stringify_finish_reason(finish_reason)
    if finish_reason_text:
        usage_payload["finish_reason"] = finish_reason_text
        usage_payload["provider_stop_reason"] = finish_reason_text
    usage_payload["provider"] = provider
    if max_tokens is not None:
        usage_payload["max_tokens"] = max_tokens
    if is_token_limit_finish_reason(finish_reason_text):
        usage_payload["output_truncated"] = True
        usage_payload["truncation_reason"] = "output_token_limit"
    else:
        usage_payload.setdefault("output_truncated", False)
    return usage_payload
