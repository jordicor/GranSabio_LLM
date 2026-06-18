"""Pure result and JSON guard helpers for content generation."""

from __future__ import annotations

from typing import Any, Callable, Optional

import json_utils as json


def run_json_post_guard(
    content: str,
    *,
    schema: Optional[dict[str, Any]] = None,
    expectations: Optional[list[dict[str, Any]]] = None,
    validate_ai_json_fn: Callable[..., Any],
    validate_loose_json_fn: Callable[..., Any],
) -> Any:
    """Validate generated JSON using the same loose/structured split as tools."""

    if schema is not None:
        return validate_ai_json_fn(content, schema=schema, expectations=expectations)
    return validate_loose_json_fn(content, expectations=expectations)


def json_guard_data_to_content(result: Any) -> Optional[str]:
    """Serialize the parsed JSON payload after a successful guard pass."""

    if result.data is None:
        return None
    return json.dumps(result.data, ensure_ascii=False)


def generation_was_truncated(session: dict[str, Any]) -> bool:
    """Return True when provider metadata says the output hit token limit."""

    finish_metadata = session.get("generation_finish_metadata") or {}
    if isinstance(finish_metadata, dict) and finish_metadata.get("output_truncated"):
        return True
    tool_metadata = session.get("generation_tool_metadata")
    if isinstance(tool_metadata, dict) and tool_metadata.get("output_truncated"):
        return True
    if hasattr(tool_metadata, "output_truncated"):
        return bool(getattr(tool_metadata, "output_truncated"))
    return False


def build_truncation_failure_reason(session: dict[str, Any], request: Any) -> str:
    """Build a user/actionable reason for output-token truncation."""

    def _metadata_get(metadata: Any, key: str) -> Any:
        if isinstance(metadata, dict):
            return metadata.get(key)
        if hasattr(metadata, key):
            return getattr(metadata, key)
        return None

    finish_metadata = session.get("generation_finish_metadata") or {}
    tool_metadata = session.get("generation_tool_metadata")
    stop_reason = None
    if finish_metadata:
        stop_reason = _metadata_get(finish_metadata, "finish_reason") or _metadata_get(
            finish_metadata,
            "provider_stop_reason",
        )
    if stop_reason is None and tool_metadata is not None:
        stop_reason = _metadata_get(tool_metadata, "finish_reason") or _metadata_get(
            tool_metadata,
            "provider_stop_reason",
        )
    max_tokens = (
        _metadata_get(finish_metadata, "max_tokens")
        or _metadata_get(tool_metadata, "max_tokens")
        or getattr(request, "max_tokens", None)
    )
    details = []
    if stop_reason:
        details.append(f"stop_reason={stop_reason}")
    if max_tokens:
        details.append(f"max_tokens={max_tokens}")
    suffix = f" ({', '.join(details)})" if details else ""
    return (
        "Generation output was truncated because the provider exhausted the "
        f"output token budget{suffix}. Regenerate a shorter response or increase max_tokens."
    )


def get_final_content(
    session: dict[str, Any],
    raw_content: str,
    request: Any,
    *,
    is_json_output_requested_fn: Callable[[Any], bool],
    is_gran_sabio: bool = False,
) -> Any:
    """Return parsed JSON content when available, otherwise raw content."""

    if is_json_output_requested_fn(request):
        parsed_key = "gran_sabio_json_parsed_content" if is_gran_sabio else "json_parsed_content"
        parsed_content = session.get(parsed_key)
        if parsed_content is not None:
            return parsed_content
    return raw_content
