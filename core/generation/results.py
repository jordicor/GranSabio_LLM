"""Pure result and JSON guard helpers for content generation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import json_utils as json
from model_aliasing import to_prompt_safe_data
from models import ContentRequest, GenerationStatus, is_json_output_requested
from usage_tracking import UsageTracker, inject_costs_into_json_payload, merge_costs_into_json_string


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
    """Return the current JSON candidate, never a previous parsed snapshot."""

    if is_json_output_requested_fn(request):
        try:
            return json.loads(raw_content)
        except (ValueError, TypeError):
            pass
    return raw_content


def attach_usage_metadata(session: Dict[str, Any], final_result: Dict[str, Any], request: ContentRequest) -> None:
    """Attach usage and cost metadata to the final payload.

    The ``costs``/``query_stats`` blocks are part of the default result
    contract (C15): ``show_query_costs``/``show_query_stats`` default to 1, and
    a client that does not want them sends 0. Embedding the cost block INSIDE
    the returned content stays opt-in, because that mutates canonical content:
    it only happens when the request set the field explicitly.
    Cancellation reports remain separate from accepted content and are marked
    partial: the interrupted provider call may not have delivered its receipt.
    """
    local_execution = session.get("local_execution")
    if local_execution is not None:
        from local_execution_policy import build_local_execution_attestation

        final_result["local_only_attestation"] = build_local_execution_attestation(
            local_execution.manifest,
            local_execution.ledger,
        ).to_payload()

    tracker: Optional[UsageTracker] = session.get("usage_tracker")
    level = getattr(request, "show_query_costs", 0)
    stats_level = getattr(request, "show_query_stats", 0)
    cancelled = final_result.get("status") == GenerationStatus.CANCELLED.value
    embed_costs_in_content = not cancelled and "show_query_costs" in (
        getattr(request, "model_fields_set", None) or set()
    )

    if not tracker or level <= 0:
        final_result.pop("costs", None)
    else:
        summary = tracker.build_summary(level)
        if not summary:
            final_result.pop("costs", None)
        else:
            alias_registry = getattr(request, "_model_alias_registry", None) or session.get("model_alias_registry")
            prompt_safe_summary = to_prompt_safe_data(summary, alias_registry)
            if cancelled:
                prompt_safe_summary["partial"] = True
            final_result["costs"] = prompt_safe_summary
            if embed_costs_in_content:
                content_value = final_result.get("content")
                if not is_json_output_requested(request):
                    if isinstance(content_value, str):
                        final_result["content"] = tracker.embed_text_summary(content_value, 1)
                elif isinstance(content_value, str):
                    final_result["content"] = merge_costs_into_json_string(content_value, prompt_safe_summary)
                elif content_value is not None:
                    final_result["content"] = inject_costs_into_json_payload(content_value, prompt_safe_summary)

    if not tracker or stats_level <= 0:
        final_result.pop("query_stats", None)
        return

    query_stats = tracker.build_query_stats(stats_level)
    if not query_stats:
        final_result.pop("query_stats", None)
        return

    if cancelled:
        query_stats["partial"] = True
    final_result["query_stats"] = query_stats
