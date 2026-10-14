"""
Shared utilities for sync and async GranSabio LLM clients.

Contains constants, helper classes, and pure computation functions
used by both GranSabioClient and AsyncGranSabioClient.
No HTTP calls are made from this module.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import logging
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------

DEFAULT_PROCESS_TIMEOUT_SECONDS = 12000
DEFAULT_STREAM_TIMEOUT_SECONDS = 12000
STREAM_TIMEOUT_GRACE_SECONDS = 0
RESULT_POLL_GRACE_SECONDS = 0
RESULT_POLL_INTERVAL_SECONDS = 5.0
STREAM_ACTIVITY_CHECK_SECONDS = 30.0

# Exponential backoff curve for transient polling errors in wait_for_result.
# Applied only to idempotent GETs (get_status / get_result). Ten attempts
# totalling ~7 minutes, enough to ride out VPN flaps, brief DNS failures, and
# momentary server overload without abandoning a session the server is still
# working on. POST /generate is not retried here because retrying it would start
# duplicate sessions.
POLL_RETRY_BACKOFF_SECONDS: Tuple[int, ...] = (5, 10, 20, 40, 60, 60, 60, 60, 60, 60)


# ---------------------------------------------------------------------------
# Idempotent generation submissions
# ---------------------------------------------------------------------------

# POST /generate deduplicates submissions that carry this header: the same ID
# with the same payload replays the original admission (same session_id)
# instead of starting a second session, for as long as the server keeps the
# entry (process-local registry, 24 h TTL by default).
SAAS_REQUEST_ID_HEADER = "X-GranSabio-SaaS-Request-ID"
SAAS_REQUEST_ID_MAX_CHARS = 128
_SAAS_REQUEST_ID_PUNCTUATION = frozenset("._:-")


def validate_saas_request_id(request_id: Any) -> str:
    """Return ``request_id`` when it satisfies the server's key grammar.

    The server accepts 1-128 ASCII letters, digits and ``._:-``. An invalid key
    is a caller bug, so it fails before any request is sent.
    """
    if (
        not isinstance(request_id, str)
        or not request_id
        or len(request_id) > SAAS_REQUEST_ID_MAX_CHARS
        or not all(
            character.isascii()
            and (character.isalnum() or character in _SAAS_REQUEST_ID_PUNCTUATION)
            for character in request_id
        )
    ):
        raise ValueError(
            "request_id must be 1-128 ASCII letters, digits or '._:-' characters."
        )
    return request_id


def parse_retry_after_seconds(headers: Any) -> Optional[float]:
    """Return a ``Retry-After`` header as non-negative seconds, or ``None``.

    Accepts delta-seconds and HTTP-date values. Anything absent, malformed or
    not a header mapping yields ``None``; callers keep their own backoff.
    """
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("Retry-After")
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():
        return float(int(value))
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _body_has_json_detail(body_text: str) -> bool:
    """Whether an error body is a JSON object with a ``detail`` member.

    The engine's HTTP errors use that shape; proxy and gateway error pages
    (HTML or empty bodies) do not. Callers use this flag, never the body
    itself, to tell an engine answer from a transport intermediary.
    """
    try:
        body = json.loads(body_text)
    except (TypeError, ValueError):
        return False
    return isinstance(body, dict) and "detail" in body


def generation_http_error_details(
    payload: Dict[str, Any],
    *,
    body_text: str,
    headers: Any,
    request_id: Optional[str],
) -> Dict[str, Any]:
    """Build non-sensitive details for a failed POST /generate.

    ``json_error_detail`` tells whether the engine itself answered;
    ``retry_after_seconds`` and ``request_id`` appear only when known.
    """
    details: Dict[str, Any] = {
        "payload_metadata": payload_error_metadata(payload),
        "json_error_detail": _body_has_json_detail(body_text),
    }
    retry_after = parse_retry_after_seconds(headers)
    if retry_after is not None:
        details["retry_after_seconds"] = retry_after
    if request_id is not None:
        details["request_id"] = request_id
    return details


def admission_rejection_error(init_response: Any) -> Optional[Exception]:
    """Return the error for an admission that preflight or Auto-QA rejected.

    ``generate()`` raises this ``GranSabioClientError`` (``provider_outcome``
    ``"rejected"``) instead of returning such a response. Reconciliation callers
    read the ``init_response`` of a ``get_request_status()`` snapshot with the
    same rule. Returns ``None`` for any other admission response.
    """
    from . import GranSabioClientError

    if not isinstance(init_response, Mapping):
        return None
    status = init_response.get("status")
    feedback = init_response.get("preflight_feedback")
    if status in {"preflight_rejected", "rejected"} and isinstance(feedback, Mapping) and feedback:
        return GranSabioClientError(
            f"Request rejected by preflight: {feedback.get('user_feedback', 'Unknown reason')}",
            details={"preflight_feedback": feedback, "response": init_response},
            provider_outcome="rejected",
        )
    feedback = init_response.get("auto_qa_feedback")
    if status == "auto_qa_rejected" and isinstance(feedback, Mapping) and feedback:
        return GranSabioClientError(
            f"Request rejected by Auto-QA: {feedback.get('message', 'Unknown reason')}",
            details={"auto_qa_feedback": feedback, "response": init_response},
            provider_outcome="rejected",
        )
    return None


def session_list_params(
    limit: Any, offset: Any, project_id: Optional[str],
) -> Dict[str, Any]:
    """Validate ``GET /debugger/sessions`` paging before any request is sent."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer.")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer.")
    params: Dict[str, Any] = {"limit": limit, "offset": offset}
    if project_id is not None:
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("project_id must be a non-empty string or None.")
        params["project_id"] = project_id
    return params


def payload_error_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build non-sensitive metadata about a request payload for error details.

    The full payload (which contains the user's prompt, potentially PII or
    confidential business data) must never be attached to exceptions, because
    consumers routinely ship exceptions to logs and telemetry. Only the key
    names, the prompt size, and a short hash for correlation are exposed.
    """
    prompt = payload.get("prompt")
    metadata: Dict[str, Any] = {"payload_keys": sorted(str(key) for key in payload)}
    if isinstance(prompt, str):
        metadata["prompt_chars"] = len(prompt)
        metadata["prompt_sha256_16"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return metadata

# Timeouts for reasoning models - significantly increased for multi-iteration reasoning.
# These models can iterate many times and take hours to complete.
REASONING_TIMEOUT_MAP: Dict[str, int] = {
    "minimal": 3600,      # 1 hour
    "low": 7200,          # 2 hours
    "medium": 14400,      # 4 hours
    "high": 28800,        # 8 hours
}

# ---------------------------------------------------------------------------
# Hardcoded fallback token limits for when /models is unreachable
# ---------------------------------------------------------------------------

MODEL_TOKEN_FALLBACKS: Dict[str, int] = {
    "grok-4": 128000,
    "gpt-5": 128000,
    "claude-opus-4": 128000,
    "claude-sonnet-4": 128000,
    "gemini-2.5": 65536,
    "o3": 100000,
}
DEFAULT_TOKEN_LIMIT = 16000


# ---------------------------------------------------------------------------
# ActivityMonitor - tracks last meaningful server activity during streaming
# ---------------------------------------------------------------------------

class ActivityMonitor:
    """Thread-safe tracker of last meaningful server activity during streaming.

    Used by streaming methods to extend deadlines when the server is still
    making progress (e.g., multi-iteration QA) even if no *completion* signal
    has arrived yet.
    """

    def __init__(self, inactivity_window: float) -> None:
        self._inactivity_window = max(float(inactivity_window or 0.0), 30.0)
        self._lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._last_iteration: Optional[int] = None
        self._last_message: Optional[str] = None

    def mark(self, *, iteration: Optional[int] = None, message: Optional[str] = None) -> None:
        """Record that meaningful activity was observed."""
        with self._lock:
            self._last_activity = time.monotonic()
            if iteration is not None:
                self._last_iteration = iteration
            if message:
                self._last_message = message.strip() or self._last_message

    def has_recent_activity(self) -> bool:
        with self._lock:
            return (time.monotonic() - self._last_activity) <= self._inactivity_window

    def last_activity_timestamp(self) -> float:
        with self._lock:
            return self._last_activity

    def describe(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "last_iteration": self._last_iteration,
                "last_message": self._last_message,
                "last_activity": self._last_activity,
            }


# ---------------------------------------------------------------------------
# Heartbeat detection
# ---------------------------------------------------------------------------

def is_heartbeat(chunk: str) -> bool:
    """Check if a chunk is a JSON heartbeat message.

    Heartbeats are sent by GranSabio LLM to keep connections alive during
    long-running operations (especially thinking/reasoning phases).
    Format: ``{"type":"heartbeat","timestamp":<millis>}``
    """
    stripped = chunk.strip()
    if not stripped:
        return False
    try:
        data = json.loads(stripped)
        return isinstance(data, dict) and data.get("type") == "heartbeat"
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Reasoning effort normalization
# ---------------------------------------------------------------------------

_REASONING_EFFORT_ALIASES: Dict[str, str] = {
    "min": "minimal",
    "minimum": "minimal",
    "lo": "low",
    "mid": "medium",
    "med": "medium",
    "hi": "high",
}


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Normalize reasoning effort labels (e.g. ``"hi"`` -> ``"high"``)."""
    if not value:
        return None
    normalized = value.strip().lower()
    normalized = _REASONING_EFFORT_ALIASES.get(normalized, normalized)
    return normalized if normalized in REASONING_TIMEOUT_MAP else None


# ---------------------------------------------------------------------------
# Timeout computation
# ---------------------------------------------------------------------------

def compute_generation_timeout(
    payload: Dict[str, Any],
    generation_response: Dict[str, Any],
    catalog_timeout: Optional[int] = None,
) -> int:
    """Determine an appropriate streaming timeout based on request metadata.

    Args:
        payload: The original generation request payload.
        generation_response: The response from POST /generate (may contain
            ``recommended_timeout_seconds``).
        catalog_timeout: Optional pre-fetched timeout from the model catalog
            (avoids an HTTP call inside this pure function).

    Returns:
        Timeout in seconds (including grace period).
    """
    def _coerce_seconds(value: Any) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        try:
            seconds = int(float(value))
        except (TypeError, ValueError):
            return None
        return seconds if seconds > 0 else None

    timeouts = payload.get("timeouts")
    if not isinstance(timeouts, dict):
        timeouts = {}

    explicit_timeout = (
        _coerce_seconds(timeouts.get("stream_seconds"))
        or _coerce_seconds(payload.get("timeout_seconds"))
        or _coerce_seconds(timeouts.get("default_seconds"))
    )
    if explicit_timeout is not None:
        return explicit_timeout + STREAM_TIMEOUT_GRACE_SECONDS

    candidates: list[int] = []

    recommended = generation_response.get("recommended_timeout_seconds")
    if isinstance(recommended, (int, float)) and recommended > 0:
        candidates.append(int(recommended))

    reasoning_effort = normalize_reasoning_effort(payload.get("reasoning_effort"))
    if reasoning_effort:
        effort_timeout = REASONING_TIMEOUT_MAP.get(reasoning_effort)
        if effort_timeout:
            candidates.append(effort_timeout)

    if not candidates and catalog_timeout and catalog_timeout > 0:
        candidates.append(catalog_timeout)

    baseline = max(candidates) if candidates else DEFAULT_PROCESS_TIMEOUT_SECONDS
    return max(DEFAULT_STREAM_TIMEOUT_SECONDS, baseline) + STREAM_TIMEOUT_GRACE_SECONDS


# ---------------------------------------------------------------------------
# Token budget utilities (pure computation, no HTTP)
# ---------------------------------------------------------------------------

def compute_token_margin(limit: int) -> int:
    """Compute a safety margin (5 %, clamped 128-2048) for a given token limit."""
    if limit <= 0:
        return 512
    margin = max(128, min(2048, limit // 20))
    if margin >= limit:
        margin = max(1, limit // 4)
    return margin


def compute_token_budgets(
    model_limit: int,
    desired_max_tokens: int,
    desired_thinking_tokens: Optional[int] = None,
) -> Tuple[int, Optional[int]]:
    """Determine safe max_tokens / thinking_tokens pair honoring model limits.

    Args:
        model_limit: The model's maximum output token limit (from catalog or fallback).
        desired_max_tokens: Requested max_tokens.
        desired_thinking_tokens: Requested thinking budget (for Claude thinking models).

    Returns:
        ``(max_tokens, thinking_tokens)`` where ``thinking_tokens`` may be ``None``.
    """
    def _non_negative_int(value: Any) -> int:
        if value is None or isinstance(value, bool):
            return 0
        try:
            return max(int(value), 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    desired_max = _non_negative_int(desired_max_tokens) or 2048
    desired_thinking = _non_negative_int(desired_thinking_tokens)
    effective_model_limit = _non_negative_int(model_limit)

    max_tokens = (
        min(desired_max, effective_model_limit)
        if effective_model_limit > 0
        else desired_max
    )
    margin = compute_token_margin(max_tokens)

    if desired_thinking <= 0:
        return max_tokens, None

    available_gap = max_tokens - margin
    if available_gap <= 0:
        logger.warning(
            "Unable to allocate thinking budget within token limit %s", max_tokens,
        )
        return max_tokens, None

    thinking = min(desired_thinking, available_gap)
    if thinking < desired_thinking:
        logger.info(
            "Clamping thinking budget from %s to %s tokens (max_tokens=%s, margin=%s)",
            desired_thinking, thinking, max_tokens, margin,
        )

    return max_tokens, thinking


def resolve_model_token_fallback(model: str) -> int:
    """Resolve token limit from hardcoded fallbacks when /models is unreachable."""
    model_lower = model.strip().lower()
    for prefix, limit in MODEL_TOKEN_FALLBACKS.items():
        if prefix in model_lower:
            return limit
    return DEFAULT_TOKEN_LIMIT


# ---------------------------------------------------------------------------
# Result status validation (shared between sync and async)
# ---------------------------------------------------------------------------

def _terminal_failure_details(result: Dict[str, Any], raw_status: Any) -> Dict[str, Any]:
    """Build the details allowlist for a non-success generation result.

    The terminal fields ``finish_reason_category``, ``retryable``, and
    ``prompt_block_reason`` / ``native_finish_reason`` are included only when the
    server actually sent them, so results from older
    servers do not acquire new ``None``-valued keys.
    """
    details = {
        "status": raw_status,
        "failure_reason": result.get("failure_reason"),
        "error_type": result.get("error_type"),
        "error_code": result.get("error_code"),
        "output_truncated": result.get("output_truncated"),
        "finish_reason": result.get("finish_reason"),
        "provider_error": result.get("provider_error"),
        "session_id": result.get("session_id"),
    }
    for terminal_key in ("finish_reason_category", "retryable", "prompt_block_reason", "native_finish_reason"):
        if terminal_key in result:
            details[terminal_key] = result[terminal_key]
    return details


def validate_result(result: Dict[str, Any]) -> None:
    """Validate a generation result dict.

    Raises:
        GranSabioGenerationCancelled: If the generation was cancelled.
        GranSabioGenerationRejected: If QA rejected the content.
        GranSabioClientError: For any other non-success status.

    Note: Exception classes are imported lazily to avoid circular imports.
    """
    from . import GranSabioClientError, GranSabioGenerationCancelled, GranSabioGenerationRejected

    if result is None:
        raise GranSabioClientError("Generation completed without providing a result")
    if not isinstance(result, dict):
        raise GranSabioClientError(
            f"Generation completed with unexpected result type: {type(result).__name__}"
        )

    raw_status = result.get("status")
    status = str(raw_status or "").lower().strip()

    if status == "cancelled":
        raise GranSabioGenerationCancelled(
            "Generation was cancelled by user request.",
            session_id=result.get("session_id"),
        )

    if status == "rejected":
        failure = (
            result.get("failure_reason")
            or result.get("error")
            or result.get("qa_summary")
            or "QA rejected the content."
        )
        raise GranSabioGenerationRejected(
            f"Generation rejected by QA ({failure}).",
            details={
                **_terminal_failure_details(result, raw_status),
                "qa_summary": result.get("qa_summary"),
                "qa_results": result.get("qa_results"),
                "final_score": result.get("final_score"),
            },
        )

    if status == "failed":
        message = (
            result.get("error")
            or result.get("failure_reason")
            or result.get("qa_summary")
            or "unknown error"
        )
        raise GranSabioClientError(
            f"Generation returned status '{status}' ({message}).",
            details=_terminal_failure_details(result, raw_status),
        )

    if status and status not in {"completed", "success", "succeeded"}:
        message = (
            result.get("error")
            or result.get("failure_reason")
            or result.get("qa_summary")
            or "unknown error"
        )
        raise GranSabioClientError(
            f"Generation returned status '{status}' ({message}).",
            details=_terminal_failure_details(result, raw_status),
        )

    if result.get("approved") is False:
        failure = (
            result.get("failure_reason")
            or result.get("qa_summary")
            or "QA rejected the content."
        )
        raise GranSabioGenerationRejected(
            f"Generation rejected by QA ({failure}).",
            details={
                "status": raw_status,
                "failure_reason": result.get("failure_reason"),
                "qa_summary": result.get("qa_summary"),
                "qa_results": result.get("qa_results"),
                "final_score": result.get("final_score"),
                "session_id": result.get("session_id"),
            },
        )
