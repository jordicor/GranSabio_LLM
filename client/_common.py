"""
Shared utilities for sync and async GranSabio LLM clients.

Contains constants, helper classes, and pure computation functions
used by both GranSabioClient and AsyncGranSabioClient.
No HTTP calls are made from this module.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------

DEFAULT_STREAM_TIMEOUT_SECONDS = 600
STREAM_TIMEOUT_GRACE_SECONDS = 180
RESULT_POLL_GRACE_SECONDS = 120
RESULT_POLL_INTERVAL_SECONDS = 5.0
STREAM_ACTIVITY_CHECK_SECONDS = 30.0

# Timeouts for reasoning models - significantly increased for multi-iteration reasoning.
# These models can iterate many times and take hours to complete.
REASONING_TIMEOUT_MAP: Dict[str, int] = {
    "minimal": 3600,      # 1 hour
    "low": 7200,          # 2 hours
    "medium": 14400,      # 4 hours
    "high": 28800,        # 8 hours
}

# ---------------------------------------------------------------------------
# Provider API key environment variable mapping
# ---------------------------------------------------------------------------

PROVIDER_KEY_ENV_MAP: Dict[str, Tuple[str, ...]] = {
    "X-OpenAI-Key": ("OPENAI_API_KEY",),
    "X-Gemini-Key": ("GOOGLE_API_KEY", "GEMINI_KEY"),
    "X-Anthropic-Key": ("ANTHROPIC_API_KEY",),
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

    baseline = max(candidates) if candidates else DEFAULT_STREAM_TIMEOUT_SECONDS
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
    desired_max = max(int(desired_max_tokens or 0), 0) or 2048
    desired_thinking = max(int(desired_thinking_tokens or 0), 0)
    margin = compute_token_margin(model_limit if model_limit > 0 else desired_max)

    if model_limit > 0:
        max_tokens = model_limit
    else:
        max_tokens = max(desired_max, desired_thinking + margin)

    if desired_thinking <= 0:
        return max_tokens, None

    # Ensure max_tokens can accommodate both completion and thinking requirements
    required = desired_thinking + margin
    if max_tokens < required:
        max_tokens = min(model_limit, required) if model_limit > 0 else required

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

    if status and status not in {"completed", "success", "succeeded"}:
        if status == "cancelled":
            raise GranSabioGenerationCancelled(
                "Generation was cancelled by user request.",
                session_id=result.get("session_id"),
            )
        message = (
            result.get("error")
            or result.get("failure_reason")
            or result.get("qa_summary")
            or "unknown error"
        )
        raise GranSabioClientError(
            f"Generation returned status '{status}' ({message})."
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
