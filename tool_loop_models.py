"""
Neutral module for shared tool-loop primitives.

This module is deliberately free of imports from ``ai_service`` and ``models``
so it can be depended on by both without creating circular imports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OutputContract(str, Enum):
    """Expected output format of a tool-loop caller.

    Values may arrive as enum members or matching string literals; the
    tool-loop entry point coerces them with ``OutputContract(value)`` and
    raises ``ToolLoopContractError`` for unknown values.

    - ``FREE_TEXT``: prose/free-form output. ``response_format`` must be None.
    - ``JSON_LOOSE``: JSON requested without schema. ``response_format`` must
      be None; post-loop validation extracts/repairs a top-level object or
      array from common AI wrappers.
    - ``JSON_STRUCTURED``: schema-enforced JSON. ``response_format`` must be a
      non-empty top-level object schema; the parsed envelope payload is a dict.
    """

    FREE_TEXT = "free_text"
    JSON_LOOSE = "json_loose"
    JSON_STRUCTURED = "json_structured"


class LoopScope(str, Enum):
    """Which subsystem initiated the tool loop."""

    GENERATOR = "generator"
    QA = "qa"
    ARBITER = "arbiter"
    GRAN_SABIO = "gran_sabio"


class PayloadScope(str, Enum):
    """Controls which fields of a ``DraftValidationResult`` are exposed to the LLM.

    - ``GENERATOR``: full payload including ``approved`` / ``hard_failed`` gate
      signals. Intended for iterative creative loops where the generator relies
      on those flags to steer the next revision.
    - ``MEASUREMENT_ONLY``: objective metrics and neutral descriptive feedback
      only. No ``approved`` / ``hard_failed`` gate fields. Intended for
      evaluators (QA, Arbiter, GranSabio) that must judge the draft without
      being primed by a gate verdict from the framework itself.
    """

    GENERATOR = "generator"
    MEASUREMENT_ONLY = "measurement_only"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolLoopTraceEntry(BaseModel):
    """Single entry recorded in the tool-loop trace.

    The shape is intentionally permissive: providers (OpenAI, Claude, Gemini)
    record slightly different structural fields per turn. The common header
    is ``turn`` + ``scope`` + ``event``; everything else is carried in the
    open ``extra`` map to stay future-proof without forcing refactors when
    new telemetry fields are added.
    """

    model_config = ConfigDict(extra="allow")

    turn: int
    scope: LoopScope
    event: str
    tool: Optional[str] = None
    approved: Optional[bool] = None
    score: Optional[float] = None
    word_count: Optional[int] = None
    hard_failed: Optional[bool] = None
    metrics: Optional[Dict[str, Any]] = None
    stage: Optional[str] = None
    reason: Optional[str] = None


class ToolLoopEnvelope(BaseModel):
    """Structured envelope returned by ``call_ai_with_validation_tools``.

    ``payload`` holds the parsed-and-validated dict when the caller uses
    ``OutputContract.JSON_STRUCTURED``; it is ``None`` for
    ``OutputContract.FREE_TEXT`` and ``OutputContract.JSON_LOOSE``. The loose
    contract extracts and validates JSON parseability but does not expose the
    parsed value.
    """

    model_config = ConfigDict(extra="allow")

    loop_scope: LoopScope
    trace: List[ToolLoopTraceEntry] = Field(default_factory=list)
    output_schema_valid: bool = True
    streaming_disabled_reason: Optional[str] = None
    tools_skipped_reason: Optional[str] = None
    turns: int = 0
    accepted: bool = False
    accepted_via: str = ""
    context_size_estimate: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolLoopSchemaViolationError(Exception):
    """Raised when a tool-loop turn produces output that violates the contract.

    Not retried by ``retries_enabled`` (that flag only covers transient
    errors such as rate limits, 5xx, or timeouts). Schema violations indicate
    a deterministic logic failure and must surface to the caller.
    """


class ValidationToolInputTooLarge(Exception):
    """Raised when the LLM invokes ``validate_draft`` with a ``text`` argument
    that exceeds ``VALIDATE_DRAFT_MAX_LENGTH``.

    The tool loop catches this exception inside its tool-call handler, emits
    a ``validate_draft_oversize`` telemetry event, forces a finalize turn,
    and surfaces a neutral ``{"error": "text_exceeds_limit"}`` dict to the
    model for the current tool_response only. The exception itself never
    mixes with ``DraftValidationResult`` internally.
    """

    def __init__(self, actual_length: int, max_length: int) -> None:
        super().__init__(
            f"validate_draft input length {actual_length} exceeds limit {max_length}"
        )
        self.actual_length = actual_length
        self.max_length = max_length


class ToolLoopContextOverflow(Exception):
    """Raised when a provider reports a context-window overflow mid-loop.

    The loop translates provider-specific errors (``context_length_exceeded``
    from OpenAI, ``invalid_request_error: max_tokens`` from Anthropic,
    ``InvalidArgument: context window`` from Gemini) into this tagged
    exception so the caller can distinguish authoritative overflow from
    other transient failures.
    """

    def __init__(
        self,
        turn: int,
        accumulated_chars_estimate: int,
        provider_error: Optional[BaseException] = None,
    ) -> None:
        super().__init__(
            f"Tool loop context overflow at turn {turn} "
            f"(accumulated_chars_estimate={accumulated_chars_estimate})"
        )
        self.turn = turn
        self.accumulated_chars_estimate = accumulated_chars_estimate
        self.provider_error = provider_error


class JsonContractError(Exception):
    """Raised when JSON output contracts cannot be parsed or validated.

    Covers both:
    - ``OutputContract.JSON_STRUCTURED``: structured-output response that fails
      parsing or schema/expectation validation after the loop finishes.
    - ``OutputContract.JSON_LOOSE``: response where no JSON object/array can
      be extracted/repaired, or where expectations fail.

    Unlike ``ToolLoopSchemaViolationError`` (which describes the contract
    itself being broken, e.g. final turn returns the wrong shape after a
    forced finalize), ``JsonContractError`` covers the narrower case of a
    JSON parse/validation failure surfacing to the caller. This exception
    exists so callers can catch JSON-contract issues independently of other
    schema-violation semantics.
    """


class ToolLoopContractError(Exception):
    """Raised when a caller violates the ``OutputContract`` contract.

    This intentionally does not inherit from ``ValueError`` because the
    generator treats ``ValueError`` as a signal that tools are unavailable and
    falls back to standard generation. Contract errors must surface directly.
    """


# ---------------------------------------------------------------------------
# LLM JSON parsing utility
# ---------------------------------------------------------------------------


def _format_cleanroom_issues(errors: List[Any]) -> str:
    """Return a compact error summary from ai_json_cleanroom issues."""

    return "; ".join(
        f"{getattr(issue, 'path', '$')}: {getattr(issue, 'message', str(issue))}"
        for issue in (errors or [])
    ) or "unknown JSON validation error"


def parse_json_with_markdown_fences(
    response: str,
    *,
    schema: Optional[Dict[str, Any]] = None,
    context: str = "JSON payload",
) -> Dict[str, Any]:
    """Parse an LLM JSON object that may include common model wrappers.

    This is the shared single-shot fallback parser for JSON emitted by models.
    It delegates to ``ai_json_cleanroom`` so fenced blocks, balanced JSON
    blocks, and conservative repairs are handled consistently while optional
    schema validation still fails closed.

    Args:
        response: Raw text returned by the provider single-shot call.
        schema: Optional JSON Schema to validate the parsed object against.
        context: Human-readable label included in errors.

    Returns:
        Parsed JSON dict.

    Raises:
        JsonContractError: If the input fails to parse as JSON.
    """
    from tools.ai_json_cleanroom import validate_ai_json

    validation_result = validate_ai_json(response or "", schema=schema)
    if not validation_result.json_valid:
        details = _format_cleanroom_issues(validation_result.errors)
        raise JsonContractError(f"{context} failed JSON validation: {details}")

    parsed = validation_result.data
    if not isinstance(parsed, dict):
        raise JsonContractError(
            f"{context} must be a JSON object, got {type(parsed).__name__}."
        )
    return parsed
