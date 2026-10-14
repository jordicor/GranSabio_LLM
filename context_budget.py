"""
Context Admission Policy — single source of truth for token-budget math.

This module is the ONLY place where context-window arithmetic, safety margins,
counting confidence, and admission decisions live. No caller may re-implement
margins or ratios. Design constraints:

- FAIL FAST. Unknown capacity is never an invented default window. If an
  unverified receptor still has a numeric limit, that number remains a
  conservative ceiling: a proven overflow is rejected, while a fit retains
  unverified telemetry. Only receptors with no usable number are banded by
  their input plus reserved generation footprint: silent pass through
  ``CONTEXT_UNKNOWN_PASS_TOKENS``, warned admission through
  ``CONTEXT_UNKNOWN_REJECT_TOKENS``, then
  ``ContextCapacityUnknown`` in ``enforce`` mode.
- No implicit transformations on overflow (no truncation, no summarization,
  no model swap, no evaluator drop). This module only measures and decides.
- Counting is arithmetic plus mechanical Unicode-range density buckets. There
  is no regex/keyword semantics anywhere here.
- The only local exact tokenizer is ``tiktoken`` (OpenAI cl100k / o200k
  vocabularies). Every other provider uses the conservative estimator.
- Native provider counters (Anthropic ``count_tokens``, Gemini
  ``count_tokens``) are injected as an async callback and used only as a
  tiebreak inside the ambiguous band. This module never imports ``ai_service``
  and issues no network calls itself.

The module deliberately avoids importing ``config`` at import time; the catalog
lookup performs a late import so there is no import cycle
(``config`` -> ``context_budget``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# Guaranteed lower-bound skeletons for the fixed instruction scaffolding that
# the preflight/auto-QA planners always send around the prompt/source text.
# Kept deliberately small so they never turn a fitting payload into a false
# overflow; the dominant guaranteed component is always source_text. Shared by
# the request-admission barrier and the stateless /context/preflight builder.
PREFLIGHT_VALIDATOR_SKELETON = (
    "You are a preflight validator. Analyze the request for feasibility, "
    "contradictions, and incompatibilities. Return a structured decision."
) * 6
AUTO_QA_PLANNER_SKELETON = (
    "You are an Auto-QA planner. Derive quality-assurance layers with criteria "
    "and scores for the request. Return a structured plan."
) * 6

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CountMethod(str, Enum):
    """How the serialized input token count was obtained."""

    EXACT_PROVIDER = "exact_provider"
    EXACT_LOCAL = "exact_local"
    CONSERVATIVE_ESTIMATE = "conservative_estimate"
    UNKNOWN = "unknown"


class CapacityConfidence(str, Enum):
    """How much the resolved capacity metadata can be trusted."""

    VERIFIED = "verified"
    PROVIDER_REPORTED = "provider_reported"
    STALE = "stale"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class AdmissionDecision(str, Enum):
    """Outcome of evaluating one rendered call against its receptor limit."""

    FITS = "fits"
    NEAR_LIMIT = "near_limit"
    OVERFLOW = "overflow"
    CAPACITY_UNKNOWN = "capacity_unknown"
    CAPACITY_UNKNOWN_TOLERATED = "capacity_unknown_tolerated"
    DYNAMIC = "dynamic"


# ---------------------------------------------------------------------------
# Defaults and env-overridable knobs (single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_ADMISSION_MODE = "enforce"
DEFAULT_NEAR_LIMIT_THRESHOLD = 0.85
DEFAULT_MARGIN_EXACT_PROVIDER = 0.02
DEFAULT_MARGIN_EXACT_LOCAL = 0.05
DEFAULT_MARGIN_CONSERVATIVE = 0.18
DEFAULT_NATIVE_COUNT_CACHE_SIZE = 256
DEFAULT_CAPACITY_STALE_DAYS = 365
# Unverified-capacity bands (serialized input + reserved output/thinking). Below
# the pass threshold a receptor is admitted silently; between the two it is
# admitted with a warning; above the reject threshold it stays reject-eligible.
# These are generic generation fallbacks, not claims about every model type:
# embeddings and other specialized surfaces must keep explicit catalog limits
# and must never inherit a text-generation family assumption.
DEFAULT_UNKNOWN_PASS_TOKENS = 32768
DEFAULT_UNKNOWN_REJECT_TOKENS = 131072

# Conservative estimator ratios (chars per token). Overestimating tokens is the
# safe direction. Content-type hint and CJK density can only LOWER the ratio
# (which raises the token estimate). Phase-1 recalibration target.
RATIO_PROSE = 3.2
RATIO_JSON = 2.5
RATIO_CJK = 0.9
CJK_DENSITY_THRESHOLD = 0.2

# Provider framing overhead (single table; Phase-1 recalibration target).
FRAMING_FIXED_TOKENS = 256
FRAMING_PER_MESSAGE_TOKENS = 64
# Embeddings carry no chat framing (no system/role scaffolding, no tool
# schemas); a single small fixed overhead covers the request envelope.
FRAMING_EMBEDDING_TOKENS = 16
_MESSAGE_KINDS = frozenset({"system", "message"})


def _resolve_float_env(name: str, default: float, *, minimum: float, maximum: Optional[float]) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < minimum:
        return minimum
    if maximum is not None and value > maximum:
        return maximum
    return value


def _resolve_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def resolve_admission_mode() -> str:
    """Return ``enforce`` (default) or ``observe``. There is no ``off`` mode."""

    raw = os.getenv("CONTEXT_ADMISSION_MODE")
    if raw is None:
        return DEFAULT_ADMISSION_MODE
    normalized = raw.strip().lower()
    return normalized if normalized in ("enforce", "observe") else DEFAULT_ADMISSION_MODE


def resolve_near_limit_threshold() -> float:
    return _resolve_float_env(
        "CONTEXT_NEAR_LIMIT_THRESHOLD", DEFAULT_NEAR_LIMIT_THRESHOLD, minimum=0.0, maximum=1.0
    )


def resolve_margin(count_method: Union[CountMethod, str]) -> float:
    """Return the safety margin for a counting method (env-overridable)."""

    method = count_method.value if isinstance(count_method, CountMethod) else str(count_method)
    if method == CountMethod.EXACT_PROVIDER.value:
        return _resolve_float_env(
            "CONTEXT_MARGIN_EXACT_PROVIDER", DEFAULT_MARGIN_EXACT_PROVIDER, minimum=0.0, maximum=0.95
        )
    if method == CountMethod.EXACT_LOCAL.value:
        return _resolve_float_env(
            "CONTEXT_MARGIN_EXACT_LOCAL", DEFAULT_MARGIN_EXACT_LOCAL, minimum=0.0, maximum=0.95
        )
    # CONSERVATIVE_ESTIMATE and the (unused in practice) UNKNOWN fall back to the
    # conservative margin, which is the safest non-zero value.
    return _resolve_float_env(
        "CONTEXT_MARGIN_CONSERVATIVE", DEFAULT_MARGIN_CONSERVATIVE, minimum=0.0, maximum=0.95
    )


def resolve_native_count_cache_size() -> int:
    return _resolve_positive_int_env("CONTEXT_NATIVE_COUNT_CACHE_SIZE", DEFAULT_NATIVE_COUNT_CACHE_SIZE)


def resolve_capacity_stale_days() -> int:
    return _resolve_positive_int_env("CONTEXT_CAPACITY_STALE_DAYS", DEFAULT_CAPACITY_STALE_DAYS)


def _resolve_threshold_env(name: str, default: int) -> int:
    """Resolve an admission-threshold knob, failing fast on a bad value.

    Unlike ``_resolve_positive_int_env`` (a soft cache-size knob), these
    thresholds decide who is admitted: a typo must never silently fall back to
    the default and change admission behavior.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a positive integer number of tokens; got {raw!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"{name} must be a positive integer number of tokens; got {value}."
        )
    return value


def resolve_unknown_pass_tokens() -> int:
    """Required-footprint ceiling for silently admitting unverified capacity."""

    return _resolve_threshold_env("CONTEXT_UNKNOWN_PASS_TOKENS", DEFAULT_UNKNOWN_PASS_TOKENS)


def resolve_unknown_reject_tokens() -> int:
    """Required-footprint ceiling above which unverified capacity may reject."""

    return _resolve_threshold_env("CONTEXT_UNKNOWN_REJECT_TOKENS", DEFAULT_UNKNOWN_REJECT_TOKENS)


def resolve_unknown_capacity_thresholds() -> Tuple[int, int]:
    """Return the validated ``(pass, reject)`` unverified-capacity thresholds."""

    pass_tokens = resolve_unknown_pass_tokens()
    reject_tokens = resolve_unknown_reject_tokens()
    if pass_tokens >= reject_tokens:
        raise ValueError(
            "CONTEXT_UNKNOWN_PASS_TOKENS must be strictly below "
            f"CONTEXT_UNKNOWN_REJECT_TOKENS; got {pass_tokens} >= {reject_tokens}."
        )
    return pass_tokens, reject_tokens


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenLimitsResolution:
    """The resolved capacity view for one model, with provenance/confidence."""

    model: str
    provider: str
    max_input_tokens: Optional[int]
    max_output_tokens: Optional[int]
    max_total_context_tokens: Optional[int]
    output_accounting: str  # within_total | separate | unknown
    reasoning_accounting: str  # within_output | separate | unknown
    provenance: str  # manual_curated | provider_api | inferred | local_config | unknown
    confidence: CapacityConfidence
    verified_at: Optional[str] = None
    source: Optional[str] = None
    stale: bool = False
    suspect: bool = False
    expires_at: Optional[str] = None
    family_rule_id: Optional[str] = None


@dataclass(frozen=True)
class PayloadComponent:
    """One measurable slice of a rendered call payload."""

    name: str
    chars: int
    tokens_estimate: int
    kind: str  # system | message | tool_defs | tool_results | schema | image | framing
    text: Optional[str] = None  # optional raw text, used only for native counting


@dataclass(frozen=True)
class RenderedCallBudget:
    """The measured input to an admission decision for a single call."""

    provider: str
    model: str
    call_id: str
    components: List[PayloadComponent]
    reserved_output_tokens: int = 0
    thinking_budget_tokens: int = 0
    content_type_hint: str = "prose"
    framing: str = "chat"  # chat | embedding — selects the framing overhead model


@dataclass(frozen=True)
class CallAdmissionReport:
    """The result of evaluating one rendered call."""

    decision: AdmissionDecision
    serialized_input_tokens: int
    hard_input_limit: Optional[int]
    safe_input_limit: Optional[int]
    utilization: Optional[float]
    margin_applied: float
    count_method: CountMethod
    limits: TokenLimitsResolution
    components: List[PayloadComponent]
    detail: str
    reserved_generation_tokens: int = 0
    native_count_used: bool = False
    # True whenever capacity provenance is untrusted or no numeric window exists,
    # even when a numeric inferred limit was still usable as a conservative cap.
    # Consumers that keep an extra paranoid guard for unknown models (e.g. the
    # tool-loop char hard cap) must key off this, not off the decision alone:
    # inside the pass band the decision is FITS and would otherwise look
    # indistinguishable from a verified fit.
    capacity_unverified: bool = False

    @property
    def required_context_footprint_tokens(self) -> int:
        """Conservative input + reserved generation footprint for unknown bands."""

        return self.serialized_input_tokens + self.reserved_generation_tokens


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------


def _components_breakdown(components: Sequence[PayloadComponent]) -> List[dict]:
    """Return a content-free component breakdown safe for public payloads."""

    return [
        {"name": component.name, "kind": component.kind, "tokens": component.tokens_estimate}
        for component in components
    ]


class ContextAdmissionError(Exception):
    """
    Base error for context admission. Non-retryable by contract.

    Public payloads (``to_public_dict``) never include prompt content or
    internal model/provider identifiers. Internal diagnostics
    (``to_diagnostics``) carry the effective model/provider for telemetry and
    the debugger only.
    """

    error_code = "context_admission"

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        report: Optional[CallAdmissionReport] = None,
        call_id: Optional[str] = None,
        public_label: Optional[str] = None,
        local_or_provider: str = "local",
    ) -> None:
        self.report = report
        self.call_id = call_id
        self.public_label = public_label or call_id
        self.local_or_provider = local_or_provider
        super().__init__(message or self._default_message())

    def _default_message(self) -> str:
        if self.report is None:
            return f"Context admission failed for call '{self.call_id}'."
        return (
            f"Context admission '{self.report.decision.value}' for call "
            f"'{self.call_id}': {self.report.detail}"
        )

    def to_public_dict(self) -> dict:
        """Public, mirror-rule-safe payload. No prompt content, no real IDs."""

        report = self.report
        payload: dict = {
            "error_type": "context_admission",
            "error_code": self.error_code,
            "receptor_label": self.public_label,
            "local_or_provider": self.local_or_provider,
        }
        if report is not None:
            required_tokens = (
                report.required_context_footprint_tokens
                if report.decision is AdmissionDecision.CAPACITY_UNKNOWN
                else report.serialized_input_tokens
            )
            payload.update(
                {
                    "decision": report.decision.value,
                    "required_tokens": required_tokens,
                    "serialized_input_tokens": report.serialized_input_tokens,
                    "available_tokens": report.safe_input_limit,
                    "hard_input_limit": report.hard_input_limit,
                    "reserved_generation_tokens": report.reserved_generation_tokens,
                    "utilization": report.utilization,
                    "count_method": report.count_method.value,
                    "confidence": report.limits.confidence.value,
                    "components": _components_breakdown(report.components),
                    "detail": report.detail,
                }
            )
        return payload

    def to_diagnostics(self) -> dict:
        """Internal diagnostics (telemetry/debugger only)."""

        payload = self.to_public_dict()
        report = self.report
        if report is not None:
            payload.update(
                {
                    "model": report.limits.model,
                    "provider": report.limits.provider,
                    "provenance": report.limits.provenance,
                    "verified_at": report.limits.verified_at,
                    "expires_at": report.limits.expires_at,
                    "family_rule_id": report.limits.family_rule_id,
                    "stale": report.limits.stale,
                    "suspect": report.limits.suspect,
                }
            )
        payload["call_id"] = self.call_id
        return payload


class ContextBudgetExceeded(ContextAdmissionError):
    """The known payload exceeds the receptor's safe input capacity."""

    error_code = "context_budget_exceeded"


class ContextCapacityUnknown(ContextAdmissionError):
    """No trustworthy capacity metadata exists to decide admission."""

    error_code = "context_capacity_unknown"


class ProviderContextOverflow(ContextAdmissionError):
    """The provider reported a context overflow despite the local guard."""

    error_code = "provider_context_overflow"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("local_or_provider", "provider")
        super().__init__(*args, **kwargs)


class FanoutAdmissionError(ContextBudgetExceeded):
    """A strict_all fan-out failed: at least one receptor cannot be admitted."""

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        reports: Optional[Sequence[Tuple[str, CallAdmissionReport]]] = None,
        failing: Optional[Sequence[Tuple[str, CallAdmissionReport]]] = None,
    ) -> None:
        self.reports: List[Tuple[str, CallAdmissionReport]] = list(reports or [])
        self.failing: List[Tuple[str, CallAdmissionReport]] = list(failing or [])
        super().__init__(message or self._fanout_message())

    def _fanout_message(self) -> str:
        labels = ", ".join(label for label, _ in self.failing) or "unknown"
        return f"Fan-out admission failed for receptors: {labels}."

    def receptor_table(self) -> List[dict]:
        """Public per-receptor table using the mirror-rule labels."""

        table: List[dict] = []
        for label, report in self.reports:
            required_tokens = (
                report.required_context_footprint_tokens
                if report.decision
                in (
                    AdmissionDecision.CAPACITY_UNKNOWN,
                    AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED,
                )
                else report.serialized_input_tokens
            )
            table.append(
                {
                    "receptor_label": label,
                    "required_tokens": required_tokens,
                    "serialized_input_tokens": report.serialized_input_tokens,
                    "available_tokens": report.safe_input_limit,
                    "utilization": report.utilization,
                    "decision": report.decision.value,
                }
            )
        return table


def context_budget_error_from_overflow(
    *,
    call_id: str,
    public_label: Optional[str] = None,
    overflow_kind: Optional[str] = None,
    overflow_details: Optional[dict] = None,
    context_size_estimate: Optional[int] = None,
) -> ContextAdmissionError:
    """Build a typed context-admission error from tool-loop envelope overflow fields.

    A tool-loop envelope with ``tools_skipped_reason == "context_too_large"``
    carries ``context_overflow_kind`` / ``context_overflow_details`` but no full
    ``CallAdmissionReport`` (the loop only exposes a content-free detail dict).
    Callers that must fail fast on that skip reason (Gran Sabio, Long Text) use
    this helper so the raised error is typed, non-retryable, and mirror-safe.

    ``overflow_kind == "provider_context_overflow"`` (a provider SDK reported the
    overflow mid-loop) returns a ``ProviderContextOverflow`` so the taxonomy is
    consistent with single-shot frontiers (error_code
    ``provider_context_overflow``, ``local_or_provider="provider"``); every
    locally computed overflow returns a ``ContextBudgetExceeded`` and stays
    ``local``. ``overflow_details`` is the content-free dict from
    ``_tool_loop_overflow_details`` and is embedded in the message and attached
    as ``.overflow_details`` for diagnostics only.
    """

    details = dict(overflow_details or {})
    is_provider = overflow_kind == "provider_context_overflow"
    local_or_provider = "provider" if is_provider else "local"
    required = details.get("required_tokens")
    available = details.get("available_tokens")
    parts = [f"Tool-loop context overflow for call '{call_id}'"]
    if overflow_kind:
        parts.append(f"kind={overflow_kind}")
    if required is not None and available is not None:
        parts.append(f"required_tokens={required} available_tokens={available}")
    elif context_size_estimate is not None:
        parts.append(f"context_size_estimate={context_size_estimate}")
    message = "; ".join(parts) + "."

    error_cls = ProviderContextOverflow if is_provider else ContextBudgetExceeded
    error = error_cls(
        message,
        call_id=call_id,
        public_label=public_label,
        local_or_provider=local_or_provider,
    )
    error.overflow_kind = overflow_kind
    error.overflow_details = details
    return error


# ---------------------------------------------------------------------------
# Native provider counter cache (injected async callback; interface only)
# ---------------------------------------------------------------------------

# Signature: (provider, model, text) -> awaitable token count. A ``None`` result
# means "no native count available for this provider" (not an error): the caller
# keeps its local estimate.
NativeTokenCounter = Callable[[str, str, str], Awaitable[Optional[int]]]


class _NativeCountCache:
    """LRU cache keyed by (provider, model, sha256(text)) for native counts."""

    def __init__(self) -> None:
        self._store: "OrderedDict[Tuple[str, str, str], int]" = OrderedDict()

    @staticmethod
    def _key(provider: str, model: str, text: str) -> Tuple[str, str, str]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return (provider, model, digest)

    async def count(
        self, provider: str, model: str, text: str, counter: NativeTokenCounter
    ) -> Optional[int]:
        key = self._key(provider, model, text)
        cached = self._store.get(key)
        if cached is not None:
            self._store.move_to_end(key)
            return cached
        raw = await counter(provider, model, text)
        if raw is None:
            # No native count is available for this provider (e.g. OpenAI, which
            # is covered by local tiktoken, or any provider without a native
            # counter). This is not an error and must not raise a noisy
            # traceback: the caller keeps its local estimate silently. A missing
            # count is not cached, so a later provider addition is picked up.
            return None
        result = int(raw)
        self._store[key] = result
        self._store.move_to_end(key)
        self._evict()
        return result

    def _evict(self) -> None:
        max_size = resolve_native_count_cache_size()
        while len(self._store) > max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._store)


_native_count_cache = _NativeCountCache()


def reset_native_count_cache() -> None:
    """Clear the shared native-count cache (used by tests)."""

    _native_count_cache.clear()


def native_count_cache_size() -> int:
    """Current number of cached native counts (used by tests)."""

    return len(_native_count_cache)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODING_CACHE: dict = {}
_TIKTOKEN_UNAVAILABLE_WARNED = False


def _get_encoding(encoding_name: str):
    """
    Return a cached tiktoken encoding, importing tiktoken lazily.

    Returns ``None`` when tiktoken is not installed. This is a deliberate
    policy, not an accidental crash: a missing tiktoken degrades OpenAI-family
    counting to the conservative estimator (the universal safe method) rather
    than raising ``ImportError`` mid-admission. Exactly one warning is emitted
    per process.
    """

    cached = _ENCODING_CACHE.get(encoding_name)
    if cached is not None:
        return cached
    try:
        import tiktoken  # lazy: heavy optional dependency
    except ImportError:
        global _TIKTOKEN_UNAVAILABLE_WARNED
        if not _TIKTOKEN_UNAVAILABLE_WARNED:
            logger.warning(
                "tiktoken unavailable; falling back to conservative estimation "
                "for OpenAI-family models"
            )
            _TIKTOKEN_UNAVAILABLE_WARNED = True
        return None

    encoding = tiktoken.get_encoding(encoding_name)
    _ENCODING_CACHE[encoding_name] = encoding
    return encoding


def _openai_encoding_name(model: str) -> str:
    """
    Map an OpenAI model name to a tiktoken encoding mechanically.

    Never relies on ``encoding_for_model`` (unknown modern names have no
    mapping). gpt-5.x / gpt-4o / gpt-4.1 / o-series use o200k_base;
    text-embedding-3-* and legacy gpt-4/gpt-3.5 use cl100k_base.
    """

    name = str(model or "").strip().lower()
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    if name.startswith("text-embedding-3"):
        return "cl100k_base"
    if name.startswith(
        ("gpt-5", "gpt-4o", "gpt-4.1", "chatgpt-4o", "o1", "o3", "o4")
    ):
        return "o200k_base"
    if name.startswith(("gpt-4", "gpt-3.5", "text-embedding-ada", "davinci", "curie")):
        return "cl100k_base"
    # Modern OpenAI default.
    return "o200k_base"


def _is_openai_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() == "openai"


_CJK_RE = re.compile(
    "[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff"
    "\uac00-\ud7af\u1100-\u11ff\U00020000-\U0002a6df]"
)
_CJK_RUN_RE = re.compile(_CJK_RE.pattern + "+")
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def _cjk_fraction(text: str) -> float:
    if text.isascii() or _CJK_RE.search(text) is None:
        return 0.0

    # Count runs without allocating a match or character list for the full text.
    cjk = sum(match.end() - match.start() for match in _CJK_RUN_RE.finditer(text))
    whitespace = sum(
        match.end() - match.start() for match in _WHITESPACE_RUN_RE.finditer(text)
    )
    return cjk / (len(text) - whitespace)


def _estimate_ratio(text: str, hint: str) -> float:
    """Return chars/token ratio; hint and CJK density can only LOWER it."""

    ratio = RATIO_PROSE
    if str(hint).lower() in ("json", "code"):
        ratio = min(ratio, RATIO_JSON)
    if _cjk_fraction(text) > CJK_DENSITY_THRESHOLD:
        ratio = min(ratio, RATIO_CJK)
    return ratio


def estimate_tokens(text: str, hint: str = "prose") -> int:
    """Conservative token estimate: ceil(chars / ratio)."""

    chars = len(text)
    if chars == 0:
        return 0
    return int(math.ceil(chars / _estimate_ratio(text, hint)))


def estimate_tokens_from_size(size: int, hint: str = "prose") -> int:
    """Conservative token estimate from a raw byte/char count (advisory only).

    Used when only a size is known and not the text itself (e.g. attachment
    byte totals for barrier-1 worst-case warnings). UTF-8 bytes are an upper
    bound on character count, so dividing by the prose (or JSON) ratio yields a
    conservative over-estimate. CJK density cannot be measured from a size
    alone, so the density-lowered ratio is deliberately not applied here.
    """

    if size <= 0:
        return 0
    ratio = RATIO_JSON if str(hint).lower() in ("json", "code") else RATIO_PROSE
    return int(math.ceil(size / ratio))


def count_tokens(
    text_or_components: Union[str, Iterable[Union[str, PayloadComponent]]],
    provider: str,
    model: str,
    hint: str = "prose",
) -> Tuple[int, CountMethod]:
    """
    Count input tokens for a string or a sequence of strings/components.

    OpenAI uses the exact local tiktoken vocabulary; everything else uses the
    conservative estimator. For a sequence of ``PayloadComponent`` objects the
    already-computed ``tokens_estimate`` values are summed (the caller counted
    them with this same function).
    """

    method = CountMethod.EXACT_LOCAL if _is_openai_provider(provider) else CountMethod.CONSERVATIVE_ESTIMATE

    encoding = None
    if method is CountMethod.EXACT_LOCAL:
        encoding = _get_encoding(_openai_encoding_name(model))
        if encoding is None:
            # Deliberate policy: when tiktoken is unavailable, OpenAI-family
            # counting degrades to the conservative estimator (the universal
            # safe method) so the larger conservative margin applies.
            method = CountMethod.CONSERVATIVE_ESTIMATE

    if isinstance(text_or_components, str):
        if encoding is not None:
            return len(encoding.encode(text_or_components, disallowed_special=())), method
        return estimate_tokens(text_or_components, hint), method

    total = 0
    for item in text_or_components:
        if isinstance(item, PayloadComponent):
            total += int(item.tokens_estimate)
        elif isinstance(item, str):
            item_tokens, _ = count_tokens(item, provider, model, hint)
            total += item_tokens
        else:
            raise TypeError(f"Unsupported component type: {type(item)!r}")
    return total, method


def make_payload_component(
    name: str,
    text: str,
    kind: str,
    provider: str,
    model: str,
    hint: str = "prose",
    *,
    keep_text: bool = True,
) -> PayloadComponent:
    """Build a ``PayloadComponent`` with its token estimate already computed."""

    tokens, _ = count_tokens(text, provider, model, hint)
    return PayloadComponent(
        name=name,
        chars=len(text),
        tokens_estimate=tokens,
        kind=kind,
        text=text if keep_text else None,
    )


# ---------------------------------------------------------------------------
# Capacity resolution
# ---------------------------------------------------------------------------


def _confidence_for_provenance(provenance: str) -> CapacityConfidence:
    if provenance == "manual_curated":
        return CapacityConfidence.VERIFIED
    if provenance == "provider_api":
        return CapacityConfidence.PROVIDER_REPORTED
    if provenance in ("family_floor", "inferred", "local_config"):
        # local_config (Ollama num_ctx) is fail-closed until curated.
        return CapacityConfidence.INFERRED
    return CapacityConfidence.UNKNOWN


def _is_stale(verified_at: Optional[str]) -> bool:
    if not verified_at:
        return False
    raw = str(verified_at).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - parsed).days
    return age_days > resolve_capacity_stale_days()


def resolve_token_limits(model_name: str) -> TokenLimitsResolution:
    """
    Resolve the capacity view for ``model_name`` with provenance/confidence.

    Precedence (implemented in ``config.get_model_token_limits``): curated
    overlay > numeric per-model ``token_limits`` object > provider-reported or
    local numeric legacy fields > active direct-provider family floor > generic
    inferred legacy fields. A family floor remains INFERRED and acts only as a
    conservative ceiling; missing capacity yields UNKNOWN confidence and uses
    the required-footprint bands downstream.
    """

    from config import config  # late import to avoid an import cycle

    raw = config.get_model_token_limits(model_name)
    if not raw:
        return TokenLimitsResolution(
            model=model_name,
            provider="",
            max_input_tokens=None,
            max_output_tokens=None,
            max_total_context_tokens=None,
            output_accounting="unknown",
            reasoning_accounting="unknown",
            provenance="unknown",
            confidence=CapacityConfidence.UNKNOWN,
        )

    provenance = str(raw.get("provenance") or "unknown")
    confidence = _confidence_for_provenance(provenance)
    verified_at = raw.get("verified_at")
    stale = _is_stale(verified_at)
    if stale and confidence in (CapacityConfidence.VERIFIED, CapacityConfidence.PROVIDER_REPORTED):
        confidence = CapacityConfidence.STALE

    return TokenLimitsResolution(
        model=str(raw.get("model") or model_name),
        provider=str(raw.get("provider") or ""),
        max_input_tokens=raw.get("max_input_tokens"),
        max_output_tokens=raw.get("max_output_tokens"),
        max_total_context_tokens=raw.get("max_total_context_tokens"),
        output_accounting=str(raw.get("output_accounting") or "unknown"),
        reasoning_accounting=str(raw.get("reasoning_accounting") or "unknown"),
        provenance=provenance,
        confidence=confidence,
        verified_at=verified_at,
        source=raw.get("source"),
        stale=stale,
        suspect=bool(raw.get("suspect")),
        expires_at=raw.get("expires_at"),
        family_rule_id=raw.get("family_rule_id"),
    )


# ---------------------------------------------------------------------------
# Admission math
# ---------------------------------------------------------------------------


def _reserved_generation_tokens(budget: RenderedCallBudget, limits: TokenLimitsResolution) -> int:
    reserved = max(0, int(budget.reserved_output_tokens or 0))
    thinking = max(0, int(budget.thinking_budget_tokens or 0))
    # within_output: thinking is already part of the output reservation -> do
    # not subtract it twice. separate/unknown: count thinking in the shared
    # window (conservative for unknown).
    if limits.reasoning_accounting != "within_output":
        reserved += thinking
    return reserved


def _hard_input_limit(limits: TokenLimitsResolution, reserved_generation: int) -> Optional[int]:
    max_input = limits.max_input_tokens
    max_total = limits.max_total_context_tokens
    candidates: List[int] = []
    if limits.output_accounting == "separate":
        # Independent input/output budgets: no subtraction from a shared window.
        if max_input is not None:
            candidates.append(int(max_input))
    else:
        # within_total or unknown -> treat as a shared window (conservative).
        if max_input is not None:
            candidates.append(int(max_input))
        if max_total is not None:
            candidates.append(int(max_total) - reserved_generation)
    if not candidates:
        return None
    return min(candidates)


def _framing_overhead(
    components: Sequence[PayloadComponent], provider: str, framing: str = "chat"
) -> int:
    if framing == "embedding":
        # Embedding requests have no chat/tool scaffolding: a minimal fixed
        # envelope overhead instead of the per-message chat framing.
        return FRAMING_EMBEDDING_TOKENS
    message_like = sum(1 for component in components if component.kind in _MESSAGE_KINDS)
    return FRAMING_FIXED_TOKENS + FRAMING_PER_MESSAGE_TOKENS * max(1, message_like)


def _payload_text(components: Sequence[PayloadComponent]) -> str:
    parts = [component.text for component in components if component.text]
    return "\n".join(parts)


def _classify_unknown_capacity(
    serialized_input_tokens: int,
    reserved_generation_tokens: int,
    cause: str,
) -> Tuple[AdmissionDecision, str]:
    """Band a receptor with no usable numeric capacity, by total footprint.

    Missing capacity is not itself evidence of overflow. The configured banded
    policy admits below the pass threshold, admits with a warning up to the
    reject threshold, and fails closed above it. ``cause`` states WHY capacity
    is unavailable so the detail (and every message derived from it) stays
    honest. Callers must not use this generic path to invent limits for known
    specialized model types.
    """

    pass_tokens, reject_tokens = resolve_unknown_capacity_thresholds()
    required_tokens = serialized_input_tokens + reserved_generation_tokens
    footprint = (
        f"required context footprint {required_tokens} tokens "
        f"(serialized input {serialized_input_tokens} + reserved generation "
        f"{reserved_generation_tokens})"
    )

    if required_tokens <= pass_tokens:
        return (
            AdmissionDecision.FITS,
            f"Capacity is unverified ({cause}), but the {footprint} is at or below the safe-admission "
            f"threshold {pass_tokens}; admitted without a capacity check.",
        )
    if required_tokens <= reject_tokens:
        return (
            AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED,
            f"Capacity is unverified ({cause}) and the {footprint} is above the safe-admission threshold "
            f"{pass_tokens} but at or below the reject threshold {reject_tokens}; "
            "admitted with a warning.",
        )
    return (
        AdmissionDecision.CAPACITY_UNKNOWN,
        f"Capacity is unverified ({cause}) and the {footprint} is above the reject threshold "
        f"{reject_tokens}; the request cannot be confirmed to fit.",
    )


def _build_report(
    budget: RenderedCallBudget,
    limits: TokenLimitsResolution,
    *,
    serialized_input_tokens: int,
    count_method: CountMethod,
    native_used: bool = False,
) -> CallAdmissionReport:
    margin = resolve_margin(count_method)
    reserved_generation = _reserved_generation_tokens(budget, limits)
    hard = _hard_input_limit(limits, reserved_generation)
    safe = int(math.floor(hard * (1.0 - margin))) if hard is not None else None
    utilization = (
        (serialized_input_tokens / safe) if (safe is not None and safe > 0) else None
    )
    near_threshold = resolve_near_limit_threshold()

    untrusted_confidence = limits.confidence in (
        CapacityConfidence.UNKNOWN,
        CapacityConfidence.INFERRED,
    )

    capacity_unverified = untrusted_confidence or hard is None or safe is None

    has_numeric_capacity = hard is not None and safe is not None

    if has_numeric_capacity and (safe <= 0 or serialized_input_tokens > safe):
        # A numeric limit remains useful as a conservative ceiling even when its
        # provenance is inferred/unverified. Confidence controls how strongly a
        # fit can be stated; it must never erase positive evidence of overflow.
        decision = AdmissionDecision.OVERFLOW
        qualifier = "unverified " if untrusted_confidence else ""
        detail = (
            f"Serialized input {serialized_input_tokens} tokens exceeds {qualifier}safe "
            f"input limit {safe} (hard {hard}, margin {margin:.2f})."
        )
    elif not has_numeric_capacity:
        # With no usable numeric capacity at all, the input plus reserved
        # generation footprint is the only available risk signal. Apply the
        # global pass/tolerate/reject bands.
        cause = (
            f"capacity confidence '{limits.confidence.value}' is not trustworthy "
            f"for a strict decision, provenance={limits.provenance}"
            if untrusted_confidence
            else "no max_input_tokens or max_total_context_tokens is available"
        )
        decision, detail = _classify_unknown_capacity(
            serialized_input_tokens,
            reserved_generation,
            cause,
        )
    elif utilization is not None and utilization >= near_threshold:
        decision = AdmissionDecision.NEAR_LIMIT
        qualifier = "unverified " if untrusted_confidence else ""
        detail = (
            f"Utilization {utilization:.2f} is at/above the near-limit threshold "
            f"{near_threshold:.2f} against the {qualifier}numeric capacity; admitted "
            "with warning."
        )
    elif untrusted_confidence:
        pass_tokens = resolve_unknown_pass_tokens()
        required_tokens = serialized_input_tokens + reserved_generation
        if required_tokens <= pass_tokens:
            decision = AdmissionDecision.FITS
            detail = (
                f"Serialized input {serialized_input_tokens} tokens fits within the "
                f"unverified numeric safe input limit {safe}; total required footprint "
                f"{required_tokens} tokens (including {reserved_generation} reserved "
                f"generation tokens) is at or below the safe-admission threshold "
                f"{pass_tokens}, so it is admitted silently."
            )
        else:
            decision = AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
            detail = (
                f"Serialized input {serialized_input_tokens} tokens fits within the "
                f"unverified numeric safe input limit {safe}, but total required footprint "
                f"{required_tokens} tokens (including {reserved_generation} reserved "
                f"generation tokens) is above the safe-admission threshold {pass_tokens}; "
                "admitted with a warning."
            )
    else:
        decision = AdmissionDecision.FITS
        detail = (
            f"Serialized input {serialized_input_tokens} tokens fits within safe "
            f"input limit {safe}."
        )

    return CallAdmissionReport(
        decision=decision,
        serialized_input_tokens=serialized_input_tokens,
        hard_input_limit=hard,
        safe_input_limit=safe,
        utilization=utilization,
        margin_applied=margin,
        count_method=count_method,
        limits=limits,
        components=list(budget.components),
        detail=detail,
        reserved_generation_tokens=reserved_generation,
        native_count_used=native_used,
        capacity_unverified=capacity_unverified,
    )


def _emit_telemetry(telemetry_cb: Optional[Callable[[CallAdmissionReport], Any]], report: CallAdmissionReport) -> None:
    if telemetry_cb is None:
        return
    try:
        telemetry_cb(report)
    except Exception:  # telemetry must never break admission
        logger.warning("context admission telemetry callback failed", exc_info=True)


async def evaluate_call_admission(
    budget: RenderedCallBudget,
    *,
    native_counter: Optional[NativeTokenCounter] = None,
    mode: Optional[str] = None,  # accepted for signature parity; does not raise here
    limits: Optional[TokenLimitsResolution] = None,
) -> CallAdmissionReport:
    """
    Evaluate one rendered call and return a report (never raises).

    ``limits`` may be supplied to reuse a resolution and to keep the math
    testable without a live catalog; otherwise it is resolved from the model.
    The native counter is consulted only inside the ambiguous band.
    """

    resolved_limits = limits if limits is not None else resolve_token_limits(budget.model)
    body_tokens, count_method = count_tokens(
        budget.components, budget.provider, budget.model, budget.content_type_hint
    )
    framing = _framing_overhead(budget.components, budget.provider, budget.framing)
    serialized = body_tokens + framing

    report = _build_report(
        budget,
        resolved_limits,
        serialized_input_tokens=serialized,
        count_method=count_method,
    )

    if (
        native_counter is not None
        and report.hard_input_limit is not None
        and report.safe_input_limit is not None
        and resolved_limits.confidence
        not in (CapacityConfidence.UNKNOWN, CapacityConfidence.INFERRED)
        # Native refinement runs ONLY for conservative estimates. tiktoken
        # exact_local is already authoritative and there is no OpenAI native
        # counter in v1, so EXACT_LOCAL/EXACT_PROVIDER never re-count.
        and count_method is CountMethod.CONSERVATIVE_ESTIMATE
    ):
        band = report.margin_applied * report.hard_input_limit
        if abs(serialized - report.safe_input_limit) <= band:
            payload_text = _payload_text(budget.components)
            if payload_text:
                try:
                    exact_body = await _native_count_cache.count(
                        budget.provider, budget.model, payload_text, native_counter
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A raising native counter must not crash admission; keep
                    # the already-computed conservative report unchanged.
                    logger.warning(
                        "native token count failed; keeping local estimate",
                        exc_info=True,
                    )
                else:
                    if exact_body is not None:
                        # The native counter measured only the text portion:
                        # ``_payload_text`` concatenates components carrying
                        # ``.text``. Components without text (images carry only
                        # ``tokens_estimate``) are invisible to the counter and
                        # must be added back, otherwise the recomputed total
                        # silently drops them and a payload that overflows can be
                        # falsely admitted. The exact_provider margin still
                        # applies: only the natively-counted text portion gains
                        # accuracy; the non-text estimates come from documented
                        # provider vision formulas and are acceptable as-is.
                        non_text_tokens = sum(
                            int(component.tokens_estimate)
                            for component in budget.components
                            if not component.text
                        )
                        report = _build_report(
                            budget,
                            resolved_limits,
                            serialized_input_tokens=exact_body + non_text_tokens + framing,
                            count_method=CountMethod.EXACT_PROVIDER,
                            native_used=True,
                        )

    return report


async def admit_call(
    budget: RenderedCallBudget,
    *,
    native_counter: Optional[NativeTokenCounter] = None,
    telemetry_cb: Optional[Callable[[CallAdmissionReport], Any]] = None,
    mode: Optional[str] = None,
    limits: Optional[TokenLimitsResolution] = None,
) -> CallAdmissionReport:
    """
    Admit a single call. In ``enforce`` mode raises on OVERFLOW /
    CAPACITY_UNKNOWN; in ``observe`` mode logs and returns without raising.
    CAPACITY_UNKNOWN_TOLERATED is admitted with a warning in BOTH modes.
    Utilization telemetry is emitted for every evaluated call.
    """

    report = await evaluate_call_admission(budget, native_counter=native_counter, limits=limits)
    _emit_telemetry(telemetry_cb, report)

    if report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED:
        # Admitted in both modes: below the reject threshold an unverified
        # window is a risk to warn about, not evidence of overflow.
        logger.warning(
            "context admission tolerated unverified capacity: call=%s model=%s "
            "required=%s confidence=%s",
            budget.call_id,
            report.limits.model,
            report.required_context_footprint_tokens,
            report.limits.confidence.value,
        )
        return report

    effective_mode = mode or resolve_admission_mode()
    if effective_mode == "observe":
        if report.decision in (AdmissionDecision.OVERFLOW, AdmissionDecision.CAPACITY_UNKNOWN):
            logger.warning(
                "context admission observe: call=%s decision=%s required=%s available=%s",
                budget.call_id,
                report.decision.value,
                (
                    report.required_context_footprint_tokens
                    if report.decision is AdmissionDecision.CAPACITY_UNKNOWN
                    else report.serialized_input_tokens
                ),
                report.safe_input_limit,
            )
        return report

    if report.decision is AdmissionDecision.OVERFLOW:
        raise ContextBudgetExceeded(report=report, call_id=budget.call_id, public_label=budget.call_id)
    if report.decision is AdmissionDecision.CAPACITY_UNKNOWN:
        raise ContextCapacityUnknown(report=report, call_id=budget.call_id, public_label=budget.call_id)
    return report


async def evaluate_fanout_admission(
    budgets: Sequence[RenderedCallBudget],
    *,
    policy: str = "strict_all",
    native_counter: Optional[NativeTokenCounter] = None,
    telemetry_cb: Optional[Callable[[CallAdmissionReport], Any]] = None,
    mode: Optional[str] = None,
    public_labels: Optional[Sequence[str]] = None,
) -> List[CallAdmissionReport]:
    """
    Evaluate a group of calls that must all be admitted together (strict_all).

    In ``enforce`` mode, if ANY receptor is OVERFLOW or CAPACITY_UNKNOWN a
    ``FanoutAdmissionError`` is raised before any dispatch, carrying the full
    per-receptor table with mirror-rule labels. ``observe`` mode never raises.
    CAPACITY_UNKNOWN_TOLERATED receptors are admitted with a warning in both
    modes and never break the fan-out.
    """

    if policy != "strict_all":
        raise ValueError(f"Unsupported fan-out policy '{policy}'. Only 'strict_all' exists in v1.")

    labels = list(public_labels) if public_labels is not None else [b.call_id for b in budgets]
    if len(labels) != len(budgets):
        raise ValueError("public_labels length must match budgets length.")

    reports: List[CallAdmissionReport] = []
    labeled: List[Tuple[str, CallAdmissionReport]] = []
    for label, budget in zip(labels, budgets):
        report = await evaluate_call_admission(budget, native_counter=native_counter)
        _emit_telemetry(telemetry_cb, report)
        reports.append(report)
        labeled.append((label, report))

    tolerated = [
        (label, report)
        for label, report in labeled
        if report.decision is AdmissionDecision.CAPACITY_UNKNOWN_TOLERATED
    ]
    if tolerated:
        logger.warning(
            "context admission fan-out tolerated unverified capacity for %d "
            "receptor(s): %s",
            len(tolerated),
            ", ".join(
                f"{label}={report.required_context_footprint_tokens} tokens"
                for label, report in tolerated
            ),
        )

    effective_mode = mode or resolve_admission_mode()
    if effective_mode == "observe":
        return reports

    failing = [
        (label, report)
        for label, report in labeled
        if report.decision in (AdmissionDecision.OVERFLOW, AdmissionDecision.CAPACITY_UNKNOWN)
    ]
    if failing:
        raise FanoutAdmissionError(reports=labeled, failing=failing)
    return reports


async def admit_embedding_texts(
    texts: Sequence[str],
    *,
    provider: str,
    model: str,
    call_id: str,
    admit: Optional[Callable[[RenderedCallBudget], Awaitable[CallAdmissionReport]]] = None,
) -> None:
    """Admit each embedding input against the model's per-input budget.

    Embeddings reserve no output tokens and use the minimal embedding framing
    instead of chat framing. On OVERFLOW in enforce mode this raises
    ``ContextBudgetExceeded``; callers MUST let it propagate rather than
    return empty vectors (the empty-vector fallback is reserved for provider
    failures, never for admission verdicts).

    ``admit`` lets AIService inject its own wrapper (actionable messages,
    native counter, telemetry ContextVar); other callers use plain
    ``admit_call``. The per-input budget construction lives here so every
    embedding caller shares one loop.
    """

    admit_fn = admit or admit_call
    for index, text in enumerate(texts):
        budget = RenderedCallBudget(
            provider=provider,
            model=model,
            call_id=call_id,
            components=[
                make_payload_component(
                    name=f"input[{index}]",
                    text=text or "",
                    kind="message",
                    provider=provider,
                    model=model,
                    hint="prose",
                )
            ],
            reserved_output_tokens=0,
            thinking_budget_tokens=0,
            content_type_hint="prose",
            framing="embedding",
        )
        await admit_fn(budget)
