"""
Validation context factory for non-generator tool loops (QA, Arbiter, GranSabio).

Responsibility (Fase 2, PROPOSAL_TOOLS_FOR_QA_ARBITER_GRANSABIO.md §3.4.1, §4.11):
give the shared ``call_ai_with_validation_tools`` loop a **lightweight,
measurement-only** representation of the original request so that deterministic
validators (word count, phrase frequency, lexical diversity, JSON schema) can
run against the candidate text without dragging in generator-only concepts
(``cumulative_text``, ``llm_accent_guard``, stylistic-metrics opt-in, etc.).

The returned ``MeasurementRequest`` is a plain dataclass intentionally shaped
like the subset of ``ContentRequest`` attributes read by
``deterministic_validation.validate_generation_candidate``. Using a dataclass
(instead of reusing ``ContentRequest``) keeps the surface explicit and prevents
accidental whole-request leakage into measurement-only callers.

Fail-closed whitelist: only the fields listed in ``_WHITELISTED_FIELDS`` are
carried over. If none of them is present on the original request, the factory
returns ``None`` and the caller treats that as "no measurable check possible",
which is the documented signal for QA/Arbiter/GranSabio to skip the tool loop.

This module MUST NOT import ``app_state`` and MUST NOT depend on any runtime
session/project state. It is a pure transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from models import is_json_output_requested
from phrase_frequency_config import is_phrase_frequency_active


# ---------------------------------------------------------------------------
# Whitelist (fail-closed)
# ---------------------------------------------------------------------------


# Explicit whitelist of request attributes that may be carried over into a
# measurement-only context. The list is intentionally narrow:
#
# - ``min_words`` / ``max_words`` / ``word_count_enforcement``: word-count
#   deterministic validator.
# - ``phrase_frequency``: n-gram repetition deterministic validator.
# - ``lexical_diversity``: lexical diversity deterministic validator.
# - ``json_output`` / ``content_type`` / ``json_schema`` /
#   ``json_expectations`` / ``target_field``: JSON contract deterministic
#   validator. ``content_type="json"`` remains a legacy alias for
#   ``json_output=True``.
#
# Fields EXCLUDED by design (see §3.4.1 in the proposal):
# - ``cumulative_text``: generator-only corpus of previous attempts; has no
#   meaning for a QA judging a single turn of content.
# - ``llm_accent_guard``: AI-cost generator tool, not a measurement.
# - ``include_stylistic_metrics``: opt-in stylistic snapshot; only relevant
#   when the layer explicitly requests stylistic analysis.
_WHITELISTED_FIELDS = (
    "min_words",
    "max_words",
    "word_count_enforcement",
    "phrase_frequency",
    "lexical_diversity",
    "json_output",
    "content_type",
    "json_schema",
    "json_expectations",
    "target_field",
)


# ---------------------------------------------------------------------------
# MeasurementRequest
# ---------------------------------------------------------------------------


@dataclass
class MeasurementRequest:
    """Minimal request-like object consumed by deterministic validators.

    Shape mirrors the subset of ``ContentRequest`` attributes read by
    ``deterministic_validation.validate_generation_candidate`` and its helpers.

    Exposed attributes must stay in sync with ``_WHITELISTED_FIELDS``.
    Generator-only attributes (cumulative_text, llm_accent_guard,
    include_stylistic_metrics) are intentionally absent so that a reviewer
    cannot accidentally leak them via ``getattr``.
    """

    min_words: Optional[int] = None
    max_words: Optional[int] = None
    word_count_enforcement: Optional[Any] = None
    phrase_frequency: Optional[Any] = None
    lexical_diversity: Optional[Any] = None
    content_type: str = "other"
    json_output: bool = False
    json_schema: Optional[Dict[str, Any]] = None
    json_expectations: Optional[List[Dict[str, Any]]] = None
    target_field: Optional[Any] = None

    # Explicitly-neutral attributes so that deterministic validators that probe
    # these fields via ``getattr(..., default)`` see a consistent "off" value
    # regardless of the original request.
    include_stylistic_metrics: bool = False
    cumulative_text: Optional[str] = None

    # Track the source layer name for observability/logging. Never fed into
    # the validators themselves.
    _source_layer_name: Optional[str] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _extract_field(original_request: Any, name: str) -> Any:
    """Read a whitelisted attribute from ``original_request`` defensively.

    Returns ``None`` when the attribute is missing. Does not probe private or
    non-whitelisted attributes.
    """
    return getattr(original_request, name, None)


def _has_measurable_validator(values: Dict[str, Any]) -> bool:
    """Return True when at least one whitelisted validator is effectively active.

    Activity rules:
    - ``min_words`` / ``max_words``: truthy positive integer.
    - ``word_count_enforcement``: present and ``.enabled`` truthy.
    - ``phrase_frequency``: present, ``.enabled`` truthy, and has rules.
    - ``lexical_diversity``: present and ``.enabled`` truthy.
    - ``json_output`` or ``content_type="json"``: effective JSON request.
    - ``json_schema``: truthy dict.
    - ``target_field``: truthy (non-empty string or list).
    """
    if values.get("min_words"):
        return True
    if values.get("max_words"):
        return True

    wce = values.get("word_count_enforcement")
    if wce is not None and getattr(wce, "enabled", False):
        return True

    pf = values.get("phrase_frequency")
    if is_phrase_frequency_active(pf, context="measurement request discovery"):
        return True

    ld = values.get("lexical_diversity")
    if ld is not None and getattr(ld, "enabled", False):
        return True

    if values.get("json_output") or values.get("content_type") == "json":
        return True
    if values.get("json_schema"):
        return True
    if values.get("target_field"):
        return True

    return False


def build_measurement_request_for_layer(
    original_request: Any,
    layer: Any,
) -> Optional[MeasurementRequest]:
    """Build a measurement-only request for a QA layer, or ``None`` if nothing applies.

    Args:
        original_request: The full ``ContentRequest`` (or compatible
            ``SimpleNamespace`` used by tests). May be ``None``.
        layer: The ``QALayer`` the caller is preparing to evaluate. Present in
            the signature so future layer-opt-in (e.g. explicit
            ``include_stylistic_metrics`` per layer) can be wired without
            changing the call sites. Not inspected semantically: we never
            parse ``layer.criteria`` text (regex-free semantics, see
            project-wide rule).

    Returns:
        A fresh ``MeasurementRequest`` when at least one whitelisted
        validator is active on the original request. ``None`` otherwise —
        callers should treat that as "no measurable check available for this
        layer" and skip the tool loop.
    """
    if original_request is None:
        return None

    collected: Dict[str, Any] = {
        name: _extract_field(original_request, name) for name in _WHITELISTED_FIELDS
    }
    collected["phrase_frequency"] = _normalize_phrase_frequency_for_layer(
        collected.get("phrase_frequency"),
        layer,
    )

    if not _has_measurable_validator(collected):
        return None

    # json_output must account for the legacy content_type="json" alias even
    # though the measurement dataclass carries a normalized bool.
    source_probe = SimpleNamespace(
        json_output=bool(collected.get("json_output")),
        content_type=collected.get("content_type") or "other",
    )
    json_output_value = is_json_output_requested(source_probe)

    return MeasurementRequest(
        min_words=collected.get("min_words"),
        max_words=collected.get("max_words"),
        word_count_enforcement=collected.get("word_count_enforcement"),
        phrase_frequency=collected.get("phrase_frequency"),
        lexical_diversity=collected.get("lexical_diversity"),
        content_type=collected.get("content_type") or "other",
        json_output=json_output_value,
        json_schema=collected.get("json_schema"),
        json_expectations=collected.get("json_expectations"),
        target_field=collected.get("target_field"),
        _source_layer_name=getattr(layer, "name", None),
    )


def _normalize_phrase_frequency_for_layer(config: Any, layer: Any) -> Any:
    """Normalize invalid phrase-frequency config before building measurements."""

    layer_name = getattr(layer, "name", None)
    context = (
        f"measurement request for layer '{layer_name}'"
        if layer_name
        else "measurement request"
    )
    return config if is_phrase_frequency_active(config, context=context) else None


__all__: List[str] = [
    "MeasurementRequest",
    "build_measurement_request_for_layer",
]
