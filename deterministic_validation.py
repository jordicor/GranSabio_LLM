"""
Shared deterministic validation helpers for generation and QA bypass.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import json_utils
from json_field_utils import try_extract_json_from_content
from models import QAEvaluation, WordCountEnforcement, is_json_output_requested
from phrase_frequency_config import is_phrase_frequency_active
from tool_loop_models import PayloadScope
from tools.ai_json_cleanroom import ValidateOptions, validate_ai_json
from tools.lexical_diversity_utils import analyze_text_lexical_diversity
from tools.phrase_frequency_utils import analyze_phrase_frequency
from tools.string_utils import remove_invisible_control
from word_count_utils import (
    check_word_count_compliance,
    count_words,
    create_word_count_qa_layer,
    word_count_config_to_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class DeterministicIssue:
    """One concrete issue emitted by deterministic validation."""

    code: str
    severity: str
    message: str
    hard: bool = False
    suggestion: str = ""
    count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterministicCheckResult:
    """Result for one deterministic validator."""

    name: str
    score: float
    pass_threshold: float
    passed: bool
    deal_breaker: bool
    summary: str
    issues: List[DeterministicIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterministicValidationContext:
    """Prepared content and JSON metadata for deterministic checks."""

    raw_content: str
    text_for_validation: str
    word_count: int
    json_check: Optional[DeterministicCheckResult] = None
    target_field_paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DraftValidationResult:
    """Combined validation result consumed by the reusable tool loop.

    The internal fields (``approved``, ``hard_failed``, ``feedback``) are used
    by the loop itself for gate/feedback decisions. The ``build_visible_payload``
    method emits the subset that should travel to the LLM in the ``tool_response``,
    filtered by the caller's ``PayloadScope``.
    """

    approved: bool
    hard_failed: bool
    score: float
    word_count: int
    feedback: str
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    checks: Dict[str, Any]
    stylistic_metrics: Optional[Dict[str, Any]] = None
    visible_payload: Dict[str, Any] = field(default_factory=dict)

    def _has_text_measurement(self) -> bool:
        """Return True when word/text metrics are meaningful for this result."""

        return "word_count" in self.metrics

    def build_visible_payload(self, scope: PayloadScope) -> Dict[str, Any]:
        """Return the dict that should reach the LLM for this ``scope``.

        ``GENERATOR`` returns the full payload (``approved`` / ``hard_failed``
        act as a gate signal for the iterative creative loop).

        ``MEASUREMENT_ONLY`` omits the gate fields and renames ``feedback``
        to ``feedback_neutral`` so evaluators see an objective description
        rather than a corrective order.
        """
        if scope == PayloadScope.GENERATOR:
            payload: Dict[str, Any] = {
                "tool": "validate_draft",
                "approved": self.approved,
                "hard_failed": self.hard_failed,
                "score": self.score,
                "feedback": self.feedback,
                "issues": list(self.issues),
                "metrics": dict(self.metrics),
                "checks": dict(self.checks),
            }
            if self._has_text_measurement():
                payload["word_count"] = self.word_count
            if self.stylistic_metrics is not None:
                payload["stylistic_metrics"] = dict(self.stylistic_metrics)
            return payload

        if scope == PayloadScope.MEASUREMENT_ONLY:
            payload = {
                "tool": "validate_draft",
                "score": self.score,
                "feedback_neutral": self.feedback,
                "issues": list(self.issues),
                "metrics": dict(self.metrics),
                "checks": dict(self.checks),
            }
            if self._has_text_measurement():
                payload["word_count"] = self.word_count
            if self.stylistic_metrics is not None:
                payload["stylistic_metrics"] = dict(self.stylistic_metrics)
            return payload

        raise ValueError(f"Unsupported PayloadScope: {scope!r}")


def has_active_generation_validators(request: Any) -> bool:
    """Return True when generation has measurable constraints worth validating."""

    if is_json_output_requested(request):
        return True
    if getattr(request, "target_field", None):
        return True
    if _requires_text_for_validation(request):
        return True
    return False


def _requires_text_for_validation(request: Any) -> bool:
    """Return True when active deterministic checks need extracted/plain text."""

    if _segment_budget_contract_dict(request) is not None:
        return True

    if getattr(request, "min_words", None) or getattr(request, "max_words", None):
        return True

    word_count_enforcement = getattr(request, "word_count_enforcement", None)
    if word_count_enforcement:
        if isinstance(word_count_enforcement, dict):
            if word_count_enforcement.get("enabled"):
                return True
        elif getattr(word_count_enforcement, "enabled", False):
            return True

    phrase_frequency = getattr(request, "phrase_frequency", None)
    phrase_frequency_active = is_phrase_frequency_active(
        phrase_frequency,
        context="text validation discovery",
    )
    if phrase_frequency_active:
        return True

    lexical_diversity = getattr(request, "lexical_diversity", None)
    if lexical_diversity and getattr(lexical_diversity, "enabled", False):
        return True

    if getattr(request, "cumulative_text", None) and phrase_frequency_active:
        return True

    if getattr(request, "include_stylistic_metrics", False):
        return True

    return False


def prepare_validation_context(
    raw_content: str,
    request: Any,
    *,
    include_json_validation: bool = True,
    json_options: Optional[ValidateOptions] = None,
) -> DeterministicValidationContext:
    """Prepare validation text, including JSON extraction when requested."""

    text_for_validation = raw_content or ""
    json_check: Optional[DeterministicCheckResult] = None
    target_field_paths: List[str] = []

    json_output = is_json_output_requested(request)
    target_field = getattr(request, "target_field", None)
    should_extract_text = bool(target_field) or _requires_text_for_validation(request)

    if json_output or target_field:
        if include_json_validation and json_output:
            validation_result = validate_ai_json(
                raw_content,
                schema=getattr(request, "json_schema", None),
                expectations=getattr(request, "json_expectations", None),
                options=json_options,
            )
            if not validation_result.json_valid:
                issues = [
                    DeterministicIssue(
                        code="json_invalid",
                        severity="critical",
                        message=getattr(issue, "message", "JSON output failed validation"),
                        hard=False,
                    )
                    for issue in validation_result.errors
                ] or [
                    DeterministicIssue(
                        code="json_invalid",
                        severity="critical",
                        message="JSON output failed validation",
                        hard=False,
                    )
                ]
                json_check = DeterministicCheckResult(
                    name="JSON Output Contract",
                    score=0.0,
                    pass_threshold=10.0,
                    passed=False,
                    deal_breaker=False,
                    summary="JSON output is invalid or does not satisfy the requested contract.",
                    issues=issues,
                    metadata=validation_result.to_dict(),
                )
                return DeterministicValidationContext(
                    raw_content=raw_content,
                    text_for_validation="",
                    word_count=0,
                    json_check=json_check,
                    target_field_paths=target_field_paths,
                )
            json_check = DeterministicCheckResult(
                name="JSON Output Contract",
                score=10.0,
                pass_threshold=10.0,
                passed=True,
                deal_breaker=False,
                summary="JSON output satisfies the requested contract.",
                issues=[],
                metadata=validation_result.to_dict(),
            )
            if not should_extract_text:
                return DeterministicValidationContext(
                    raw_content=raw_content,
                    text_for_validation="",
                    word_count=0,
                    json_check=json_check,
                    target_field_paths=target_field_paths,
                )

        if should_extract_text:
            try:
                json_context, extracted_text = try_extract_json_from_content(
                    content=raw_content,
                    json_output=json_output,
                    target_field=target_field,
                )
            except Exception as exc:
                json_check = DeterministicCheckResult(
                    name="JSON Target Field Extraction",
                    score=0.0,
                    pass_threshold=10.0,
                    passed=False,
                    deal_breaker=False,
                    summary=f"Could not extract validation text from JSON payload: {exc}",
                    issues=[
                        DeterministicIssue(
                            code="json_target_field_extraction_failed",
                            severity="critical",
                            message=f"Could not extract validation text from JSON payload: {exc}",
                            hard=False,
                        )
                    ],
                    metadata={"error": str(exc)},
                )
                return DeterministicValidationContext(
                    raw_content=raw_content,
                    text_for_validation="",
                    word_count=0,
                    json_check=json_check,
                    target_field_paths=target_field_paths,
                )

            if json_context and json_context.get("error") == "ambiguous_fields":
                field_names = json_context.get("candidates", [])
                json_check = DeterministicCheckResult(
                    name="JSON Target Field Resolution",
                    score=0.0,
                    pass_threshold=10.0,
                    passed=False,
                    deal_breaker=False,
                    summary="Validation could not determine which JSON text field should be measured.",
                    issues=[
                        DeterministicIssue(
                            code="json_target_field_ambiguous",
                            severity="critical",
                            message=json_context.get("message", "Multiple candidate text fields found."),
                            hard=False,
                            metadata={"candidates": field_names},
                        )
                    ],
                    metadata=json_context,
                )
                return DeterministicValidationContext(
                    raw_content=raw_content,
                    text_for_validation="",
                    word_count=0,
                    json_check=json_check,
                    target_field_paths=target_field_paths,
                )

            if not json_context:
                json_check = DeterministicCheckResult(
                    name="JSON Target Field Resolution",
                    score=0.0,
                    pass_threshold=10.0,
                    passed=False,
                    deal_breaker=False,
                    summary="Expected JSON text fields but none could be extracted for validation.",
                    issues=[
                        DeterministicIssue(
                            code="json_target_field_missing",
                            severity="critical",
                            message="Expected JSON text fields but none could be extracted for validation.",
                            hard=False,
                        )
                    ],
                    metadata={},
                )
                return DeterministicValidationContext(
                    raw_content=raw_content,
                    text_for_validation="",
                    word_count=0,
                    json_check=json_check,
                    target_field_paths=target_field_paths,
                )

            text_for_validation = extracted_text
            if json_context:
                target_field_paths = list(json_context.get("target_field_paths", []))
                metadata = {
                    "target_field_paths": target_field_paths,
                    "target_field_discovered": json_context.get("target_field_discovered", False),
                }
                if json_check is not None:
                    metadata["json_validation"] = json_check.metadata
                json_check = DeterministicCheckResult(
                    name="JSON Output Contract",
                    score=10.0,
                    pass_threshold=10.0,
                    passed=True,
                    deal_breaker=False,
                    summary="JSON output and target field extraction succeeded.",
                    issues=[],
                    metadata=metadata,
                )

    return DeterministicValidationContext(
        raw_content=raw_content,
        text_for_validation=text_for_validation,
        word_count=count_words(text_for_validation),
        json_check=json_check,
        target_field_paths=target_field_paths,
    )


def evaluate_word_count_check(text: str, request: Any) -> Optional[DeterministicCheckResult]:
    """Evaluate word-count constraints for a text draft."""

    min_words = getattr(request, "min_words", None)
    max_words = getattr(request, "max_words", None)
    if not min_words and not max_words:
        return None

    config_dict = word_count_config_to_dict(getattr(request, "word_count_enforcement", None))
    if config_dict is None:
        config_dict = WordCountEnforcement().model_dump()

    compliance = check_word_count_compliance(text, min_words, max_words, config_dict)
    layer = create_word_count_qa_layer(min_words, max_words, config_dict)

    actual_count = compliance["actual_count"]
    required_min = compliance["required_min"]
    required_max = compliance["required_max"]
    target_min = compliance["target_min"]
    target_max = compliance["target_max"]
    score = float(compliance["score"])
    deal_breaker = score == 0.0
    passed = score >= float(layer.min_score or 8.0)

    if score == 10.0:
        summary = (
            f"Word count is within the ideal target range ({target_min or 'N/A'}-{target_max or 'N/A'})."
        )
        issues: List[DeterministicIssue] = []
    elif actual_count < required_min:
        summary = f"Word count is below the acceptable range ({actual_count} < {required_min})."
        issues = [
            DeterministicIssue(
                code="word_count_below_min",
                severity="critical",
                message=summary,
                hard=True,
                suggestion=f"Expand the draft to at least {target_min or required_min} words.",
                count=required_min - actual_count,
            )
        ]
    elif actual_count > required_max:
        summary = f"Word count is above the acceptable range ({actual_count} > {required_max})."
        issues = [
            DeterministicIssue(
                code="word_count_above_max",
                severity="critical",
                message=summary,
                hard=True,
                suggestion=f"Condense the draft to no more than {target_max or required_max} words.",
                count=actual_count - required_max,
            )
        ]
    elif target_min and actual_count < target_min:
        summary = f"Word count is inside flexibility but still below the ideal minimum ({actual_count} < {target_min})."
        issues = [
            DeterministicIssue(
                code="word_count_below_target",
                severity="warning",
                message=summary,
                hard=False,
                suggestion=f"Add concrete content until the draft reaches at least {target_min} words.",
                count=target_min - actual_count,
            )
        ]
    else:
        summary = f"Word count is inside flexibility but still above the ideal maximum ({actual_count} > {target_max})."
        issues = [
            DeterministicIssue(
                code="word_count_above_target",
                severity="warning",
                message=summary,
                hard=False,
                suggestion=f"Trim redundant content until the draft stays at or below {target_max} words.",
                count=actual_count - int(target_max or actual_count),
            )
        ]

    return DeterministicCheckResult(
        name="Word Count Enforcement",
        score=score,
        pass_threshold=float(layer.min_score or 8.0),
        passed=passed,
        deal_breaker=deal_breaker,
        summary=summary,
        issues=issues,
        metadata=compliance,
    )


def evaluate_phrase_frequency_check(text: str, request: Any) -> Optional[DeterministicCheckResult]:
    """Evaluate configured phrase-frequency rules."""

    config = getattr(request, "phrase_frequency", None)
    if not config or not getattr(config, "enabled", False):
        return None

    if not is_phrase_frequency_active(config, context="deterministic phrase-frequency check"):
        return None

    settings = config.to_settings()
    result = analyze_phrase_frequency(text, settings)
    layer = config.build_layer(order=1)
    issues = result.issues
    deal_breaker = any(issue.severity == "deal_breaker" for issue in issues)
    score = 10.0 if not issues else (2.0 if deal_breaker else max(5.0, 8.0 - (len(issues) * 0.5)))
    passed = score >= float(layer.min_score or 8.0)

    if not issues:
        summary = "No configured phrase-frequency limits were exceeded."
        report_issues: List[DeterministicIssue] = []
    else:
        first = issues[0]
        summary = f"Detected {len(issues)} phrase-frequency issue(s); first issue: '{first.phrase}' ({first.count}/{first.limit})."
        report_issues = [
            DeterministicIssue(
                code="phrase_frequency_exceeded",
                severity="critical" if issue.severity == "deal_breaker" else "warning",
                message=f"Phrase '{issue.phrase}' appears {issue.count} times (limit {issue.limit}).",
                hard=issue.severity == "deal_breaker",
                suggestion=issue.guidance or "Reduce direct repetition or vary the phrasing.",
                count=issue.count,
                metadata={
                    "phrase": issue.phrase,
                    "n": issue.n,
                    "limit": issue.limit,
                    "rule": issue.rule_label,
                },
            )
            for issue in issues
        ]

    return DeterministicCheckResult(
        name="Phrase Frequency Guard",
        score=score,
        pass_threshold=float(layer.min_score or 8.0),
        passed=passed,
        deal_breaker=deal_breaker,
        summary=summary,
        issues=report_issues,
        metadata={
            "analysis": result.analyzer_output,
            "issues": [
                {
                    "phrase": issue.phrase,
                    "n": issue.n,
                    "count": issue.count,
                    "limit": issue.limit,
                    "severity": issue.severity,
                    "rule": issue.rule_label,
                    "guidance": issue.guidance,
                    "repeat_ratio_tokens": issue.repeat_ratio_tokens,
                }
                for issue in issues
            ],
            "rules_total": len(settings.rules),
        },
    )


def evaluate_lexical_diversity_check(text: str, request: Any) -> Optional[DeterministicCheckResult]:
    """Evaluate lexical-diversity rules."""

    config = getattr(request, "lexical_diversity", None)
    if not config or not getattr(config, "enabled", False):
        return None

    settings = config.to_settings()
    result = analyze_text_lexical_diversity(text, settings)
    layer = config.build_layer(order=1)
    passed = result.score >= float(layer.min_score or 8.0)

    if result.deal_breaker:
        issues = [
            DeterministicIssue(
                code="lexical_diversity_red",
                severity="critical",
                message="Lexical diversity reached a RED decision.",
                hard=True,
                suggestion="Vary wording, reduce repeated vocabulary, and broaden the lexicon.",
                metadata={"decision": result.decision_label},
            )
        ]
    elif not passed:
        issues = [
            DeterministicIssue(
                code="lexical_diversity_amber",
                severity="warning",
                message="Lexical diversity is below the configured target.",
                hard=False,
                suggestion="Use more varied wording and reduce repeated high-frequency terms.",
                metadata={"decision": result.decision_label},
            )
        ]
    else:
        issues = []

    return DeterministicCheckResult(
        name="Lexical Diversity Guard",
        score=float(result.score),
        pass_threshold=float(layer.min_score or 8.0),
        passed=passed,
        deal_breaker=bool(result.deal_breaker),
        summary=f"Lexical diversity decision: {result.decision_label}.",
        issues=issues,
        metadata={
            "decision": result.decision_label,
            "adjusted_grades": result.adjusted_grades,
            "analysis": result.analysis,
        },
    )


def evaluate_cumulative_repetition_check(text: str, request: Any) -> Optional[DeterministicCheckResult]:
    """Evaluate repetition across cumulative text plus the current draft."""

    cumulative_text = getattr(request, "cumulative_text", None)
    config = getattr(request, "phrase_frequency", None)
    if not cumulative_text or not config or not getattr(config, "enabled", False):
        return None

    full_text = cumulative_text + "\n\n" + text
    settings = config.to_settings()
    result = analyze_phrase_frequency(full_text, settings)
    layer = config.build_layer(order=1)
    total_word_count = (
        getattr(request, "cumulative_word_count", None) or count_words(cumulative_text)
    ) + count_words(text)

    issues = result.issues
    deal_breaker = any(issue.severity == "deal_breaker" for issue in issues)
    score = 10.0 if not issues else (2.0 if deal_breaker else max(5.0, 8.0 - (len(issues) * 0.5)))
    passed = score >= float(layer.min_score or 8.0)

    if not issues:
        summary = f"No cumulative repetition issues detected across {total_word_count} words."
        report_issues: List[DeterministicIssue] = []
    else:
        first = issues[0]
        summary = (
            f"Detected {len(issues)} cumulative repetition issue(s) across {total_word_count} words; "
            f"first issue: '{first.phrase}' ({first.count}/{first.limit})."
        )
        report_issues = [
            DeterministicIssue(
                code="cumulative_repetition_exceeded",
                severity="critical" if issue.severity == "deal_breaker" else "warning",
                message=f"Cumulative phrase '{issue.phrase}' appears {issue.count} times (limit {issue.limit}).",
                hard=issue.severity == "deal_breaker",
                suggestion=issue.guidance or "Reduce repeated phrasing across chapters.",
                count=issue.count,
                metadata={
                    "phrase": issue.phrase,
                    "n": issue.n,
                    "limit": issue.limit,
                    "rule": issue.rule_label,
                },
            )
            for issue in issues
        ]

    return DeterministicCheckResult(
        name="Cumulative Repetition Guard",
        score=score,
        pass_threshold=float(layer.min_score or 8.0),
        passed=passed,
        deal_breaker=deal_breaker,
        summary=summary,
        issues=report_issues,
        metadata={
            "analysis": result.analyzer_output,
            "issues": [
                {
                    "phrase": issue.phrase,
                    "n": issue.n,
                    "count": issue.count,
                    "limit": issue.limit,
                    "severity": issue.severity,
                    "rule": issue.rule_label,
                    "guidance": issue.guidance,
                    "repeat_ratio_tokens": issue.repeat_ratio_tokens,
                }
                for issue in issues
            ],
            "total_word_count": total_word_count,
            "rules_total": len(settings.rules),
        },
    )


# ---------------------------------------------------------------------------
# Segment budgets contract (segment-budgets/v1)
# ---------------------------------------------------------------------------

#: Check/QA-layer name used by the deterministic segment budget guard.
SEGMENT_BUDGET_CHECK_NAME = "Segment Budget Guard"

_SEGMENT_BUDGET_SCHEMA_VERSION = "segment-budgets/v1"
_SEGMENT_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$")
_SEGMENT_ID_RE = re.compile(r"^[^:\s]+$")
_SEGMENT_BUDGET_FIELDS = ("min_words", "target_words", "max_words")
_SEGMENT_BUDGET_MAX_ISSUES = 6


def _segment_budget_contract_dict(request: Any) -> Optional[Dict[str, Any]]:
    """Return the normalized segment budgets contract, or None when inactive."""

    contract = word_count_config_to_dict(getattr(request, "segment_budgets", None))
    if not contract or not contract.get("enabled", True):
        return None
    return contract


def _json_payload_from_context(context: DeterministicValidationContext) -> Any:
    """Return the decoded JSON payload of the candidate when available."""

    metadata = context.json_check.metadata if context.json_check else {}
    data = metadata.get("data")
    if data is None and isinstance(metadata.get("json_validation"), dict):
        data = metadata["json_validation"].get("data")
    if data is None:
        try:
            data = json_utils.loads(context.raw_content)
        except Exception:
            data = None
    return data


def _segment_budget_issue(
    code: str,
    message: str,
    suggestion: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> DeterministicIssue:
    """Build one hard segment-budget issue (all contract violations are hard)."""

    return DeterministicIssue(
        code=code,
        severity="critical",
        message=message,
        hard=True,
        suggestion=suggestion,
        metadata=metadata or {},
    )


def _parse_segment_word_budget(
    value: Any,
    *,
    label: str,
    issues: List[DeterministicIssue],
) -> Optional[Dict[str, int]]:
    """Validate one min/target/max budget triple; extra keys are ignored (v1)."""

    if not isinstance(value, dict) or any(
        type(value.get(field)) is not int for field in _SEGMENT_BUDGET_FIELDS
    ):
        issues.append(_segment_budget_issue(
            "segment_budget_structure_invalid",
            f"{label} word_budget must declare integer min_words, target_words and max_words.",
            suggestion="Declare all three budget fields as integers.",
        ))
        return None
    budget = {field: int(value[field]) for field in _SEGMENT_BUDGET_FIELDS}
    if (
        not 0 <= budget["min_words"] <= budget["target_words"] <= budget["max_words"]
        or budget["target_words"] < 1
    ):
        issues.append(_segment_budget_issue(
            "segment_budget_structure_invalid",
            f"{label} word_budget is incoherent (min={budget['min_words']}, "
            f"target={budget['target_words']}, max={budget['max_words']}); "
            "expected 0 <= min <= target <= max with target >= 1.",
            suggestion="Fix the budget so that min <= target <= max and target >= 1.",
        ))
        return None
    return budget


def _validate_segment_budget_structure(
    structure: Any,
    *,
    structure_field: str,
    request_min_words: Optional[int],
    request_max_words: Optional[int],
) -> Tuple[Optional[Dict[str, int]], List[Dict[str, Any]], List[DeterministicIssue]]:
    """Validate the self-declared structure object (segment-budgets/v1).

    Returns ``(document_budget, sections, issues)``; the first two values are
    only meaningful when ``issues`` is empty. Fields not owned by v1 are
    ignored. Distribution rules and the request-bounds anti-elusion rule only
    run when the base structure is fully valid, to avoid cascading noise.
    """

    issues: List[DeterministicIssue] = []
    if not isinstance(structure, dict):
        issues.append(_segment_budget_issue(
            "segment_budget_structure_invalid",
            f"Structure field '{structure_field}' is missing or is not a JSON object.",
            suggestion=(
                f"Return a '{structure_field}' JSON object declaring word_budget "
                "and the ordered sections."
            ),
        ))
        return None, [], issues

    document_budget = _parse_segment_word_budget(
        structure.get("word_budget"), label="Document", issues=issues,
    )

    raw_sections = structure.get("sections")
    if not isinstance(raw_sections, list):
        issues.append(_segment_budget_issue(
            "segment_budget_structure_invalid",
            "Structure 'sections' must be a list (empty for a continuous document).",
            suggestion="Declare sections as an ordered list, or [] for a continuous document.",
        ))
        return document_budget, [], issues

    sections: List[Dict[str, Any]] = []
    by_id: Dict[str, Optional[str]] = {}
    active_parent: Optional[str] = None
    for index, raw_section in enumerate(raw_sections):
        position = f"sections[{index}]"
        if not isinstance(raw_section, dict):
            issues.append(_segment_budget_issue(
                "segment_budget_structure_invalid",
                f"{position} must be a JSON object with title and word_budget.",
            ))
            continue
        title = raw_section.get("title")
        if (
            not isinstance(title, str)
            or not title.strip()
            or title != title.strip()
            or "\n" in title
            or "\r" in title
        ):
            issues.append(_segment_budget_issue(
                "segment_budget_structure_invalid",
                f"{position} title must be a non-empty single-line string without "
                "surrounding whitespace.",
            ))
            continue
        subsection_id = raw_section.get("subsection_id")
        if subsection_id is not None:
            if not isinstance(subsection_id, str) or not _SEGMENT_ID_RE.fullmatch(subsection_id):
                issues.append(_segment_budget_issue(
                    "segment_budget_structure_invalid",
                    f"{position} subsection_id '{subsection_id}' is invalid; it must "
                    "be non-empty and contain no ':' or whitespace.",
                ))
                continue
            if subsection_id in by_id:
                issues.append(_segment_budget_issue(
                    "segment_budget_structure_invalid",
                    f"{position} duplicates subsection_id '{subsection_id}'.",
                    suggestion="Keep every subsection_id unique.",
                ))
                continue
        parent = raw_section.get("parent_subsection_id")
        if parent is not None:
            if not isinstance(parent, str) or parent not in by_id:
                issues.append(_segment_budget_issue(
                    "segment_budget_structure_invalid",
                    f"{position} parent_subsection_id '{parent}' does not reference a "
                    "preceding top-level section.",
                ))
                continue
            if by_id[parent] is not None:
                issues.append(_segment_budget_issue(
                    "segment_budget_structure_invalid",
                    f"{position} exceeds the maximum depth: only two section levels "
                    "are allowed (section and subsection).",
                ))
                continue
            if parent != active_parent:
                issues.append(_segment_budget_issue(
                    "segment_budget_structure_invalid",
                    f"{position} is not contiguous to its parent '{parent}'; child "
                    "sections must immediately follow their parent and siblings.",
                ))
                continue
        section_budget = _parse_segment_word_budget(
            raw_section.get("word_budget"),
            label=f"Section '{title}'",
            issues=issues,
        )
        if section_budget is None:
            continue
        if parent is None:
            active_parent = subsection_id
        if subsection_id is not None:
            by_id[subsection_id] = parent
        sections.append({
            "title": title,
            "subsection_id": subsection_id,
            "parent_subsection_id": parent,
            "word_budget": section_budget,
        })

    if issues or document_budget is None:
        return document_budget, sections, issues

    top_level = [section for section in sections if section["parent_subsection_id"] is None]
    if top_level:
        target_sum = sum(section["word_budget"]["target_words"] for section in top_level)
        max_sum = sum(section["word_budget"]["max_words"] for section in top_level)
        if target_sum != document_budget["target_words"]:
            issues.append(_segment_budget_issue(
                "segment_budget_distribution_invalid",
                f"Top-level section targets sum to {target_sum}, but the document "
                f"target is {document_budget['target_words']}.",
                suggestion="Adjust section targets so they sum exactly to the document target.",
            ))
        if max_sum > document_budget["max_words"]:
            issues.append(_segment_budget_issue(
                "segment_budget_distribution_invalid",
                f"Top-level section maximums sum to {max_sum}, above the document "
                f"maximum of {document_budget['max_words']}.",
                suggestion="Lower section maximums so their sum fits the document maximum.",
            ))
    for parent_section in top_level:
        children = [
            section for section in sections
            if section["parent_subsection_id"] == parent_section["subsection_id"]
            and parent_section["subsection_id"] is not None
        ]
        if not children:
            continue
        parent_budget = parent_section["word_budget"]
        parent_label = f"'{parent_section['title']}' ({parent_section['subsection_id']})"
        child_target_sum = sum(section["word_budget"]["target_words"] for section in children)
        child_max_sum = sum(section["word_budget"]["max_words"] for section in children)
        if child_target_sum != parent_budget["target_words"]:
            issues.append(_segment_budget_issue(
                "segment_budget_distribution_invalid",
                f"Child targets of {parent_label} sum to {child_target_sum}, but the "
                f"parent target is {parent_budget['target_words']}.",
                suggestion="Adjust child targets so they sum exactly to the parent target.",
            ))
        if child_max_sum > parent_budget["max_words"]:
            issues.append(_segment_budget_issue(
                "segment_budget_distribution_invalid",
                f"Child maximums of {parent_label} sum to {child_max_sum}, above the "
                f"parent maximum of {parent_budget['max_words']}.",
                suggestion="Lower child maximums so their sum fits the parent maximum.",
            ))

    if request_min_words is not None and document_budget["min_words"] < request_min_words:
        issues.append(_segment_budget_issue(
            "segment_budget_document_exceeds_request",
            f"Declared document minimum ({document_budget['min_words']}) is below "
            f"the request min_words ({request_min_words}).",
            suggestion="Declare a document budget inside the request min_words/max_words.",
        ))
    if request_max_words is not None and document_budget["max_words"] > request_max_words:
        issues.append(_segment_budget_issue(
            "segment_budget_document_exceeds_request",
            f"Declared document maximum ({document_budget['max_words']}) exceeds "
            f"the request max_words ({request_max_words}).",
            suggestion="Declare a document budget inside the request min_words/max_words.",
        ))

    return document_budget, sections, issues


def _evaluate_segment_budget_text(
    text: str,
    document_budget: Dict[str, int],
    sections: List[Dict[str, Any]],
    *,
    base_heading_level: int,
) -> Tuple[List[DeterministicIssue], Dict[str, Any]]:
    """Cross the target text with the declared structure.

    Heading sequence and inclusive segment lengths mirror the reference
    ``validate_composed_text`` semantics; word counting uses ``count_words``
    (``\\w+`` runs, headings included).
    """

    lines = text.splitlines()
    headings = [
        (index, len(match.group(1)), match.group(2).rstrip())
        for index, line in enumerate(lines)
        if (match := _SEGMENT_HEADING_RE.match(line))
    ]
    expected = [
        (
            base_heading_level if section["parent_subsection_id"] is None else base_heading_level + 1,
            section["title"],
        )
        for section in sections
    ]
    found = [(level, title) for _, level, title in headings]
    if found != expected:
        expected_render = " -> ".join(f"(H{level}) {title}" for level, title in expected)
        if not expected_render:
            expected_render = "no headings (continuous document)"
        issue = _segment_budget_issue(
            "segment_budget_headings_mismatch",
            f"Text headings do not match the declared structure: expected "
            f"{len(expected)} heading(s), found {len(found)}.",
            suggestion=f"Use exactly these headings in this order: {expected_render}.",
            metadata={
                "expected_headings": [[level, title] for level, title in expected],
                "found_headings": [[level, title] for level, title in found],
            },
        )
        return [issue], {"headings_mismatch": issue.metadata}

    issues: List[DeterministicIssue] = []
    document_count = count_words(text)
    document_report = {
        "actual": document_count,
        "min_words": document_budget["min_words"],
        "max_words": document_budget["max_words"],
        "within_budget": True,
    }
    if not document_budget["min_words"] <= document_count <= document_budget["max_words"]:
        document_report["within_budget"] = False
        if document_count < document_budget["min_words"]:
            suggestion = (
                f"Expand the document by at least "
                f"{document_budget['min_words'] - document_count} words."
            )
        else:
            suggestion = (
                f"Condense the document by at least "
                f"{document_count - document_budget['max_words']} words."
            )
        issues.append(_segment_budget_issue(
            "segment_budget_document_out_of_bounds",
            f"Document has {document_count} words; declared budget is "
            f"{document_budget['min_words']}-{document_budget['max_words']}.",
            suggestion=suggestion,
            metadata=dict(document_report),
        ))

    section_reports: List[Dict[str, Any]] = []
    for position, section in enumerate(sections):
        start, level, _ = headings[position]
        end = next(
            (line for line, next_level, _ in headings[position + 1:] if next_level <= level),
            len(lines),
        )
        section_count = count_words("\n".join(lines[start:end]))
        budget = section["word_budget"]
        within_budget = budget["min_words"] <= section_count <= budget["max_words"]
        report = {
            "title": section["title"],
            "subsection_id": section["subsection_id"],
            "actual": section_count,
            "min_words": budget["min_words"],
            "max_words": budget["max_words"],
            "within_budget": within_budget,
        }
        section_reports.append(report)
        if within_budget:
            continue
        id_part = (
            f" (subsection_id '{section['subsection_id']}')"
            if section["subsection_id"]
            else ""
        )
        if section_count < budget["min_words"]:
            suggestion = (
                f"Expand section '{section['title']}' by at least "
                f"{budget['min_words'] - section_count} words."
            )
        else:
            suggestion = (
                f"Trim section '{section['title']}' by at least "
                f"{section_count - budget['max_words']} words."
            )
        issues.append(_segment_budget_issue(
            "segment_budget_section_out_of_bounds",
            f"Section '{section['title']}'{id_part} has {section_count} words; "
            f"declared budget is {budget['min_words']}-{budget['max_words']}.",
            suggestion=suggestion,
            metadata=dict(report),
        ))

    return issues, {"document": document_report, "sections": section_reports}


def _segment_budget_result(
    issues: List[DeterministicIssue],
    metadata: Dict[str, Any],
) -> DeterministicCheckResult:
    """Build the pass/fail result for the segment budget guard."""

    passed = not issues
    result_metadata = dict(metadata)
    result_metadata["issues_count"] = len(issues)
    if passed:
        summary = "Text matches the self-declared segment budgets."
    else:
        summary = (
            f"Segment budget validation failed with {len(issues)} issue(s); "
            f"first: {issues[0].message}"
        )
    return DeterministicCheckResult(
        name=SEGMENT_BUDGET_CHECK_NAME,
        score=10.0 if passed else 0.0,
        pass_threshold=10.0,
        passed=passed,
        deal_breaker=not passed,
        summary=summary,
        issues=issues[:_SEGMENT_BUDGET_MAX_ISSUES],
        metadata=result_metadata,
    )


def evaluate_segment_budget_check(
    context: DeterministicValidationContext,
    request: Any,
) -> Optional[DeterministicCheckResult]:
    """Validate the candidate text against its self-declared segment budgets.

    Contract ``segment-budgets/v1`` (opt-in via ``request.segment_budgets``):
    the decoded JSON payload must declare the document budget and the ordered
    sections in ``structure_field``; the target text must reproduce the
    declared headings in order and keep every segment (heading included)
    inside its budget. Returns None when the contract is inactive or the JSON
    guard already rejects the candidate.
    """

    contract = _segment_budget_contract_dict(request)
    if contract is None:
        return None

    structure_field = contract["structure_field"]
    base_heading_level = contract["base_heading_level"]

    if (
        is_json_output_requested(request)
        and context.json_check is not None
        and not context.json_check.passed
    ):
        # The JSON guard already rejects this candidate; avoid duplicate noise.
        return None

    metadata: Dict[str, Any] = {
        "schema_version": _SEGMENT_BUDGET_SCHEMA_VERSION,
        "structure_field": structure_field,
        "base_heading_level": base_heading_level,
    }

    if contract.get("schema_version", _SEGMENT_BUDGET_SCHEMA_VERSION) != _SEGMENT_BUDGET_SCHEMA_VERSION:
        return _segment_budget_result(
            [_segment_budget_issue(
                "segment_budget_structure_invalid",
                f"Unsupported segment budgets schema_version "
                f"'{contract.get('schema_version')}'; expected '{_SEGMENT_BUDGET_SCHEMA_VERSION}'.",
            )],
            metadata,
        )

    payload = _json_payload_from_context(context)
    if not isinstance(payload, dict):
        return _segment_budget_result(
            [_segment_budget_issue(
                "segment_budget_structure_invalid",
                "Candidate is not a JSON object; the segment budgets contract "
                f"requires JSON output declaring '{structure_field}'.",
                suggestion="Return one JSON object with the target text and the "
                "self-declared structure.",
            )],
            metadata,
        )

    document_budget, sections, structure_issues = _validate_segment_budget_structure(
        payload.get(structure_field),
        structure_field=structure_field,
        request_min_words=getattr(request, "min_words", None),
        request_max_words=getattr(request, "max_words", None),
    )
    if structure_issues:
        return _segment_budget_result(structure_issues, metadata)

    text = context.text_for_validation
    text_issues, text_metadata = _evaluate_segment_budget_text(
        text,
        document_budget or {"min_words": 0, "target_words": 1, "max_words": 1},
        sections,
        base_heading_level=base_heading_level,
    )
    metadata.update(text_metadata)
    metadata["sections_checked"] = len(sections)
    return _segment_budget_result(text_issues, metadata)


def evaluate_segment_budget_gate(
    raw_content: str,
    request: Any,
) -> Optional[DeterministicCheckResult]:
    """Run only the segment budget check over one accepted-looking candidate.

    Used by the post-generation gates (main loop and Gran Sabio fallback) so a
    candidate that violates its self-declared budgets is never accepted, even
    when a tool loop exhausted its rounds. Returns None when the contract is
    not active or does not apply to this candidate.
    """

    if _segment_budget_contract_dict(request) is None:
        return None
    context = prepare_validation_context(raw_content, request)
    if context.json_check is not None and not context.json_check.passed:
        return _segment_budget_result(
            [_segment_budget_issue(
                "segment_budget_text_unavailable",
                "Candidate JSON or target text could not be validated for segment budgets.",
                suggestion=context.json_check.summary,
            )],
            {"schema_version": _SEGMENT_BUDGET_SCHEMA_VERSION},
        )
    return evaluate_segment_budget_check(context, request)


def build_segment_budget_gate_evaluations(
    check_result: DeterministicCheckResult,
    qa_models: Optional[List[Any]],
) -> Dict[str, QAEvaluation]:
    """Build the synthetic deal-breaker QA evaluations for a gate rejection.

    Mirrors the algorithmic bypass evaluations (``Arithmos-Prime``) so the
    rejection flows through consensus and iteration feedback exactly like a
    word count deal-breaker, forcing full regeneration with actionable
    feedback instead of SmartEdit.
    """

    detail = _compact_feedback(False, list(check_result.issues), {})
    feedback = (
        "[ALGORITHMIC EVALUATION BY Arithmos-Prime]\n\n"
        "Segment budget contract (segment-budgets/v1) analysis\n"
        "Score: 0.0/10.0\n"
        "Status: VIOLATION\n\n"
        f"{detail}"
    )
    evaluation = QAEvaluation(
        model="Arithmos-Prime (Algorithmic)",
        layer=SEGMENT_BUDGET_CHECK_NAME,
        score=0.0,
        feedback=feedback,
        deal_breaker=True,
        deal_breaker_reason=check_result.summary,
        passes_score=False,
        timestamp=datetime.now(),
        metadata=dict(check_result.metadata),
    )
    model_names = [model if isinstance(model, str) else model.model for model in qa_models or []]
    return {model: evaluation for model in (model_names or ["default"])}


def _compact_feedback(
    approved: bool,
    issues: List[DeterministicIssue],
    checks: Dict[str, DeterministicCheckResult],
    max_items: int = 6,
) -> str:
    """Create a short validator feedback block suitable for model repair prompts."""

    if approved:
        return "All deterministic checks passed."

    lines: List[str] = []
    for issue in issues[:max_items]:
        prefix = "HARD" if issue.hard else issue.severity.upper()
        suggestion = f" Suggestion: {issue.suggestion}" if issue.suggestion else ""
        lines.append(f"- [{prefix}] {issue.message}{suggestion}")

    if not lines:
        for check in checks.values():
            if not check.passed:
                lines.append(f"- {check.summary}")
        if not lines:
            lines.append("- The draft failed deterministic validation.")

    return "\n".join(lines)


def _build_metrics_snapshot(
    context: DeterministicValidationContext,
    checks: Dict[str, DeterministicCheckResult],
    *,
    request: Any = None,
    include_stylistic: bool = False,
) -> Dict[str, Any]:
    """Return a compact metrics payload for tool responses and debugging."""

    text_measurement_requested = bool(
        context.text_for_validation
        or context.target_field_paths
        or checks.get("word_count")
        or checks.get("phrase_frequency")
        or checks.get("lexical_diversity")
        or checks.get("cumulative_repetition")
        or checks.get("segment_budgets")
    )

    metrics: Dict[str, Any] = {}
    if text_measurement_requested:
        metrics["word_count"] = context.word_count
        metrics["target_field_paths"] = context.target_field_paths

    word_count_meta = checks.get("word_count")
    if word_count_meta:
        metrics["word_count_rule"] = {
            "score": word_count_meta.score,
            "actual_count": word_count_meta.metadata.get("actual_count"),
            "required_min": word_count_meta.metadata.get("required_min"),
            "required_max": word_count_meta.metadata.get("required_max"),
            "target_min": word_count_meta.metadata.get("target_min"),
            "target_max": word_count_meta.metadata.get("target_max"),
        }

    lexical = checks.get("lexical_diversity")
    if lexical:
        metrics["lexical_diversity"] = {
            "decision": lexical.metadata.get("decision"),
            "score": lexical.score,
            "adjusted_grades": lexical.metadata.get("adjusted_grades"),
        }

    phrase = checks.get("phrase_frequency")
    if phrase:
        metrics["phrase_frequency"] = {
            "issues_count": len(phrase.metadata.get("issues", [])),
            "rules_total": phrase.metadata.get("rules_total"),
        }

    cumulative = checks.get("cumulative_repetition")
    if cumulative:
        metrics["cumulative_repetition"] = {
            "issues_count": len(cumulative.metadata.get("issues", [])),
            "total_word_count": cumulative.metadata.get("total_word_count"),
        }

    segment_budgets = checks.get("segment_budgets")
    if segment_budgets:
        metrics["segment_budgets"] = {
            "passed": segment_budgets.passed,
            "issues_count": segment_budgets.metadata.get("issues_count"),
            "document": segment_budgets.metadata.get("document"),
            "sections_checked": segment_budgets.metadata.get("sections_checked"),
        }

    json_output = checks.get("json_output")
    if json_output:
        json_metrics = {
            "passed": json_output.passed,
        }
        if context.target_field_paths:
            json_metrics["target_field_paths"] = context.target_field_paths
        metrics["json_output"] = json_metrics

    if include_stylistic:
        metrics["stylistic"] = build_stylistic_metrics_snapshot(
            context.text_for_validation,
            request,
        )

    return metrics


def validate_generation_candidate(
    raw_content: str,
    request: Any,
    *,
    include_json_validation: bool = True,
    json_options: Optional[ValidateOptions] = None,
) -> DraftValidationResult:
    """Run the combined deterministic validation suite for a generated draft.

    Returns a ``DraftValidationResult`` consumable by the shared tool loop.
    Gate fields (``approved``, ``hard_failed``) and ``feedback`` are computed
    here and used internally by the loop; ``visible_payload`` is pre-built
    for the generator scope so the loop can emit it to the LLM without
    re-running validation.
    """

    context = prepare_validation_context(
        raw_content,
        request,
        include_json_validation=include_json_validation,
        json_options=json_options,
    )

    checks: Dict[str, DeterministicCheckResult] = {}
    if context.json_check is not None:
        checks["json_output"] = context.json_check

    if context.text_for_validation:
        word_count_check = evaluate_word_count_check(context.text_for_validation, request)
        if word_count_check:
            checks["word_count"] = word_count_check

        phrase_frequency_check = evaluate_phrase_frequency_check(context.text_for_validation, request)
        if phrase_frequency_check:
            checks["phrase_frequency"] = phrase_frequency_check

        lexical_diversity_check = evaluate_lexical_diversity_check(context.text_for_validation, request)
        if lexical_diversity_check:
            checks["lexical_diversity"] = lexical_diversity_check

        cumulative_repetition_check = evaluate_cumulative_repetition_check(context.text_for_validation, request)
        if cumulative_repetition_check:
            checks["cumulative_repetition"] = cumulative_repetition_check

    # The segment budget check needs the decoded JSON payload, not just the
    # extracted text, so it receives the full context and runs even when no
    # text was extracted (structure failures must still fail hard).
    segment_budget_check = evaluate_segment_budget_check(context, request)
    if segment_budget_check:
        checks["segment_budgets"] = segment_budget_check

    all_issues: List[DeterministicIssue] = []
    scored_checks: List[DeterministicCheckResult] = []
    approved = True
    hard_failed = False

    for check in checks.values():
        all_issues.extend(check.issues)
        scored_checks.append(check)
        if not check.passed:
            approved = False
        if check.deal_breaker or any(issue.hard for issue in check.issues):
            hard_failed = True

    average_score = 10.0
    if scored_checks:
        average_score = round(
            sum(check.score for check in scored_checks) / len(scored_checks),
            2,
        )

    include_stylistic = bool(getattr(request, "include_stylistic_metrics", False))
    metrics_snapshot = _build_metrics_snapshot(
        context,
        checks,
        request=request,
        include_stylistic=include_stylistic,
    )
    feedback = _compact_feedback(approved, all_issues, checks)

    # Cap the exported issues list to the same default as the legacy payload
    # (max_issues=12 in the removed ``to_tool_payload`` helper).
    max_issues = 12
    issues_payload: List[Dict[str, Any]] = [
        {
            "code": issue.code,
            "severity": issue.severity,
            "hard": issue.hard,
            "message": issue.message,
            "suggestion": issue.suggestion,
            "count": issue.count,
        }
        for issue in all_issues[:max_issues]
    ]

    checks_payload: Dict[str, Any] = {
        key: {
            "name": check.name,
            "score": check.score,
            "pass_threshold": check.pass_threshold,
            "passed": check.passed,
            "deal_breaker": check.deal_breaker,
        }
        for key, check in checks.items()
    }

    stylistic_metrics: Optional[Dict[str, Any]] = None
    if include_stylistic:
        stylistic_metrics = build_stylistic_metrics_snapshot(
            context.text_for_validation,
            request,
        )

    result = DraftValidationResult(
        approved=approved,
        hard_failed=hard_failed,
        score=average_score,
        word_count=context.word_count,
        feedback=feedback,
        issues=issues_payload,
        metrics=metrics_snapshot,
        checks=checks_payload,
        stylistic_metrics=stylistic_metrics,
        visible_payload={},
    )
    # Pre-build the generator visible payload so tool-loop tool_responses do
    # not have to call ``build_visible_payload`` on the hot path.
    generator_payload = result.build_visible_payload(PayloadScope.GENERATOR)
    object.__setattr__(result, "visible_payload", generator_payload)
    return result


# ---------------------------------------------------------------------------
# Surface metrics snapshot (opt-in via include_stylistic_metrics)
# ---------------------------------------------------------------------------


def build_stylistic_metrics_snapshot(text: str, request: Any) -> Dict[str, Any]:
    """Compute non-semantic surface telemetry for validator traces.

    This intentionally avoids semantic heuristics such as sentence-start
    repetition or keyword/formula detection. The LLM accent judge owns that
    interpretation when configured.
    """
    # request is accepted so language_hint (or future flags) can be forwarded without
    # touching call sites. Not used today.
    del request

    cleaned = remove_invisible_control(text or "")
    lines = cleaned.splitlines()
    line_lengths = [len(line) for line in lines]
    non_empty_line_lengths = [len(line) for line in lines if line.strip()]
    char_count = len(cleaned)
    non_whitespace_char_count = sum(1 for char in cleaned if not char.isspace())

    punctuation_counts = {
        "period": cleaned.count("."),
        "comma": cleaned.count(","),
        "em_dash": cleaned.count("\u2014"),
        "en_dash": cleaned.count("\u2013"),
        "hyphen": cleaned.count("-"),
        "ellipsis": cleaned.count("\u2026") + cleaned.count("..."),
        "exclamation": cleaned.count("!"),
        "question": cleaned.count("?"),
        "colon": cleaned.count(":"),
        "semicolon": cleaned.count(";"),
    }

    if char_count == 0:
        punctuation_density = {key: 0.0 for key in punctuation_counts}
    else:
        scale = 1000.0 / char_count
        punctuation_density = {
            key: round(value * scale, 3)
            for key, value in punctuation_counts.items()
        }

    layout = {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty_line_lengths),
        "blank_line_count": sum(1 for line in lines if not line.strip()),
        "paragraph_separator_count": cleaned.count("\n\n"),
    }

    line_metrics = {
        "avg_line_chars": round(sum(line_lengths) / len(line_lengths), 3) if line_lengths else 0.0,
        "line_length_stdev": round(statistics.pstdev(line_lengths), 3) if len(line_lengths) > 1 else 0.0,
        "max_line_chars": max(line_lengths, default=0),
        "avg_non_empty_line_chars": (
            round(sum(non_empty_line_lengths) / len(non_empty_line_lengths), 3)
            if non_empty_line_lengths
            else 0.0
        ),
    }

    return {
        "character_counts": {
            "total": char_count,
            "non_whitespace": non_whitespace_char_count,
        },
        "layout": layout,
        "line_metrics": line_metrics,
        "punctuation_counts": punctuation_counts,
        "punctuation_density_per_1000_chars": punctuation_density,
    }
