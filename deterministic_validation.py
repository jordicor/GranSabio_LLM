"""
Shared deterministic validation helpers for generation and QA bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import statistics
import traceback
from typing import Any, Dict, List, Optional

from json_field_utils import try_extract_json_from_content
from models import WordCountEnforcement
from tools.ai_json_cleanroom import validate_ai_json
from tools.lexical_diversity_utils import analyze_text_lexical_diversity
from tools.phrase_frequency_utils import (
    analyze_phrase_frequency,
    compute_top_repeated_ngrams,
)
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


@dataclass
class DeterministicValidationReport:
    """Combined validation report used by generation tools and QA bypass."""

    context: DeterministicValidationContext
    checks: Dict[str, DeterministicCheckResult]
    approved: bool
    hard_failed: bool
    score: float
    issues: List[DeterministicIssue]

    @property
    def word_count(self) -> int:
        return self.context.word_count

    def to_tool_payload(self, max_issues: int = 12, request: Any = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tool": "validate_draft",
            "approved": self.approved,
            "hard_failed": self.hard_failed,
            "score": self.score,
            "word_count": self.word_count,
            "feedback": compact_feedback(self),
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "hard": issue.hard,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "count": issue.count,
                }
                for issue in self.issues[:max_issues]
            ],
            "metrics": build_metrics_snapshot(self, request=request),
            "checks": {
                key: {
                    "name": check.name,
                    "score": check.score,
                    "pass_threshold": check.pass_threshold,
                    "passed": check.passed,
                    "deal_breaker": check.deal_breaker,
                }
                for key, check in self.checks.items()
            },
        }
        if getattr(request, "include_stylistic_metrics", False):
            payload["stylistic_metrics"] = build_stylistic_metrics_snapshot(
                self.context.text_for_validation,
                request,
            )
        return payload


def has_active_generation_validators(request: Any) -> bool:
    """Return True when generation has measurable constraints worth validating."""

    if getattr(request, "json_output", False):
        return True
    if getattr(request, "target_field", None):
        return True
    if getattr(request, "min_words", None) or getattr(request, "max_words", None):
        return True
    phrase_frequency = getattr(request, "phrase_frequency", None)
    if phrase_frequency and getattr(phrase_frequency, "enabled", False):
        return True
    lexical_diversity = getattr(request, "lexical_diversity", None)
    if lexical_diversity and getattr(lexical_diversity, "enabled", False):
        return True
    if getattr(request, "cumulative_text", None) and phrase_frequency and getattr(phrase_frequency, "enabled", False):
        return True
    if getattr(request, "include_stylistic_metrics", False):
        return True
    return False


def prepare_validation_context(
    raw_content: str,
    request: Any,
    *,
    include_json_validation: bool = True,
) -> DeterministicValidationContext:
    """Prepare validation text, including JSON extraction when requested."""

    text_for_validation = raw_content or ""
    json_check: Optional[DeterministicCheckResult] = None
    target_field_paths: List[str] = []

    json_output = bool(getattr(request, "json_output", False))
    target_field = getattr(request, "target_field", None)

    if json_output or target_field:
        if include_json_validation and json_output:
            validation_result = validate_ai_json(
                raw_content,
                schema=getattr(request, "json_schema", None),
                expectations=getattr(request, "json_expectations", None),
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

        if (json_output or target_field) and not json_context:
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
            json_check = DeterministicCheckResult(
                name="JSON Output Contract",
                score=10.0,
                pass_threshold=10.0,
                passed=True,
                deal_breaker=False,
                summary="JSON output and target field extraction succeeded.",
                issues=[],
                metadata={
                    "target_field_paths": target_field_paths,
                    "target_field_discovered": json_context.get("target_field_discovered", False),
                },
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

    if not getattr(config, "rules", None):
        logger.warning(
            "phrase_frequency config reached check with enabled=True and rules=[]. "
            "Skipping deterministic phrase-frequency layer. Stack:\n%s",
            "".join(traceback.format_stack(limit=12)),
        )
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
    total_word_count = (
        getattr(request, "cumulative_word_count", None) or count_words(cumulative_text)
    ) + count_words(text)

    issues = result.issues
    deal_breaker = any(issue.severity == "deal_breaker" for issue in issues)
    score = 10.0 if not issues else (2.0 if deal_breaker else max(5.0, 8.0 - (len(issues) * 0.5)))
    passed = score >= 8.0

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
        pass_threshold=8.0,
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


def validate_generation_candidate(
    raw_content: str,
    request: Any,
    *,
    include_json_validation: bool = True,
) -> DeterministicValidationReport:
    """Run the combined deterministic validation suite for a generated draft."""

    context = prepare_validation_context(
        raw_content,
        request,
        include_json_validation=include_json_validation,
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

    return DeterministicValidationReport(
        context=context,
        checks=checks,
        approved=approved,
        hard_failed=hard_failed,
        score=average_score,
        issues=all_issues,
    )


def compact_feedback(report: DeterministicValidationReport, max_items: int = 6) -> str:
    """Create a short validator feedback block suitable for model repair prompts."""

    if report.approved:
        return "All deterministic checks passed."

    lines = []
    for issue in report.issues[:max_items]:
        prefix = "HARD" if issue.hard else issue.severity.upper()
        suggestion = f" Suggestion: {issue.suggestion}" if issue.suggestion else ""
        lines.append(f"- [{prefix}] {issue.message}{suggestion}")

    if not lines:
        for check in report.checks.values():
            if not check.passed:
                lines.append(f"- {check.summary}")
        if not lines:
            lines.append("- The draft failed deterministic validation.")

    return "\n".join(lines)


def build_metrics_snapshot(
    report: DeterministicValidationReport,
    *,
    request: Any = None,
) -> Dict[str, Any]:
    """Return a compact metrics payload for tool responses and debugging."""

    metrics: Dict[str, Any] = {
        "word_count": report.word_count,
        "target_field_paths": report.context.target_field_paths,
    }

    word_count_meta = report.checks.get("word_count")
    if word_count_meta:
        metrics["word_count_rule"] = {
            "score": word_count_meta.score,
            "actual_count": word_count_meta.metadata.get("actual_count"),
            "required_min": word_count_meta.metadata.get("required_min"),
            "required_max": word_count_meta.metadata.get("required_max"),
            "target_min": word_count_meta.metadata.get("target_min"),
            "target_max": word_count_meta.metadata.get("target_max"),
        }

    lexical = report.checks.get("lexical_diversity")
    if lexical:
        metrics["lexical_diversity"] = {
            "decision": lexical.metadata.get("decision"),
            "score": lexical.score,
            "adjusted_grades": lexical.metadata.get("adjusted_grades"),
        }

    phrase = report.checks.get("phrase_frequency")
    if phrase:
        metrics["phrase_frequency"] = {
            "issues_count": len(phrase.metadata.get("issues", [])),
            "rules_total": phrase.metadata.get("rules_total"),
        }

    cumulative = report.checks.get("cumulative_repetition")
    if cumulative:
        metrics["cumulative_repetition"] = {
            "issues_count": len(cumulative.metadata.get("issues", [])),
            "total_word_count": cumulative.metadata.get("total_word_count"),
        }

    json_output = report.checks.get("json_output")
    if json_output:
        metrics["json_output"] = {
            "passed": json_output.passed,
            "target_field_paths": report.context.target_field_paths,
        }

    if getattr(request, "include_stylistic_metrics", False):
        metrics["stylistic"] = build_stylistic_metrics_snapshot(
            report.context.text_for_validation,
            request,
        )

    return metrics


# ---------------------------------------------------------------------------
# Stylistic metrics snapshot (Cambio 1 §4.1 — opt-in via include_stylistic_metrics)
# ---------------------------------------------------------------------------

_STYLISTIC_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s+")
_STYLISTIC_PARAGRAPH_SPLIT_RE = re.compile(r"\n\n+")
_STYLISTIC_WORD_TOKEN_RE = re.compile(r"\w+(?:['\u2019]\w+)*", re.UNICODE)


def _split_sentences_for_stylistic(text: str) -> List[str]:
    """Split text into sentences using a simple punctuation-plus-space heuristic."""
    if not text:
        return []
    raw_parts = _STYLISTIC_SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in raw_parts if part and part.strip()]


def _split_paragraphs_for_stylistic(text: str) -> List[str]:
    """Split text into paragraphs using blank-line separators."""
    if not text:
        return []
    raw_parts = _STYLISTIC_PARAGRAPH_SPLIT_RE.split(text)
    return [part.strip() for part in raw_parts if part and part.strip()]


def _count_words_stylistic(text: str) -> int:
    if not text:
        return 0
    return len(_STYLISTIC_WORD_TOKEN_RE.findall(text))


def _percentile_from_sorted(values_sorted: List[float], percentile: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not values_sorted:
        return 0.0
    if len(values_sorted) == 1:
        return float(values_sorted[0])
    rank = (percentile / 100.0) * (len(values_sorted) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values_sorted) - 1)
    fraction = rank - lower
    return float(values_sorted[lower] + (values_sorted[upper] - values_sorted[lower]) * fraction)


def build_stylistic_metrics_snapshot(text: str, request: Any) -> Dict[str, Any]:
    """Compute the cadence/openings/n-gram/punctuation stylistic snapshot.

    Opt-in via `request.include_stylistic_metrics=True`. Informational telemetry only.
    Edge cases match proposal Cambio 1 v5 §4.3.
    """
    # request is accepted so language_hint (or future flags) can be forwarded without
    # touching call sites. Not used today.
    del request

    cleaned = remove_invisible_control(text or "")

    # Cadence ----------------------------------------------------------------
    sentences = _split_sentences_for_stylistic(cleaned)
    sentence_lengths = [_count_words_stylistic(sentence) for sentence in sentences]
    sentence_count = len(sentences)

    if sentence_count == 0:
        cadence: Dict[str, Any] = {
            "sentence_count": 0,
            "avg_sentence_words": 0.0,
            "sentence_length_stdev": 0.0,
            "sentence_length_percentiles": {"p10": 0.0, "p50": 0.0, "p90": 0.0},
            "burstiness_ratio": 0.0,
            "longest_run_same_length": 0,
            "repeated_sentence_starts": [],
        }
    else:
        avg_sentence_words = sum(sentence_lengths) / sentence_count
        if sentence_count == 1:
            sentence_length_stdev = 0.0
            percentiles = {
                "p10": round(avg_sentence_words, 3),
                "p50": round(avg_sentence_words, 3),
                "p90": round(avg_sentence_words, 3),
            }
            longest_run_same_length = 1
        else:
            sentence_length_stdev = statistics.pstdev(sentence_lengths)
            sorted_lengths = sorted(sentence_lengths)
            percentiles = {
                "p10": round(_percentile_from_sorted(sorted_lengths, 10), 3),
                "p50": round(_percentile_from_sorted(sorted_lengths, 50), 3),
                "p90": round(_percentile_from_sorted(sorted_lengths, 90), 3),
            }
            longest_run_same_length = 1
            current_run = 1
            for i in range(1, len(sentence_lengths)):
                if sentence_lengths[i] == sentence_lengths[i - 1]:
                    current_run += 1
                    if current_run > longest_run_same_length:
                        longest_run_same_length = current_run
                else:
                    current_run = 1

        burstiness_ratio = (
            sentence_length_stdev / avg_sentence_words if avg_sentence_words > 0 else 0.0
        )

        repeated_sentence_starts: List[Dict[str, Any]] = []
        if sentence_count >= 6:
            start_indices: Dict[str, List[int]] = {}
            for idx, sentence in enumerate(sentences):
                match = _STYLISTIC_WORD_TOKEN_RE.search(sentence)
                if not match:
                    continue
                first_word = match.group(0).lower()
                if len(first_word) < 3:
                    continue
                start_indices.setdefault(first_word, []).append(idx)
            filtered_starts = [
                (word, indices) for word, indices in start_indices.items() if len(indices) >= 3
            ]
            filtered_starts.sort(key=lambda item: (-len(item[1]), item[0]))
            for word, indices in filtered_starts:
                repeated_sentence_starts.append(
                    {
                        "word": word,
                        "count": len(indices),
                        "indices": indices[:10],
                    }
                )

        cadence = {
            "sentence_count": sentence_count,
            "avg_sentence_words": round(avg_sentence_words, 3),
            "sentence_length_stdev": round(sentence_length_stdev, 3),
            "sentence_length_percentiles": percentiles,
            "burstiness_ratio": round(burstiness_ratio, 3),
            "longest_run_same_length": longest_run_same_length,
            "repeated_sentence_starts": repeated_sentence_starts,
        }

    # Openings ---------------------------------------------------------------
    paragraphs = _split_paragraphs_for_stylistic(cleaned)
    paragraph_openings: List[Dict[str, Any]] = []
    for index, paragraph in enumerate(paragraphs[:10]):
        paragraph_tokens = _STYLISTIC_WORD_TOKEN_RE.findall(paragraph)
        first_tokens = paragraph_tokens[:5]
        first_words = " ".join(first_tokens)
        if len(first_words) > 160:
            first_words = first_words[:160]
        paragraph_openings.append(
            {
                "index": index,
                "first_words": first_words,
                "word_count": len(paragraph_tokens),
            }
        )

    openings = {
        "paragraph_count_total": len(paragraphs),
        "paragraph_openings": paragraph_openings,
    }

    # Extended n-gram fingerprint --------------------------------------------
    top_repeated_ngrams = compute_top_repeated_ngrams(cleaned)

    # Punctuation density (per 1000 words) -----------------------------------
    word_count = _count_words_stylistic(cleaned)
    em_dash_count = cleaned.count("\u2014")
    en_dash_count = cleaned.count("\u2013")
    hyphen_count = cleaned.count("-")
    ellipsis_count = cleaned.count("\u2026") + cleaned.count("...")
    exclamation_count = cleaned.count("!")
    question_count = cleaned.count("?")
    colon_count = cleaned.count(":")
    semicolon_count = cleaned.count(";")

    if word_count == 0:
        punctuation_density = {
            "em_dash": 0.0,
            "en_dash": 0.0,
            "hyphen": 0.0,
            "ellipsis": 0.0,
            "exclamation": 0.0,
            "question": 0.0,
            "colon": 0.0,
            "semicolon": 0.0,
        }
    else:
        scale = 1000.0 / word_count
        punctuation_density = {
            "em_dash": round(em_dash_count * scale, 3),
            "en_dash": round(en_dash_count * scale, 3),
            "hyphen": round(hyphen_count * scale, 3),
            "ellipsis": round(ellipsis_count * scale, 3),
            "exclamation": round(exclamation_count * scale, 3),
            "question": round(question_count * scale, 3),
            "colon": round(colon_count * scale, 3),
            "semicolon": round(semicolon_count * scale, 3),
        }

    sentences_with_em_dash = sum(1 for sentence in sentences if "\u2014" in sentence)

    return {
        "cadence": cadence,
        "openings": openings,
        "top_repeated_ngrams": top_repeated_ngrams,
        "punctuation_density": punctuation_density,
        "sentences_with_em_dash": sentences_with_em_dash,
    }
