"""
Shared deterministic validation helpers for generation and QA bypass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import statistics
from typing import Any, Dict, List, Optional

from json_field_utils import try_extract_json_from_content
from models import WordCountEnforcement, is_json_output_requested
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
