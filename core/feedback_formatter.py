"""
Utilities for formatting QA feedback and building user-facing reasons.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def _safe_get_evaluation_attr(evaluation: Any, attr: str) -> Any:
    """Safely obtain attributes from either QAEvaluation objects or dicts."""

    if hasattr(evaluation, attr):
        return getattr(evaluation, attr)
    if isinstance(evaluation, dict):
        return evaluation.get(attr)
    return None


def _format_layer_feedback_lines(layer_entries: Any, qa_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Format per-layer feedback lines for prompt contexts."""

    formatted: List[str] = []

    if isinstance(layer_entries, list) and layer_entries:
        for entry in layer_entries:
            if not isinstance(entry, dict):
                continue

            layer_name = entry.get("layer", "Capa")
            avg_score = entry.get("average_score")
            header = f"- {layer_name}"
            if isinstance(avg_score, (int, float)):
                header += f" (promedio {avg_score:.2f}/10)"

            formatted.append(header)

            detail_lines: List[str] = []
            for db_text in entry.get("deal_breakers", []) or []:
                if db_text:
                    detail_lines.append(f"  - deal-breaker: {db_text}")

            for model_entry in entry.get("model_feedback", []) or []:
                if not isinstance(model_entry, dict):
                    continue
                model_name = model_entry.get("model", "Modelo")
                text = (model_entry.get("deal_breaker_reason") or model_entry.get("feedback") or "").strip()
                if text:
                    detail_lines.append(f"  - {model_name}: {text}")

            if detail_lines:
                formatted.extend(detail_lines)
            else:
                formatted.append("  - sin comentarios adicionales")

    if formatted:
        return formatted

    # Fallback: derive layer summary directly from qa_results
    fallback_lines: List[str] = []
    for layer_name, model_results in qa_results.items():
        fallback_lines.append(f"- {layer_name}")
        for model_name, evaluation in model_results.items():
            text = (
                _safe_get_evaluation_attr(evaluation, "deal_breaker_reason")
                or _safe_get_evaluation_attr(evaluation, "reason")
                or _safe_get_evaluation_attr(evaluation, "feedback")
                or ""
            ).strip()
            if text:
                fallback_lines.append(f"  - {model_name}: {text}")

    return fallback_lines


def _fallback_actionable_feedback(qa_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Create actionable feedback list when consensus summary is unavailable."""

    items: List[str] = []
    seen: Set[str] = set()

    for layer_name, model_results in qa_results.items():
        for model_name, evaluation in model_results.items():
            text = (
                _safe_get_evaluation_attr(evaluation, "deal_breaker_reason")
                or _safe_get_evaluation_attr(evaluation, "reason")
                or _safe_get_evaluation_attr(evaluation, "feedback")
                or ""
            ).strip()
            if not text:
                continue

            formatted = f"{layer_name} ({model_name}): {text}"
            if formatted not in seen:
                items.append(formatted)
                seen.add(formatted)

    return items


def _layer_failed_in_iteration(qa_results: Dict[str, Any], layer_name: str) -> bool:
    """Determine whether a specific QA layer failed in the previous iteration."""

    layer_results = qa_results.get(layer_name) or {}
    if not isinstance(layer_results, dict):
        return False
    for evaluation in layer_results.values():
        deal_breaker = _safe_get_evaluation_attr(evaluation, "deal_breaker")
        if deal_breaker:
            return True
        passes_score = _safe_get_evaluation_attr(evaluation, "passes_score")
        if passes_score is False:
            return True
    return False


def _format_top_words_lines(entries: List[Dict[str, Any]], limit: int = 50) -> List[str]:
    """Format repeated words into wrapped lines."""

    truncated = entries[:limit]
    words = [f"{item.get('word')} (x{item.get('count')})" for item in truncated if item.get("word")]
    if not words:
        return []
    lines: List[str] = []
    chunk: List[str] = []
    for idx, token in enumerate(words, start=1):
        chunk.append(token)
        if idx % 10 == 0:
            lines.append(", ".join(chunk))
            chunk = []
    if chunk:
        lines.append(", ".join(chunk))
    return lines


def _format_window_highlights(windows: List[Dict[str, Any]]) -> List[str]:
    """Summarize window-based lexical hotspots."""

    highlights: List[str] = []
    for window in windows[:3]:
        top_words = window.get("top_words") or []
        if not top_words:
            continue
        words_preview = ", ".join(
            f"{item.get('word')} (x{item.get('count')})"
            for item in top_words[:5]
            if item.get("word")
        )
        window_id = window.get("window_id")
        mode = window.get("mode")
        token_count = window.get("token_count")
        preview = (window.get("text_preview") or "").strip()
        highlights.append(f"Window {window_id} ({mode}, {token_count} tokens): {words_preview}")
        if preview:
            snippet = preview.replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200].rstrip() + "..."
            highlights.append(f"Preview: {snippet}")
    return highlights


def _format_lexical_feedback_block(metadata: Dict[str, Any]) -> List[str]:
    """Build lexical diversity remediation block."""

    lines = ["- Lexical Diversity Guard: broaden vocabulary and reduce repetitive wording."]

    decision = metadata.get("decision") or (metadata.get("analysis", {}).get("decision", {}).get("label"))
    if decision:
        lines.append(f"  Current decision: {str(decision).upper()}.")

    grades = metadata.get("adjusted_grades") or metadata.get("analysis", {}).get("grades", {})
    if grades:
        formatted_grades = ", ".join(f"{metric.upper()}={grade}" for metric, grade in sorted(grades.items()))
        lines.append(f"  Metric grades: {formatted_grades}.")

    top_words = metadata.get("top_words_excerpt") or metadata.get("analysis", {}).get("top_words", [])
    word_lines = _format_top_words_lines(top_words)
    if word_lines:
        lines.append("  Top repeated words (word xcount):")
        for entry in word_lines:
            lines.append(f"    {entry}")

    window_excerpt = metadata.get("window_excerpt") or metadata.get("analysis", {}).get("windows", [])
    window_lines = _format_window_highlights(window_excerpt)
    if window_lines:
        lines.append("  Local repetition hotspots:")
        for entry in window_lines:
            lines.append(f"    {entry}")

    return lines


def _format_phrase_feedback_block(metadata: Dict[str, Any]) -> List[str]:
    """Build phrase repetition remediation block."""

    lines = ["- Phrase Frequency Guard: remove or rewrite phrases that exceed repetition limits."]

    issues = metadata.get("issues") or []
    if issues:
        lines.append("  Over-limit phrases:")
        for issue in issues[:10]:
            phrase = issue.get("phrase", "").replace("\n", " ").strip()
            n = issue.get("n")
            count = issue.get("count")
            limit = issue.get("limit")
            severity = issue.get("severity")
            lines.append(
                f"    \"{phrase}\" ({n}-gram) repeats {count} times (limit {limit}, severity {severity})."
            )

    summary = metadata.get("summary_top_by_count") or {}
    if summary:
        lines.append("  Most frequent n-grams:")
        for n_key, items in list(summary.items())[:3]:
            preview = ", ".join(
                f"\"{item.get('text')}\" (x{item.get('count')})"
                for item in items[:5]
                if item.get("text")
            )
            if preview:
                lines.append(f"    n={n_key}: {preview}")

    return lines


def _compose_style_feedback_block(previous_iteration: Optional[Dict[str, Any]]) -> str:
    """Compose unified lexical and repetition remediation block."""

    if not previous_iteration:
        return ""

    qa_results = previous_iteration.get("qa_results") or {}
    lexical_meta = previous_iteration.get("lexical_diversity")
    phrase_meta = previous_iteration.get("phrase_frequency")

    lexical_failed = _layer_failed_in_iteration(qa_results, "Lexical Diversity Guard")
    phrase_failed = _layer_failed_in_iteration(qa_results, "Phrase Frequency Guard")

    include_lexical = bool(lexical_meta) and (lexical_failed or phrase_failed)
    include_phrase = bool(phrase_meta) and phrase_failed

    if not include_lexical and not include_phrase:
        return ""

    lines: List[str] = ["STYLE CHECKPOINTS TO FIX:"]

    if include_lexical and isinstance(lexical_meta, dict):
        lines.extend(_format_lexical_feedback_block(lexical_meta))

    if include_phrase and isinstance(phrase_meta, dict):
        lines.extend(_format_phrase_feedback_block(phrase_meta))

    return "\n".join(lines)


def _extract_deal_breaker_details(qa_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Extract detailed information from QA evaluations that marked deal-breakers.

    Returns a formatted string with comprehensive deal-breaker information
    for use in iteration feedback prompts.
    """
    deal_breaker_entries: List[str] = []

    for layer_name, model_results in qa_results.items():
        for model_name, evaluation in model_results.items():
            # Only process evaluations that flagged deal-breaker
            if not _safe_get_evaluation_attr(evaluation, "deal_breaker"):
                continue

            # Extract all available information
            score = _safe_get_evaluation_attr(evaluation, "score")
            deal_breaker_reason = _safe_get_evaluation_attr(evaluation, "deal_breaker_reason")
            feedback = _safe_get_evaluation_attr(evaluation, "feedback")
            reason = _safe_get_evaluation_attr(evaluation, "reason")  # Backward compat field

            # Build entry header
            entry_lines = [f"[{layer_name}] Evaluator: {model_name}"]

            if score is not None:
                entry_lines.append(f"  Score: {score}/10")

            # Determine the best explanation available
            # Priority: deal_breaker_reason > reason > feedback (if not "Passed")
            primary_explanation = (
                deal_breaker_reason
                or reason
                or (feedback if feedback and feedback.lower() != "passed" else None)
            )

            if primary_explanation:
                entry_lines.append(f"  Critical Issue: {primary_explanation}")
            else:
                entry_lines.append("  Critical Issue: Deal-breaker detected (no specific reason provided)")

            # If we have both deal_breaker_reason AND different feedback, include both
            if deal_breaker_reason and feedback and feedback.lower() != "passed":
                if deal_breaker_reason.strip() != feedback.strip():
                    entry_lines.append(f"  Additional feedback: {feedback}")

            deal_breaker_entries.append("\n".join(entry_lines))

    if not deal_breaker_entries:
        return ""

    return "\n\n".join(deal_breaker_entries)


def create_user_friendly_reason(
    qa_summary: Dict[str, Any],
    qa_results: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    request=None,
    evaluated_layers=None,
) -> str:
    """Create user-friendly rejection reason based on QA results."""

    # Build layer configuration lookup to access min_score for each layer
    layer_config_by_name: Dict[str, Any] = {}
    if evaluated_layers:
        for layer in evaluated_layers:
            if layer and layer.name not in layer_config_by_name:
                layer_config_by_name[layer.name] = layer
    elif request and hasattr(request, "qa_layers"):
        for layer in request.qa_layers:
            if layer and layer.name not in layer_config_by_name:
                layer_config_by_name[layer.name] = layer

    # Extract deal-breaker details and feedback for better messaging
    layer_failures = []

    # Check qa_results for specific layer information and feedback
    for layer_name, model_results in qa_results.items():
        layer_deal_breakers = []
        layer_scores = []
        layer_feedback = []

        # Get the configured min_score for this layer (fallback to 7.0 if not found)
        layer_config = layer_config_by_name.get(layer_name)
        layer_min_score = layer_config.min_score if layer_config else 7.0

        for model, evaluation in model_results.items():
            score = evaluation.score
            if score is not None:
                layer_scores.append(score)
            if evaluation.deal_breaker:
                # Use deal_breaker_reason if available, otherwise use feedback
                reason = evaluation.deal_breaker_reason or evaluation.feedback
                if reason and reason != "None":
                    layer_deal_breakers.append(reason)
            elif evaluation.feedback and (score is None or score < layer_min_score):
                # Collect feedback from low-scoring evaluations (using layer-specific min_score)
                layer_feedback.append(evaluation.feedback)

        if layer_deal_breakers or layer_feedback:
            # Prioritize deal-breaker reasons, fallback to feedback - show full text
            main_reason = "Quality issues detected"
            if layer_deal_breakers:
                main_reason = layer_deal_breakers[0]
            elif layer_feedback:
                main_reason = layer_feedback[0]

            avg_score = sum(layer_scores) / len(layer_scores) if layer_scores else 0

            # Create user-friendly layer name
            friendly_layer_name = layer_name
            if "Word Count" in layer_name:
                friendly_layer_name = "Word count requirements"
            elif "VerificaciÃ³n de Hechos" in layer_name or "Fact" in layer_name:
                friendly_layer_name = "Fact verification"
            elif "Calidad Narrativa" in layer_name or "Narrative" in layer_name:
                friendly_layer_name = "Writing quality"
            elif "PrecisiÃ³n TÃ©cnica" in layer_name or "Technical" in layer_name:
                friendly_layer_name = "Technical accuracy"

            layer_failures.append(f"{friendly_layer_name} (Avg: {avg_score:.1f}/10): {main_reason}")

    # Create user-friendly message
    if iteration >= max_iterations:
        # Final rejection message - show all issues
        if layer_failures:
            issues_text = ". ".join(layer_failures)
            return f"Content rejected after {max_iterations} attempts. Issues detected: {issues_text}"
        else:
            return f"Content rejected after {max_iterations} attempts due to critical quality issues"
    else:
        # Iteration message - show main issue with details
        if layer_failures:
            main_issue = layer_failures[0]
            return f"Retrying (iteration {iteration}/{max_iterations}): {main_issue}"
        else:
            return f"Retrying (iteration {iteration}/{max_iterations}): Quality issues detected"
