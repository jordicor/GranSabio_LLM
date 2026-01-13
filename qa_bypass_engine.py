"""
QA Bypass Engine for Gran Sabio LLM Engine
===========================================

Algorithmic bypass system for automatic deal-breaker detection,
avoiding expensive AI model calls for predictable criteria violations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models import QALayer, QAEvaluation
from word_count_utils import (
    check_word_count_compliance,
    count_words,
    is_word_count_enforcement_enabled,
)
from tools.lexical_diversity_utils import analyze_text_lexical_diversity
from tools.phrase_frequency_utils import analyze_phrase_frequency


logger = logging.getLogger(__name__)


class QABypassEngine:
    """
    Algorithmic QA Bypass Engine
    
    This engine handles predictable deal-breaker scenarios without requiring
    expensive AI model calls. Currently supports:
    - Word count enforcement with deal-breaker severity
    """
    
    def __init__(self):
        """Initialize the bypass engine"""
        # The Arithmoi - mathematical entities that can multiply to enforce numerical rules
        self.evaluator_names = [
            "Arithmos-Prime",
            "Arithmos-Beta",
            "Arithmos-Gamma",
            "Arithmos-Delta",
            "Arithmos-Epsilon"
        ]
        self.phrase_evaluator_names = [
            "FraseGuard-Alpha",
            "FraseGuard-Beta",
            "FraseGuard-Gamma",
            "FraseGuard-Delta",
            "FraseGuard-Epsilon",
        ]
        self.lexical_evaluator_names = [
            "LexiGuard-Alpha",
            "LexiGuard-Beta",
            "LexiGuard-Gamma",
            "LexiGuard-Delta",
            "LexiGuard-Epsilon",
        ]
        self.cumulative_evaluator_names = [
            "CumulGuard-Alpha",
            "CumulGuard-Beta",
            "CumulGuard-Gamma",
            "CumulGuard-Delta",
            "CumulGuard-Epsilon",
        ]
    
    def can_bypass_layer(self, layer: QALayer, original_request: Any) -> bool:
        """
        Determine if a QA layer can be bypassed algorithmically

        Args:
            layer: QA layer to evaluate
            original_request: Original content request

        Returns:
            True if this layer can be bypassed algorithmically
        """
        # ALWAYS bypass word count enforcement layers algorithmically
        # The severity setting controls whether violations trigger deal_breaker
        if layer.name == "Word Count Enforcement":
            config = getattr(original_request, "word_count_enforcement", None)
            if is_word_count_enforcement_enabled(config):
                return True

        if (layer.name == "Phrase Frequency Guard" and
            hasattr(original_request, 'phrase_frequency') and
            original_request.phrase_frequency and
            getattr(original_request.phrase_frequency, 'enabled', False)):
            return True

        if (layer.name == "Lexical Diversity Guard" and
            hasattr(original_request, 'lexical_diversity') and
            original_request.lexical_diversity and
            getattr(original_request.lexical_diversity, 'enabled', False)):
            return True

        if (layer.name == "Cumulative Repetition Guard" and
            hasattr(original_request, 'cumulative_text') and
            original_request.cumulative_text and
            hasattr(original_request, 'phrase_frequency') and
            original_request.phrase_frequency and
            getattr(original_request.phrase_frequency, 'enabled', False)):
            return True

        return False
    
    def bypass_layer_evaluation(
        self, 
        content: str, 
        layer: QALayer, 
        qa_models: List[str],
        original_request: Any
    ) -> Dict[str, QAEvaluation]:
        """
        Perform algorithmic bypass evaluation for a layer
        
        Args:
            content: Content to evaluate
            layer: QA layer configuration
            qa_models: List of QA models that would normally evaluate
            original_request: Original content request
            
        Returns:
            Dictionary of {model: QAEvaluation} simulating normal QA results
        """
        if layer.name == "Word Count Enforcement":
            return self._bypass_word_count_evaluation(content, layer, qa_models, original_request)
        if layer.name == "Phrase Frequency Guard":
            return self._bypass_phrase_frequency_evaluation(content, layer, qa_models, original_request)
        if layer.name == "Lexical Diversity Guard":
            return self._bypass_lexical_diversity_evaluation(content, layer, qa_models, original_request)
        if layer.name == "Cumulative Repetition Guard":
            return self._bypass_cumulative_repetition_evaluation(content, layer, qa_models, original_request)

        # Fallback - should not happen if can_bypass_layer is correct
        logger.warning(f"Attempted bypass for unsupported layer: {layer.name}")
        return {}
    
    def _bypass_word_count_evaluation(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[str],
        original_request: Any
    ) -> Dict[str, QAEvaluation]:
        """
        Perform algorithmic word count evaluation with gradual scoring

        Args:
            content: Content to evaluate
            layer: Word count QA layer
            qa_models: List of QA models to simulate
            original_request: Original content request

        Returns:
            Dictionary simulating QA evaluations with gradual scoring
        """
        # Extract word count requirements
        min_words = getattr(original_request, 'min_words', None)
        max_words = getattr(original_request, 'max_words', None)
        word_count_config = original_request.word_count_enforcement

        # Convert WordCountEnforcement model to dict if needed
        if hasattr(word_count_config, 'model_dump'):
            word_count_config_dict = word_count_config.model_dump()
        elif hasattr(word_count_config, 'dict'):
            word_count_config_dict = word_count_config.dict()
        else:
            word_count_config_dict = word_count_config

        # Check compliance algorithmically (now includes gradual score and target_field support)
        compliance_result = check_word_count_compliance(
            content, min_words, max_words, word_count_config_dict
        )

        actual_count = compliance_result['actual_count']
        required_min = compliance_result['required_min']
        required_max = compliance_result['required_max']
        target_min = compliance_result['target_min']
        target_max = compliance_result['target_max']
        score = compliance_result['score']  # Gradual score from 0 to 10
        severity = compliance_result['severity']

        # Determine deal_breaker status:
        # A score of 0 means the content is outside the absolute range.
        # Treat it as a deal-breaker regardless of configured severity so downstream
        # approval logic never publishes truncated or oversized chapters.
        deal_breaker = False
        deal_breaker_reason = None

        if score == 0.0:
            deal_breaker = True
            deal_breaker_reason = f"word count is outside the acceptable range {required_min}-{required_max} words"

        # Build detailed feedback based on score
        if score == 10.0:
            # Perfect compliance
            status_emoji = "✅"
            status_text = "PERFECT COMPLIANCE"
            detail = (
                f"The content length is within the ideal target range and meets "
                f"all word count requirements perfectly."
            )
        elif score >= 5.0:
            # Within flexibility buffer, acceptable but not ideal
            status_emoji = "⚠️"
            status_text = "ACCEPTABLE (within flexibility buffer)"
            if actual_count < target_min:
                shortfall = target_min - actual_count
                detail = (
                    f"Content is {shortfall} words below the ideal minimum of {target_min}, "
                    f"but still within acceptable flexibility range. Score: {score:.1f}/10"
                )
            elif actual_count > target_max:
                excess = actual_count - target_max
                detail = (
                    f"Content exceeds the ideal maximum of {target_max} by {excess} words, "
                    f"but still within acceptable flexibility range. Score: {score:.1f}/10"
                )
            else:
                detail = f"Content is acceptable with score {score:.1f}/10"
        elif score > 0.0:
            # Close to limits, poor but not completely failed
            status_emoji = "⚠️"
            status_text = "MARGINAL (near flexibility limit)"
            if actual_count < target_min:
                shortfall = target_min - actual_count
                detail = (
                    f"Content is {shortfall} words below ideal minimum and approaching "
                    f"the flexibility limit. Score: {score:.1f}/10. Improvement recommended."
                )
            elif actual_count > target_max:
                excess = actual_count - target_max
                detail = (
                    f"Content exceeds ideal maximum by {excess} words and is approaching "
                    f"the flexibility limit. Score: {score:.1f}/10. Improvement recommended."
                )
            else:
                detail = f"Content is near the acceptable limit with score {score:.1f}/10"
        else:
            # Complete failure - outside flexibility range
            status_emoji = "❌"
            status_text = "VIOLATION"
            if actual_count < required_min:
                shortfall = required_min - actual_count
                detail = (
                    f"Content is {shortfall} words below the absolute minimum of {required_min}. "
                    f"This exceeds the allowed flexibility and requires regeneration."
                )
            else:
                excess = actual_count - required_max
                detail = (
                    f"Content exceeds the absolute maximum of {required_max} by {excess} words. "
                    f"This exceeds the allowed flexibility and requires regeneration."
                )

        if deal_breaker and detail:
            deal_breaker_reason = detail

        # Use first evaluator name for algorithmic evaluation
        evaluator_name = self.evaluator_names[0]

        feedback = (
            f"[ALGORITHMIC EVALUATION BY {evaluator_name}]\n\n"
            f"Word count analysis: {actual_count} words\n"
            f"Target range: {target_min or 'N/A'}-{target_max or 'N/A'} words\n"
            f"Flexibility range: {required_min}-{required_max} words\n"
            f"Score: {score:.1f}/10.0\n"
            f"Status: {status_emoji} {status_text}\n\n"
            f"{detail}"
        )

        # Create single QA evaluation (algorithmic analysis is done once, not per model)
        evaluation = QAEvaluation(
            model=f"{evaluator_name} (Algorithmic)",
            layer=layer.name,
            score=score,
            feedback=feedback,
            deal_breaker=deal_breaker,
            deal_breaker_reason=deal_breaker_reason,
            passes_score=score >= layer.min_score,
            timestamp=datetime.now()
        )

        logger.info(f"Algorithmic bypass - {layer.name} via {evaluator_name}: "
                   f"Score {score:.1f}, Deal-breaker: {deal_breaker}")

        # Replicate the same evaluation for all models (system expects one per model)
        evaluations = {model: evaluation for model in qa_models}

        return evaluations

    def _bypass_phrase_frequency_evaluation(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[str],
        original_request: Any
    ) -> Dict[str, QAEvaluation]:
        """Perform algorithmic phrase frequency evaluation."""

        config = getattr(original_request, "phrase_frequency", None)
        if not config or not getattr(config, "enabled", False):
            logger.warning("Phrase frequency layer requested but configuration is missing or disabled")
            return {}

        settings = config.to_settings()
        result = analyze_phrase_frequency(content, settings)

        issues = result.issues
        deal_breaker_issues = [issue for issue in issues if issue.severity == "deal_breaker"]

        if not issues:
            score = 10.0
            deal_breaker = False
            summary_lines = [
                "No se detectaron repeticiones problemáticas en las reglas configuradas.",
                f"Reglas analizadas: {len(settings.rules)}",
            ]
            deal_breaker_reason = None
        else:
            deal_breaker = bool(deal_breaker_issues)
            score = 2.0 if deal_breaker else max(5.0, 8.0 - (len(issues) * 0.5))
            summary_lines = [
                "Frases con repeticiones por encima del límite:",
            ]
            for issue in issues[:10]:
                severity_label = "DEAL-BREAKER" if issue.severity == "deal_breaker" else "AVISO"
                ratio_text = (
                    f" (ratio tokens {issue.repeat_ratio_tokens:.3f})"
                    if issue.repeat_ratio_tokens is not None
                    else ""
                )
                rule_name = issue.rule_label or "regla"
                summary_lines.append(
                    f"- [{severity_label}] '{issue.phrase}' ({issue.n} tokens) aparece {issue.count} veces (límite {issue.limit}) [{rule_name}]{ratio_text}."
                )
                if issue.guidance:
                    summary_lines.append(f"  · Recomendación: {issue.guidance}")

            if len(issues) > 10:
                summary_lines.append(f"(Se detectaron {len(issues) - 10} frases adicionales por encima del límite)")

            first_issue = deal_breaker_issues[0] if deal_breaker_issues else issues[0]
            deal_breaker_reason = (
                f"'{first_issue.phrase}' excede el límite ({first_issue.count}/{first_issue.limit})"
                if first_issue
                else None
            )

        meta = result.analyzer_output.get("meta", {}) if result.analyzer_output else {}
        total_tokens = meta.get("total_tokens")
        if total_tokens:
            summary_lines.append(f"Tokens analizados: {total_tokens}")

        feedback = (
            "[ALGORITHMIC EVALUATION BY PhraseGuard]\n"
            + "\n".join(summary_lines)
        )

        summary_counts_trimmed: Dict[str, Any] = {}
        if result.analyzer_output:
            summary_counts = result.analyzer_output.get("summary", {}).get("top_by_count", {})
            for n_key, items in summary_counts.items():
                truncated = items[: min(10, len(items))]
                summary_counts_trimmed[str(n_key)] = truncated

        issues_payload = [
            {
                "phrase": issue.phrase,
                "n": issue.n,
                "count": issue.count,
                "limit": issue.limit,
                "severity": issue.severity,
                "rule": issue.rule_label,
            }
            for issue in issues
        ]

        metadata_payload = {
            "meta": meta,
            "issues": issues_payload,
            "summary_top_by_count": summary_counts_trimmed,
            "rules_total": len(settings.rules),
        }

        # Use first evaluator name for algorithmic evaluation
        evaluator_name = self.phrase_evaluator_names[0]

        # Create single QA evaluation (algorithmic analysis is done once, not per model)
        evaluation = QAEvaluation(
            model=f"{evaluator_name} (Algorithmic)",
            layer=layer.name,
            score=score,
            feedback=feedback,
            deal_breaker=deal_breaker,
            deal_breaker_reason=deal_breaker_reason,
            passes_score=score >= layer.min_score,
            timestamp=datetime.now(),
            metadata=metadata_payload,
        )

        logger.info(
            "Algorithmic bypass - %s via PhraseGuard: Score %.1f, Deal-breaker: %s",
            layer.name,
            score,
            deal_breaker,
        )

        # Replicate the same evaluation for all models (system expects one per model)
        evaluations = {model: evaluation for model in qa_models}

        return evaluations

    def _bypass_lexical_diversity_evaluation(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[str],
        original_request: Any
    ) -> Dict[str, QAEvaluation]:
        """Perform algorithmic lexical diversity evaluation."""

        config = getattr(original_request, "lexical_diversity", None)
        if not config or not getattr(config, "enabled", False):
            logger.warning("Lexical diversity layer requested but configuration is missing or disabled")
            return {}

        settings = config.to_settings()
        result = analyze_text_lexical_diversity(content, settings)

        decision_label = result.decision_label
        analysis = result.analysis or {}
        metrics = analysis.get("metrics", {})
        grades = {k: v for k, v in result.adjusted_grades.items()}
        meta = analysis.get("meta", {})
        top_words = analysis.get("top_words", []) or []
        windows = analysis.get("windows", []) or []

        top_words_limit = settings.top_words_k if settings.top_words_k > 0 else 50
        top_words_excerpt = [
            {
                "word": entry.get("word"),
                "count": entry.get("count"),
                "freq": entry.get("freq"),
            }
            for entry in top_words[: top_words_limit or 50]
        ]

        window_excerpt: List[Dict[str, Any]] = []
        for window in windows[:3]:
            window_excerpt.append(
                {
                    "window_id": window.get("window_id"),
                    "mode": window.get("mode"),
                    "token_count": window.get("token_count"),
                    "top_words": window.get("top_words"),
                    "text_preview": window.get("text_preview"),
                }
            )

        metrics_summary = ", ".join(
            f"{name}: {value:.2f}" for name, value in sorted(metrics.items())
        ) if metrics else "no metrics reported"

        grades_summary = ", ".join(
            f"{metric.upper()}={grade}" for metric, grade in sorted(grades.items())
        ) if grades else "no grades available"

        language_hint = (
            meta.get("language_hint")
            or settings.language
            or getattr(original_request, "language", None)
            or "unspecified"
        )
        stop_words_filtered = meta.get("stop_words_filtered", False)

        feedback_lines = [
            "[ALGORITHMIC EVALUATION BY LexiGuard]",
            f"Language hint: {language_hint} (stop-word filter: {'on' if stop_words_filtered else 'off'})",
            f"Decision: {decision_label} (score {result.score:.1f}/10)",
            f"Grading summary: {grades_summary}",
            f"Metric snapshot: {metrics_summary}",
        ]

        if top_words_excerpt:
            top_preview = ", ".join(
                f"{entry['word']} (x{entry['count']})" for entry in top_words_excerpt[:10]
            )
            feedback_lines.append(f"Most repeated words (top 10 preview): {top_preview}")

        if result.deal_breaker:
            feedback_lines.append("Deal-breaker triggered by configured lexical diversity policy.")

        if window_excerpt:
            for window in window_excerpt:
                top_window_words = window.get("top_words") or []
                if not top_window_words:
                    continue
                words_preview = ", ".join(
                    f"{item.get('word')} (x{item.get('count')})" for item in top_window_words[:5]
                )
                feedback_lines.append(
                    f"Window {window.get('window_id')} ({window.get('mode')} mode, {window.get('token_count')} tokens): {words_preview}"
                )

        deal_breaker_reason = (
            "Lexical diversity thresholds reached RED decision."
            if result.deal_breaker
            else None
        )

        metadata_payload = {
            "decision": decision_label,
            "score": result.score,
            "deal_breaker": result.deal_breaker,
            "adjusted_grades": grades,
            "analysis": analysis,
            "top_words_excerpt": top_words_excerpt,
            "window_excerpt": window_excerpt,
            "meta": meta,
        }

        # Use first evaluator name for algorithmic evaluation
        evaluator_name = self.lexical_evaluator_names[0]

        # Create single QA evaluation (algorithmic analysis is done once, not per model)
        evaluation = QAEvaluation(
            model=f"{evaluator_name} (Algorithmic)",
            layer=layer.name,
            score=result.score,
            feedback="\n".join(feedback_lines),
            deal_breaker=result.deal_breaker,
            deal_breaker_reason=deal_breaker_reason,
            passes_score=result.score >= layer.min_score,
            timestamp=datetime.now(),
            metadata=metadata_payload,
        )

        logger.info(
            "Algorithmic bypass - %s via LexiGuard: Score %.1f, Deal-breaker: %s",
            layer.name,
            result.score,
            result.deal_breaker,
        )

        # Replicate the same evaluation for all models (system expects one per model)
        evaluations = {model: evaluation for model in qa_models}

        return evaluations

    def _bypass_cumulative_repetition_evaluation(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[str],
        original_request: Any
    ) -> Dict[str, QAEvaluation]:
        """Perform algorithmic cumulative repetition evaluation."""

        cumulative_text = getattr(original_request, "cumulative_text", None)
        cumulative_word_count = getattr(original_request, "cumulative_word_count", None)
        config = getattr(original_request, "phrase_frequency", None)

        if not cumulative_text or not config or not getattr(config, "enabled", False):
            logger.warning("Cumulative repetition layer requested but required data is missing")
            return {}

        # Combine cumulative text with new content
        full_text = cumulative_text + "\n\n" + content
        total_word_count = cumulative_word_count + count_words(content) if cumulative_word_count else len(full_text.split())

        # Convert configuration to settings and analyze
        # NOTE: compute_effective_limit() in PhraseFrequencyRuleSpec automatically
        # applies ratio-based limits based on text length, no manual calculation needed
        settings = config.to_settings()
        result = analyze_phrase_frequency(full_text, settings)

        issues = result.issues
        deal_breaker_issues = [issue for issue in issues if issue.severity == "deal_breaker"]

        if not issues:
            score = 10.0
            deal_breaker = False
            summary_lines = [
                "[CUMULATIVE ANALYSIS] No repetition issues detected across accumulated chapters.",
                f"Total accumulated words analyzed: {total_word_count}",
                f"Rules analyzed: {len(settings.rules)}",
            ]
            deal_breaker_reason = None
        else:
            deal_breaker = bool(deal_breaker_issues)
            score = 2.0 if deal_breaker else max(5.0, 8.0 - (len(issues) * 0.5))
            summary_lines = [
                "[CUMULATIVE ANALYSIS] Excessive repetition detected across accumulated chapters:",
            ]
            for issue in issues[:10]:
                severity_label = "DEAL-BREAKER" if issue.severity == "deal_breaker" else "WARNING"
                ratio_text = (
                    f" (ratio {issue.repeat_ratio_tokens:.3f})"
                    if issue.repeat_ratio_tokens is not None
                    else ""
                )
                rule_name = issue.rule_label or "rule"
                summary_lines.append(
                    f"- [{severity_label}] '{issue.phrase}' ({issue.n} tokens) appears {issue.count} times "
                    f"across accumulated text (limit {issue.limit}) [{rule_name}]{ratio_text}."
                )
                if issue.guidance:
                    summary_lines.append(f"  · Recommendation: {issue.guidance}")

            if len(issues) > 10:
                summary_lines.append(f"({len(issues) - 10} additional phrases exceed limits)")

            summary_lines.append(f"\nTotal words analyzed (cumulative): {total_word_count}")

            first_issue = deal_breaker_issues[0] if deal_breaker_issues else issues[0]
            deal_breaker_reason = (
                f"Cumulative repetition: '{first_issue.phrase}' exceeds limit ({first_issue.count}/{first_issue.limit})"
                if first_issue
                else None
            )

        feedback = "\n".join(summary_lines)

        # Build metadata payload
        issues_payload = [
            {
                "phrase": issue.phrase,
                "n": issue.n,
                "count": issue.count,
                "limit": issue.limit,
                "severity": issue.severity,
                "rule_label": issue.rule_label,
            }
            for issue in issues[:20]
        ]

        metadata_payload = {
            "issues": issues_payload,
            "total_issues": len(issues),
            "deal_breaker_issues": len(deal_breaker_issues),
            "cumulative_word_count": total_word_count,
            "analyzer_output": result.analyzer_output if result.analyzer_output else {},
        }

        # Use first evaluator name for algorithmic evaluation
        evaluator_name = self.cumulative_evaluator_names[0]

        # Create single QA evaluation (algorithmic analysis is done once, not per model)
        evaluation = QAEvaluation(
            model=f"{evaluator_name} (Algorithmic)",
            layer=layer.name,
            score=score,
            feedback=f"[ALGORITHMIC EVALUATION BY {evaluator_name}]\n\n{feedback}",
            deal_breaker=deal_breaker,
            deal_breaker_reason=deal_breaker_reason,
            passes_score=score >= layer.min_score,
            timestamp=datetime.now(),
            metadata=metadata_payload,
        )

        logger.info(
            "Algorithmic bypass - %s via CumulGuard: Score %.1f, Deal-breaker: %s, Cumulative words: %d",
            layer.name,
            score,
            deal_breaker,
            total_word_count,
        )

        # Replicate the same evaluation for all models (system expects one per model)
        evaluations = {model: evaluation for model in qa_models}

        return evaluations

    def should_bypass_qa_layer(
        self, 
        layer: QALayer, 
        original_request: Any,
        extra_verbose: bool = False
    ) -> bool:
        """
        Determine if QA evaluation should be bypassed for this layer
        
        Args:
            layer: QA layer to check
            original_request: Original content request
            extra_verbose: Whether extra verbose logging is enabled
            
        Returns:
            True if QA should be bypassed algorithmically
        """
        should_bypass = self.can_bypass_layer(layer, original_request)
        
        if should_bypass and extra_verbose:
            logger.info(f"QA Bypass Engine: Layer '{layer.name}' will be evaluated algorithmically")
        
        return should_bypass
