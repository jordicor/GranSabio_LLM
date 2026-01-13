"""Utility helpers for phrase repetition QA analysis."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from tools.repetition_analyzer import (
    AnalysisConfig,
    analyze_text,
    tokenize_word_punct_with_spans,
    untokenize,
)
from tools.stopwords_utils import get_stopwords_for_language, resolve_language_hint

logger = logging.getLogger(__name__)


def calculate_dynamic_multipliers(token_count: int, n_gram_size: int) -> tuple[float, float]:
    """
    Calculate dynamic adjustment multipliers using V3 Hybrid × Rational formula.

    This formula provides continuous, smooth adjustment without language-specific keywords.
    Based on universal linguistic properties: smaller n-grams are naturally more frequent.

    Args:
        token_count: Total number of tokens in the text
        n_gram_size: Size of the n-gram (2 for bigrams, 3 for trigrams, etc.)

    Returns:
        Tuple of (length_multiplier, ngram_multiplier)
    """
    # Length adjustment: Hybrid exponential-rational
    # Short texts (exponential dominates) -> smooth transition -> long texts (rational dominates)
    exp_weight = math.exp(-token_count / 1000)  # Fades for long texts
    rat_weight = 1 - exp_weight

    # Exponential part: strong boost for very short texts
    exp_part = 1.0 + 2.5 * math.exp(-token_count / 300)

    # Rational part: moderate boost that decays smoothly
    rat_part = 1.0 + 1.5 * (500 / (token_count + 500))

    # Weighted combination for smooth transition
    length_mult = exp_weight * exp_part + rat_weight * rat_part

    # N-gram adjustment: Rational decay
    # Based on universal property that bigrams are more frequent than longer n-grams
    ngram_mult = 4.5 / (n_gram_size + 1.5)

    return length_mult, ngram_mult


def get_minimum_floor(token_count: int, n_gram_size: int) -> int:
    """
    Get minimum repetition floor for edge cases.

    Very short texts need minimum repetitions for natural language flow,
    regardless of calculated ratios.

    Args:
        token_count: Total number of tokens
        n_gram_size: Size of the n-gram

    Returns:
        Minimum allowed repetitions
    """
    if n_gram_size == 2 and token_count < 300:
        return 3  # Bigrams in very short texts need at least 3 occurrences
    elif n_gram_size <= 4 and token_count < 500:
        return 2  # Short texts need at least 2 occurrences for trigrams/4-grams
    else:
        return 1  # Default minimum


@dataclass
class PhraseFrequencyRuleSpec:
    """Configuration for a single phrase repetition rule."""

    max_repetitions: Optional[int] = None
    max_ratio_tokens: Optional[float] = None
    max_count_absolute: Optional[int] = None
    min_length: int = 2
    max_length: Optional[int] = None
    severity: str = "warn"
    phrase: Optional[str] = None
    label: Optional[str] = None
    guidance: Optional[str] = None

    def compute_effective_limit(self, total_tokens: int) -> int:
        """
        Compute effective repetition limit with dynamic adjustment.

        Uses V3 Hybrid × Rational formula for smooth, continuous adjustment
        based on text length and n-gram size. Works for all languages without
        requiring language-specific keywords.

        Args:
            total_tokens: Total number of tokens in the text

        Returns:
            Effective repetition limit (minimum 1)
        """
        if self.max_ratio_tokens is not None:
            # Determine n-gram size from rule configuration
            # Use average if range is specified, otherwise use min_length
            if self.max_length is not None and self.max_length != self.min_length:
                n_gram_size = (self.min_length + self.max_length) // 2
            else:
                n_gram_size = self.min_length

            # Calculate dynamic multipliers
            length_mult, ngram_mult = calculate_dynamic_multipliers(total_tokens, n_gram_size)

            # Apply adjustments to base ratio
            adjusted_ratio = self.max_ratio_tokens * length_mult * ngram_mult

            # Calculate ratio-based limit
            ratio_limit = int(total_tokens * adjusted_ratio)

            # Apply absolute cap if specified
            if self.max_count_absolute is not None:
                ratio_limit = min(ratio_limit, self.max_count_absolute)

            # Apply minimum floor for linguistic necessity
            min_floor = get_minimum_floor(total_tokens, n_gram_size)
            return max(ratio_limit, min_floor)

        elif self.max_repetitions is not None:
            # Legacy: use absolute limit
            return self.max_repetitions
        else:
            # Fallback (should not happen if validation works)
            return 999


@dataclass
class PhraseFrequencySettings:
    """Execution settings for phrase repetition analysis."""

    enabled: bool = False
    language: Optional[str] = None
    filter_stop_words: bool = True
    min_n: int = 2
    max_n: int = 6
    min_count: int = 2
    mp_threshold_tokens: int = 50_000
    workers: int = 0
    summary_top_k: int = 25
    diagnostics_mode: str = "off"
    diag_len_bins: str = ""
    diag_max_repeat_ratio: float = 0.0
    diag_min_distance_tokens: int = 0
    diag_cluster_gap_tokens: int = 80
    diag_cluster_min_count: int = 3
    diag_cluster_max_span_tokens: int = 250
    diag_top_k: int = 50
    diag_digest_k: int = 20
    rules: List[PhraseFrequencyRuleSpec] = field(default_factory=list)


@dataclass
class PhraseFrequencyIssue:
    """Detected repetition issue for QA consumption."""

    phrase: str
    n: int
    count: int
    limit: int
    severity: str
    rule_label: Optional[str] = None
    guidance: Optional[str] = None
    repeat_ratio_tokens: Optional[float] = None


@dataclass
class PhraseFrequencyResult:
    """Wrapper with analyzer raw output and filtered issues."""

    issues: List[PhraseFrequencyIssue]
    analyzer_output: Dict[str, Any]
    max_over_limit: int = 0


def _normalize_phrase(text: str) -> str:
    tokens, _ = tokenize_word_punct_with_spans(text, lowercase=True, remove_accents_flag=False)
    return untokenize(tokens)


def _phrase_is_stopword_only(phrase_text: str, stop_words: Set[str]) -> bool:
    tokens = [tok.strip().lower() for tok in phrase_text.split() if tok.strip()]
    if not tokens:
        return False
    return all(tok in stop_words for tok in tokens)


def _build_analysis_config(settings: PhraseFrequencySettings) -> AnalysisConfig:
    return AnalysisConfig(
        min_n=settings.min_n,
        max_n=settings.max_n,
        min_count=settings.min_count,
        tokenizer="word_punct",
        lowercase=True,
        strip_accents=False,
        respect_sentences=True,
        algo_mode="auto",
        mp_threshold_tokens=settings.mp_threshold_tokens,
        workers=settings.workers,
        core_policy="physical",
        punct_policy="drop-edge",
        summary_mode="counts",
        details_ratios=False,
        output_mode="full",
        details="all",
        details_top_k=0,
        positions_preview=0,
        enable_clusters=False,
        enable_windows=False,
        window_size_tokens=0,
        top_windows_k=0,
        summary_top_k=settings.summary_top_k,
        counts_only_limit_per_n=0,
        diagnostics=settings.diagnostics_mode,
        diag_len_bins=settings.diag_len_bins,
        diag_max_repeat_ratio=settings.diag_max_repeat_ratio,
        diag_min_distance_tokens=settings.diag_min_distance_tokens,
        diag_cluster_gap_tokens=settings.diag_cluster_gap_tokens,
        diag_cluster_min_count=settings.diag_cluster_min_count,
        diag_cluster_max_span_tokens=settings.diag_cluster_max_span_tokens,
        diag_top_k=settings.diag_top_k,
        diag_digest_k=settings.diag_digest_k,
    )


def _collect_phrase_lookup(analyzer_output: Dict[str, Any]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    phrases_section = analyzer_output.get("phrases", {})
    lookup: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for key, entries in phrases_section.items():
        try:
            n_val = int(key)
        except (TypeError, ValueError):
            continue
        bucket: Dict[str, Dict[str, Any]] = {}
        if isinstance(entries, list):
            for entry in entries:
                phrase_text = entry.get("text")
                if not phrase_text:
                    continue
                bucket[phrase_text] = entry
        lookup[n_val] = bucket

    return lookup


def analyze_phrase_frequency(text: str, settings: PhraseFrequencySettings) -> PhraseFrequencyResult:
    if not settings.enabled:
        return PhraseFrequencyResult(issues=[], analyzer_output={}, max_over_limit=0)

    text_to_analyze = text  # Content already preprocessed by caller
    analysis_cfg = _build_analysis_config(settings)
    analyzer_output = analyze_text(text_to_analyze, analysis_cfg)
    lookup = _collect_phrase_lookup(analyzer_output)

    # Get total tokens for ratio-based limit calculation
    meta = analyzer_output.get("meta", {})
    total_tokens = meta.get("total_tokens", len(text.split()))  # Fallback to word count

    issues: List[PhraseFrequencyIssue] = []
    max_over_limit = 0

    stop_words = get_stopwords_for_language(settings.language, settings.filter_stop_words)
    stop_words_applied = bool(stop_words)
    canonical_language = resolve_language_hint(settings.language) or settings.language

    for rule in settings.rules:
        # Compute effective limit based on ratio or absolute
        effective_limit = rule.compute_effective_limit(total_tokens)

        normalized_phrase = _normalize_phrase(rule.phrase) if rule.phrase else None

        for n_value, phrases in lookup.items():
            if rule.min_length and n_value < rule.min_length:
                continue
            if rule.max_length is not None and n_value > rule.max_length:
                continue

            if normalized_phrase:
                entry = phrases.get(normalized_phrase)
                if not entry:
                    continue
                candidates = [(normalized_phrase, entry)]
            else:
                candidates = list(phrases.items())

            for phrase_text, entry in candidates:
                if (
                    not normalized_phrase
                    and stop_words_applied
                    and _phrase_is_stopword_only(phrase_text, stop_words)
                ):
                    continue
                count = entry.get("count", 0)
                if count <= effective_limit:
                    continue

                over_limit = count - effective_limit
                max_over_limit = max(max_over_limit, over_limit)

                issues.append(
                    PhraseFrequencyIssue(
                        phrase=phrase_text,
                        n=n_value,
                        count=count,
                        limit=effective_limit,
                        severity=rule.severity,
                        rule_label=rule.label,
                        guidance=rule.guidance,
                        repeat_ratio_tokens=entry.get("repeat_ratio_tokens"),
                    )
                )

            if normalized_phrase:
                break

    issues.sort(key=lambda item: (-item.count, item.phrase))

    if canonical_language:
        analyzer_output.setdefault("meta", {})["language_hint"] = canonical_language
    analyzer_output.setdefault("meta", {})["stop_words_filtered"] = stop_words_applied

    return PhraseFrequencyResult(issues=issues, analyzer_output=analyzer_output, max_over_limit=max_over_limit)
