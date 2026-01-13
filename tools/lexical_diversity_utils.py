"""Utility helpers for lexical diversity QA analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tools.lexical_diversity import (
    LexDivConfig,
    Thresholds,
    analyze_lexical_diversity,
)

logger = logging.getLogger(__name__)


@dataclass
class LexicalDiversityScorePolicy:
    """Score mapping for lexical diversity outcomes."""

    green_score: float = 9.0
    amber_score: float = 7.0
    red_score: float = 3.0
    green_floor: float = 8.0

    def score_for_label(self, label: str) -> float:
        mapping = {
            "GREEN": max(self.green_score, self.green_floor),
            "AMBER": self.amber_score,
            "RED": self.red_score,
        }
        return mapping.get(label.upper(), self.amber_score)


@dataclass
class LexicalDiversityDecisionPolicy:
    """Decision policy for deal-breaker handling."""

    require_majority: int = 2
    deal_breaker_on_red: bool = True
    deal_breaker_on_amber: bool = False
    red_metrics_threshold: Optional[int] = None
    amber_metrics_threshold: Optional[int] = None
    custom_metric_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def should_trigger_deal_breaker(
        self, decision_label: str, grades: Dict[str, str]
    ) -> bool:
        label = decision_label.upper()
        if label == "RED" and self.deal_breaker_on_red:
            return True
        if label == "AMBER" and self.deal_breaker_on_amber:
            return True

        red_count = sum(1 for grade in grades.values() if grade == "RED")
        amber_count = sum(1 for grade in grades.values() if grade == "AMBER")

        if self.red_metrics_threshold and red_count >= self.red_metrics_threshold:
            return True
        if self.amber_metrics_threshold and amber_count >= self.amber_metrics_threshold:
            return True

        return False

    def apply_metric_overrides(self, metrics: Dict[str, float], grades: Dict[str, str]) -> Dict[str, str]:
        """Apply per-metric overrides based on configured thresholds."""
        if not self.custom_metric_thresholds:
            return grades

        adjusted = dict(grades)
        for metric, cfg in self.custom_metric_thresholds.items():
            metric_key = metric.strip().lower()
            if metric_key not in metrics:
                continue
            value = metrics[metric_key]
            green_min = cfg.get("green_min")
            amber_min = cfg.get("amber_min")
            green_max = cfg.get("green_max")
            amber_max = cfg.get("amber_max")

            if green_min is not None or amber_min is not None:
                # Higher-is-better interpretation
                if green_min is not None and value >= green_min:
                    adjusted[metric_key] = "GREEN"
                elif amber_min is not None and value >= amber_min:
                    adjusted[metric_key] = "AMBER"
                else:
                    adjusted[metric_key] = "RED"
            elif green_max is not None or amber_max is not None:
                # Lower-is-better interpretation
                if green_max is not None and value <= green_max:
                    adjusted[metric_key] = "GREEN"
                elif amber_max is not None and value <= amber_max:
                    adjusted[metric_key] = "AMBER"
                else:
                    adjusted[metric_key] = "RED"

        return adjusted


@dataclass
class LexicalDiversityWindowPolicy:
    """Window analysis configuration."""

    analyze_windows: bool = False
    window_mode: str = "tokens"
    window_size: int = 200
    window_step: int = 100
    window_top_k: int = 10
    include_window_metrics: bool = False
    window_include_positions: bool = False
    window_preview_chars: int = 160
    auto_window_on_large_text: bool = True
    auto_window_token_threshold: int = 1200
    auto_window_on_decision: Tuple[str, ...] = ("RED",)

    def should_analyze_windows(self, total_tokens: int, decision_label: str) -> bool:
        if self.analyze_windows:
            return True
        if (
            self.auto_window_on_large_text
            and total_tokens >= self.auto_window_token_threshold
        ):
            return True
        return decision_label.upper() in {
            label.upper() for label in self.auto_window_on_decision
        }


@dataclass
class LexicalDiversitySettings:
    """Execution settings for lexical diversity analysis."""

    enabled: bool = False
    language: Optional[str] = None
    filter_stop_words: bool = True
    metrics: str = "auto"
    include_ttr: bool = False
    distinct_max_n: int = 0
    mtld_threshold: float = 0.72
    mtld_min_factor_len: int = 10
    hdd_sample_size: int = 42
    brunet_alpha: float = 0.165
    tokenizer: str = "word_punct"
    lowercase: bool = True
    strip_accents: bool = False
    thresholds_overrides: Dict[str, float] = field(default_factory=dict)
    top_words_k: int = 50
    include_positions: bool = False
    decision_policy: LexicalDiversityDecisionPolicy = field(
        default_factory=LexicalDiversityDecisionPolicy
    )
    score_policy: LexicalDiversityScorePolicy = field(
        default_factory=LexicalDiversityScorePolicy
    )
    window_policy: LexicalDiversityWindowPolicy = field(
        default_factory=LexicalDiversityWindowPolicy
    )

    def build_lexdiv_config(self) -> LexDivConfig:
        thresholds = Thresholds()
        for key, value in self.thresholds_overrides.items():
            if value is None:
                continue
            if not hasattr(thresholds, key):
                continue
            setattr(thresholds, key, value)

        return LexDivConfig(
            tokenizer=self.tokenizer,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
             language=self.language,
             filter_stop_words=self.filter_stop_words,
            metrics=self.metrics,
            mtld_threshold=self.mtld_threshold,
            mtld_min_factor_len=self.mtld_min_factor_len,
            hdd_sample_size=self.hdd_sample_size,
            brunet_alpha=self.brunet_alpha,
            include_ttr=self.include_ttr,
            distinct_max_n=self.distinct_max_n,
            thresholds=thresholds,
            require_majority=self.decision_policy.require_majority,
            top_words_k=self.top_words_k,
            include_positions=self.include_positions,
            analyze_windows=False,  # decide later based on policy
            window_mode=self.window_policy.window_mode,
            window_size=self.window_policy.window_size,
            window_step=self.window_policy.window_step,
            window_top_k=self.window_policy.window_top_k,
            window_preview_chars=self.window_policy.window_preview_chars,
            include_window_metrics=self.window_policy.include_window_metrics,
            window_include_positions=self.window_policy.window_include_positions,
        )


@dataclass
class LexicalDiversityResult:
    """Structured result for lexical diversity QA consumption."""

    analysis: Dict[str, Any]
    adjusted_grades: Dict[str, str]
    deal_breaker: bool
    score: float
    decision_label: str


def analyze_text_lexical_diversity(
    text: str, settings: LexicalDiversitySettings
) -> LexicalDiversityResult:
    if not settings.enabled:
        return LexicalDiversityResult(
            analysis={},
            adjusted_grades={},
            deal_breaker=False,
            score=10.0,
            decision_label="GREEN",
        )

    lex_config = settings.build_lexdiv_config()
    text_to_analyze = text  # Content already preprocessed by caller
    base_result = analyze_lexical_diversity(text_to_analyze, lex_config)

    metrics = base_result.get("metrics", {})
    grades = base_result.get("grades", {})
    decision = base_result.get("decision", {})
    decision_label = decision.get("label", "AMBER")

    adjusted_grades = settings.decision_policy.apply_metric_overrides(
        {k.lower(): v for k, v in metrics.items()},
        {k.lower(): v for k, v in grades.items()},
    )
    decision_label = decision_label.upper()

    # Evaluate window analysis if required
    total_tokens = base_result.get("meta", {}).get("total_word_tokens", 0)
    if settings.window_policy.should_analyze_windows(total_tokens, decision_label):
        window_config = settings.build_lexdiv_config()
        window_config.analyze_windows = True
        base_result = analyze_lexical_diversity(text_to_analyze, window_config)
        metrics = base_result.get("metrics", metrics)
        grades = base_result.get("grades", grades)
        decision = base_result.get("decision", decision)
        decision_label = decision.get("label", decision_label).upper()
        adjusted_grades = settings.decision_policy.apply_metric_overrides(
            {k.lower(): v for k, v in metrics.items()},
            {k.lower(): v for k, v in grades.items()},
        )

    final_score = settings.score_policy.score_for_label(decision_label)
    deal_breaker = settings.decision_policy.should_trigger_deal_breaker(
        decision_label, adjusted_grades
    )

    return LexicalDiversityResult(
        analysis=base_result,
        adjusted_grades=adjusted_grades,
        deal_breaker=deal_breaker,
        score=final_score,
        decision_label=decision_label,
    )
