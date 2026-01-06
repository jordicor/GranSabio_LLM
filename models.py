"""
Data Models for Gran Sabio LLM Engine
======================================

Pydantic models for request/response handling and internal data structures.
"""

import os
from pydantic import BaseModel, Field, field_validator, AliasChoices, model_validator
from typing import List, Dict, Any, Optional, Literal, Tuple, Union
from enum import Enum
from datetime import datetime

from config import get_default_models, config
from edit_models import TextEditRange


class GenerationStatus(str, Enum):
    """Status enumeration for content generation"""
    INITIALIZING = "initializing"
    GENERATING = "generating"
    QA_EVALUATION = "qa_evaluation"
    GRAN_SABIO_REVIEW = "gran_sabio_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Valid content types (public documentation)
# Note: "other" is also supported but intentionally not listed for internal use only
VALID_CONTENT_TYPES = frozenset([
    "biography",
    "script",
    "novel",
    "article",
    "essay",
    "technical",
    "creative",
    "json",
    # Opinion/evaluation types (for smart_editing_mode='never')
    "opinion",
    "analysis",
    "review",
    "selection",
    "comparison",
    "evaluation",
    "vote",
    "preference",
    # Additional factual types
    "report",
    "story"
])


def _default_gran_sabio_model() -> str:
    """Return configured default model for Gran Sabio from configuration."""
    defaults = get_default_models()
    default_model = defaults.get("gran_sabio")
    if not default_model:
        raise RuntimeError("Gran Sabio default model is not configured in model_specs.json under 'default_models.gran_sabio'.")
    return default_model


class QALayer(BaseModel):
    """Configuration for a QA evaluation layer"""
    name: str = Field(..., description="Name of the QA layer")
    description: str = Field(..., description="What this layer evaluates")
    criteria: str = Field(..., description="Specific criteria for evaluation")
    min_score: float = Field(default=7.0, ge=0.0, le=10.0, description="Minimum score for this layer")
    is_mandatory: bool = Field(default=False, description="Whether this layer must pass after max iterations or trigger rejection")
    deal_breaker_criteria: Optional[str] = Field(default=None, description="Specific deal-breaker facts to detect (e.g., 'invents facts', 'uses offensive language')")
    concise_on_pass: bool = Field(default=True, description="If True, provide concise 'Passed' feedback when score >= min_score (saves tokens). If False, always provide detailed feedback.")
    order: int = Field(default=1, description="Evaluation order for this layer")

    # Vision support for QA evaluation
    include_input_images: bool = Field(
        default=False,
        description="When true and images are available, include input images in QA evaluation context. Useful for layers that validate image descriptions or visual accuracy."
    )

    # Deprecated field for backward compatibility
    is_deal_breaker: bool = Field(default=False, description="DEPRECATED: Use deal_breaker_criteria instead")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Accuracy",
                "description": "Factual accuracy and truth verification",
                "criteria": "Check for factual errors, invented information, or contradictions with known facts",
                "min_score": 8.0,
                "is_mandatory": True,
                "deal_breaker_criteria": "invents facts or presents false information as true",
                "order": 1
            }
        }


class QAModelConfig(BaseModel):
    """Configuration for a specific QA evaluation model"""
    model: str = Field(..., description="Model identifier (e.g., 'gpt-5-mini', 'claude-opus-4')")
    max_tokens: int = Field(
        default=8000,
        gt=0,
        description="Maximum tokens for QA evaluation"
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort for GPT-5/O1/O3 models (none, low, medium, high)"
    )
    thinking_budget_tokens: Optional[int] = Field(
        default=None,
        ge=1024,
        description="Thinking budget for Claude models (min 1024)"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Custom temperature for this QA model (default: 0.3 if not specified)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-5-mini",
                "max_tokens": 10000,
                "reasoning_effort": "medium",
                "temperature": 0.3
            }
        }


class WordCountEnforcement(BaseModel):
    """Configuration for word count enforcement in content generation."""

    enabled: bool = Field(default=False, description="Enable word count enforcement")
    flexibility_percent: float = Field(default=15.0, ge=0.0, le=100.0, description="Allowed flexibility percentage (0-100)")
    direction: Literal["both", "more", "less"] = Field(default="both", description="Direction of flexibility: both (±), more (+), or less (-)")
    severity: Literal["important", "deal_breaker"] = Field(default="important", description="Severity level when word count is violated")
    target_field: Optional[str] = Field(default=None, description="Specific JSON field to count words in (e.g., 'generated_text'). If None, counts all content.")

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "flexibility_percent": 15.0,
                "direction": "both",
                "severity": "deal_breaker",
                "target_field": "generated_text"
            }
        }


class PhraseFrequencyRule(BaseModel):
    """Rule definition for phrase repetition checks."""

    name: str = Field(..., min_length=1, description="Identifier for the rule")
    min_length: int = Field(default=2, ge=1, description="Minimum n-gram length to inspect")
    max_length: Optional[int] = Field(default=None, ge=1, description="Maximum n-gram length to inspect")
    max_repetitions: Optional[int] = Field(default=None, ge=1, description="Maximum allowed occurrences before flagging (deprecated: use max_ratio_tokens)")
    max_ratio_tokens: Optional[float] = Field(default=None, ge=0.0, le=0.5, description="Maximum ratio of text (0.0-0.5 = 0%-50%) - scales with text length")
    max_count_absolute: Optional[int] = Field(default=None, ge=1, description="Absolute maximum count as safety limit (used with max_ratio_tokens)")
    phrase: Optional[str] = Field(default=None, description="Exact phrase to monitor; leave empty for generic rule")
    severity: Literal["warn", "deal_breaker"] = Field(default="warn", description="Severity when the rule is violated")
    guidance: Optional[str] = Field(default=None, description="Optional guidance for generator iterations")

    @field_validator("max_length")
    @classmethod
    def validate_range(cls, v: Optional[int], info):
        min_length = info.data.get("min_length", 1)
        if v is not None and v < min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        return v

    @field_validator("max_ratio_tokens")
    @classmethod
    def validate_ratio(cls, v: Optional[float], info):
        # Either max_ratio_tokens or max_repetitions must be provided
        max_repetitions = info.data.get("max_repetitions")
        if v is None and max_repetitions is None:
            raise ValueError("Either max_ratio_tokens or max_repetitions must be specified")
        return v


class PhraseFrequencyConfig(BaseModel):
    """Configuration envelope for phrase repetition QA."""

    enabled: bool = Field(default=False, description="Enable phrase repetition QA layer")
    language: Optional[str] = Field(default=None, description="Language hint for repetition analysis (e.g., 'es', 'en')")
    filter_stop_words: bool = Field(default=True, description="Ignore stop-word-only phrases when language stop words are available")
    min_score: float = Field(default=8.0, ge=0.0, le=10.0, description="Minimum QA score threshold for this layer")
    min_n: Optional[int] = Field(default=None, ge=1, description="Override minimum n-gram length")
    max_n: Optional[int] = Field(default=None, ge=1, description="Override maximum n-gram length")
    min_count: int = Field(default=2, ge=1, description="Minimum occurrences to include in analysis")
    mp_threshold_tokens: int = Field(default=50000, ge=1, description="Token threshold for switching to multiprocess counting")
    workers: int = Field(default=0, ge=0, description="Explicit worker count for multiprocess mode (0 = auto)")
    summary_top_k: int = Field(default=25, ge=1, description="Top phrases per n to include in summaries")
    diagnostics_mode: Literal["off", "basic", "full"] = Field(default="off", description="Diagnostics mode for repetition analyzer")
    diag_len_bins: Optional[str] = Field(default=None, description="Length-bin rule string, e.g., '3-5:5,6-10:3'")
    diag_max_repeat_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Max repeat ratio for diagnostics")
    diag_min_distance_tokens: Optional[int] = Field(default=None, ge=0, description="Minimum token distance for diagnostics")
    diag_cluster_gap_tokens: Optional[int] = Field(default=None, ge=0, description="Cluster gap tokens for diagnostics")
    diag_cluster_min_count: Optional[int] = Field(default=None, ge=1, description="Minimum cluster count for diagnostics")
    diag_cluster_max_span_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum cluster span for diagnostics")
    diag_top_k: Optional[int] = Field(default=None, ge=0, description="Maximum diagnostics violations to include")
    diag_digest_k: Optional[int] = Field(default=None, ge=0, description="Maximum digest entries to include")
    target_field: Optional[str] = Field(default=None, description="Specific JSON field to analyze (e.g., 'generated_text', 'data.content'). If None, analyzes full content.")
    rules: List[PhraseFrequencyRule] = Field(default_factory=list, description="Phrase repetition rules to evaluate")

    @field_validator("max_n")
    @classmethod
    def validate_ngram_range(cls, v: Optional[int], info):
        min_n = info.data.get("min_n")
        if v is not None and min_n is not None and v < min_n:
            raise ValueError("max_n must be greater than or equal to min_n")
        return v

    @field_validator("rules")
    @classmethod
    def ensure_rules_when_enabled(cls, v: List[PhraseFrequencyRule], info):
        enabled = info.data.get("enabled", False)
        if enabled and not v:
            raise ValueError("At least one phrase frequency rule is required when enabled")
        return v

    def derive_min_n(self) -> int:
        candidate = [rule.min_length for rule in self.rules if rule.min_length]
        for rule in self.rules:
            if rule.phrase:
                phrase_tokens = [tok for tok in rule.phrase.strip().split() if tok]
                if phrase_tokens:
                    candidate.append(len(phrase_tokens))
        min_rule = min(candidate) if candidate else 2
        return max(1, self.min_n or min_rule)

    def derive_max_n(self) -> int:
        candidate = []
        for rule in self.rules:
            if rule.max_length is not None:
                candidate.append(rule.max_length)
            else:
                if rule.phrase:
                    phrase_tokens = [tok for tok in rule.phrase.strip().split() if tok]
                    if phrase_tokens:
                        candidate.append(len(phrase_tokens))
                        continue
                candidate.append(rule.min_length)
        max_rule = max(candidate) if candidate else max(self.min_n or 2, 6)
        configured = self.max_n or max_rule
        return max(configured, self.derive_min_n())

    def to_settings(self) -> "PhraseFrequencySettings":
        from tools.phrase_frequency_utils import (
            PhraseFrequencyRuleSpec,
            PhraseFrequencySettings,
        )

        rules = [
            PhraseFrequencyRuleSpec(
                min_length=rule.min_length,
                max_length=rule.max_length,
                max_repetitions=rule.max_repetitions,
                max_ratio_tokens=rule.max_ratio_tokens,
                max_count_absolute=rule.max_count_absolute,
                severity=rule.severity,
                phrase=rule.phrase,
                label=rule.name,
                guidance=rule.guidance,
            )
            for rule in self.rules
        ]

        settings = PhraseFrequencySettings(
            enabled=self.enabled,
            language=self.language,
            filter_stop_words=self.filter_stop_words,
            min_n=self.derive_min_n(),
            max_n=self.derive_max_n(),
            min_count=self.min_count,
            mp_threshold_tokens=self.mp_threshold_tokens,
            workers=self.workers,
            summary_top_k=self.summary_top_k,
            diagnostics_mode=self.diagnostics_mode,
            diag_len_bins=self.diag_len_bins or "",
            diag_max_repeat_ratio=(self.diag_max_repeat_ratio if self.diag_max_repeat_ratio is not None else 0.0),
            diag_min_distance_tokens=(self.diag_min_distance_tokens if self.diag_min_distance_tokens is not None else 0),
            diag_cluster_gap_tokens=(self.diag_cluster_gap_tokens if self.diag_cluster_gap_tokens is not None else 80),
            diag_cluster_min_count=(self.diag_cluster_min_count if self.diag_cluster_min_count is not None else 3),
            diag_cluster_max_span_tokens=(self.diag_cluster_max_span_tokens if self.diag_cluster_max_span_tokens is not None else 250),
            diag_top_k=(self.diag_top_k if self.diag_top_k is not None else 50),
            diag_digest_k=(self.diag_digest_k if self.diag_digest_k is not None else 20),
            target_field=self.target_field,
            rules=rules,
        )
        return settings

    def build_layer(self, order: int = 1) -> QALayer:
        if not self.enabled or not self.rules:
            raise ValueError("Cannot build phrase frequency layer without enabled rules")

        description = "Evalúa repeticiones de frases configuradas"
        criteria_lines = [
            "Analiza el texto con tokenizador word_punct, respetando los límites de oración.",
            "Revisa estas reglas de repetición:",
        ]

        deal_breaker_rules: List[str] = []
        for rule in self.rules:
            length_desc = (
                f"{rule.min_length}-{rule.max_length} palabras"
                if rule.max_length is not None and rule.max_length != rule.min_length
                else (f"{rule.min_length} palabras" if rule.max_length == rule.min_length else f"≥{rule.min_length} palabras")
            )
            phrase_desc = f"frase exacta '{rule.phrase}'" if rule.phrase else f"frases de {length_desc}"
            severity_desc = "deal-breaker" if rule.severity == "deal_breaker" else "aviso"
            criteria_lines.append(
                f"- {rule.name}: {phrase_desc}, máximo {rule.max_repetitions} repeticiones ({severity_desc})."
            )
            if rule.severity == "deal_breaker":
                deal_breaker_rules.append(rule.name)

        deal_breaker_text = None
        if deal_breaker_rules:
            deal_breaker_text = (
                "Repetir cualquiera de las reglas marcadas como deal-breaker por encima del límite configurado"
            )

        return QALayer(
            name="Phrase Frequency Guard",
            description=description,
            criteria="\n".join(criteria_lines),
            min_score=self.min_score,
            is_mandatory=True,
            deal_breaker_criteria=deal_breaker_text,
            concise_on_pass=True,
            order=order,
        )


class LexicalDiversityThresholdsConfig(BaseModel):
    """Optional overrides for lexical diversity thresholds."""

    herdan_green_min: Optional[float] = Field(default=None, description="Override Herdan's C GREEN minimum")
    herdan_amber_min: Optional[float] = Field(default=None, description="Override Herdan's C AMBER minimum")
    yulek_green_max: Optional[float] = Field(default=None, description="Override Yule's K GREEN maximum")
    yulek_amber_max: Optional[float] = Field(default=None, description="Override Yule's K AMBER maximum")
    mtld_green_min: Optional[float] = Field(default=None, description="Override MTLD GREEN minimum")
    mtld_amber_min: Optional[float] = Field(default=None, description="Override MTLD AMBER minimum")
    hdd_green_min: Optional[float] = Field(default=None, description="Override HD-D GREEN minimum")
    hdd_amber_min: Optional[float] = Field(default=None, description="Override HD-D AMBER minimum")
    brunet_green_max: Optional[float] = Field(default=None, description="Override Brunet's W GREEN maximum")
    brunet_amber_max: Optional[float] = Field(default=None, description="Override Brunet's W AMBER maximum")


class LexicalDiversityDecisionConfig(BaseModel):
    """Decision policy for lexical diversity QA layer."""

    require_majority: int = Field(default=2, ge=1, description="Number of GREEN/RED metrics required for decisive label")
    deal_breaker_on_red: bool = Field(default=True, description="Mark layer as deal-breaker whenever decision is RED")
    deal_breaker_on_amber: bool = Field(default=False, description="Mark layer as deal-breaker when decision is AMBER")
    red_metrics_threshold: Optional[int] = Field(default=None, ge=1, description="Deal-breaker when this many metrics are RED")
    amber_metrics_threshold: Optional[int] = Field(default=None, ge=1, description="Deal-breaker when this many metrics are AMBER")
    custom_metric_thresholds: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-metric threshold overrides. Keys: metric name, values: threshold dictionary.",
    )


class LexicalDiversityWindowConfig(BaseModel):
    """Window analysis settings for lexical diversity layer."""

    analyze_windows: bool = Field(default=False, description="Force window/paragraph analysis on every run")
    window_mode: Literal["tokens", "paragraphs"] = Field(default="tokens", description="Window analysis mode")
    window_size: int = Field(default=200, ge=1, description="Token count per window when window_mode=tokens")
    window_step: int = Field(default=100, ge=1, description="Step size between windows when window_mode=tokens")
    window_top_k: int = Field(default=10, ge=0, description="Top words per window")
    include_window_metrics: bool = Field(default=False, description="Include metrics for each window")
    window_include_positions: bool = Field(default=False, description="Include token indices for top window words")
    window_preview_chars: int = Field(default=160, ge=1, description="Preview characters per window in metadata")
    auto_window_on_large_text: bool = Field(default=True, description="Enable window analysis automatically on large texts")
    auto_window_token_threshold: int = Field(default=1200, ge=1, description="Token threshold for automatic window analysis")
    auto_window_on_decision: List[str] = Field(
        default_factory=lambda: ["RED"],
        description="Decision labels that should trigger window analysis automatically",
    )

    def build_policy(self) -> "LexicalDiversityWindowPolicy":
        from tools.lexical_diversity_utils import LexicalDiversityWindowPolicy

        return LexicalDiversityWindowPolicy(
            analyze_windows=self.analyze_windows,
            window_mode=self.window_mode,
            window_size=self.window_size,
            window_step=self.window_step,
            window_top_k=self.window_top_k,
            include_window_metrics=self.include_window_metrics,
            window_include_positions=self.window_include_positions,
            window_preview_chars=self.window_preview_chars,
            auto_window_on_large_text=self.auto_window_on_large_text,
            auto_window_token_threshold=self.auto_window_token_threshold,
            auto_window_on_decision=tuple(label.upper() for label in self.auto_window_on_decision),
        )


class LexicalDiversityScoringConfig(BaseModel):
    """Scoring policy for lexical diversity evaluations."""

    green_score: float = Field(default=9.0, ge=8.0, le=10.0, description="Score applied when decision=GREEN")
    amber_score: float = Field(default=7.0, ge=0.0, le=9.5, description="Score applied when decision=AMBER")
    red_score: float = Field(default=3.0, ge=0.0, le=8.0, description="Score applied when decision=RED")
    green_floor: float = Field(default=8.0, ge=0.0, le=10.0, description="Lower bound enforced on GREEN scores")

    def build_policy(self) -> "LexicalDiversityScorePolicy":
        from tools.lexical_diversity_utils import LexicalDiversityScorePolicy

        return LexicalDiversityScorePolicy(
            green_score=self.green_score,
            amber_score=self.amber_score,
            red_score=self.red_score,
            green_floor=self.green_floor,
        )


class LexicalDiversityConfig(BaseModel):
    """Configuration envelope for lexical diversity QA."""

    enabled: bool = Field(default=False, description="Enable lexical diversity QA layer")
    metrics: str = Field(
        default="auto",
        description='Metrics to compute: "auto", "all", or comma-separated list (e.g., "mtld,hdd,yulek,c")',
    )
    include_ttr: bool = Field(default=False, description="Include type-token ratio when metrics=auto")
    distinct_max_n: int = Field(default=0, ge=0, description="Compute distinct-n metrics up to this n (0 disables)")
    mtld_threshold: float = Field(default=0.72, gt=0.0, description="MTLD factor threshold")
    mtld_min_factor_len: int = Field(default=10, ge=1, description="Minimum factor length for MTLD calculation")
    hdd_sample_size: int = Field(default=42, ge=1, description="Sample size for HD-D metric")
    brunet_alpha: float = Field(default=0.165, gt=0.0, description="Brunet's W alpha parameter")
    tokenizer: Literal["word_punct", "alnum"] = Field(default="word_punct", description="Tokenizer mode")
    lowercase: bool = Field(default=True, description="Lowercase tokens before analysis")
    strip_accents: bool = Field(default=False, description="Strip accents before analysis")
    language: Optional[str] = Field(default=None, description="Language hint for stop-word filtering (e.g., 'es', 'en')")
    filter_stop_words: bool = Field(default=True, description="Filter language stop words from top-word summaries when available")
    top_words_k: int = Field(default=50, ge=0, description="Top repeated words to expose in metadata")
    include_positions: bool = Field(default=False, description="Include token positions for top words")
    target_field: Optional[str] = Field(
        default=None,
        description="Specific JSON field to analyze for lexical diversity (e.g., 'generated_text', 'data.content'). If None, analyzes all content."
    )
    thresholds: Optional[LexicalDiversityThresholdsConfig] = Field(
        default=None, description="Threshold overrides for grading"
    )
    decision: LexicalDiversityDecisionConfig = Field(
        default_factory=LexicalDiversityDecisionConfig, description="Decision policy tuning"
    )
    windows: LexicalDiversityWindowConfig = Field(
        default_factory=LexicalDiversityWindowConfig, description="Window analysis configuration"
    )
    scoring: LexicalDiversityScoringConfig = Field(
        default_factory=LexicalDiversityScoringConfig, description="Score mapping configuration"
    )

    def to_settings(self) -> "LexicalDiversitySettings":
        from tools.lexical_diversity_utils import (
            LexicalDiversityDecisionPolicy,
            LexicalDiversityScorePolicy,
            LexicalDiversitySettings,
        )

        decision_policy = LexicalDiversityDecisionPolicy(
            require_majority=self.decision.require_majority,
            deal_breaker_on_red=self.decision.deal_breaker_on_red,
            deal_breaker_on_amber=self.decision.deal_breaker_on_amber,
            red_metrics_threshold=self.decision.red_metrics_threshold,
            amber_metrics_threshold=self.decision.amber_metrics_threshold,
            custom_metric_thresholds={
                key.lower(): value for key, value in self.decision.custom_metric_thresholds.items()
            },
        )

        thresholds_overrides: Dict[str, float] = {}
        if self.thresholds:
            for field_name, value in self.thresholds.model_dump(exclude_none=True).items():
                thresholds_overrides[field_name] = value

        return LexicalDiversitySettings(
            enabled=self.enabled,
            language=self.language,
            filter_stop_words=self.filter_stop_words,
            metrics=self.metrics,
            include_ttr=self.include_ttr,
            distinct_max_n=self.distinct_max_n,
            mtld_threshold=self.mtld_threshold,
            mtld_min_factor_len=self.mtld_min_factor_len,
            hdd_sample_size=self.hdd_sample_size,
            brunet_alpha=self.brunet_alpha,
            tokenizer=self.tokenizer,
            lowercase=self.lowercase,
            strip_accents=self.strip_accents,
            thresholds_overrides=thresholds_overrides,
            top_words_k=self.top_words_k,
            include_positions=self.include_positions,
            target_field=self.target_field,
            decision_policy=decision_policy,
            score_policy=self.scoring.build_policy(),
            window_policy=self.windows.build_policy(),
        )

    def build_layer(self, order: int = 1) -> QALayer:
        if not self.enabled:
            raise ValueError("Cannot build lexical diversity layer when disabled")

        description = "Evaluates lexical variety across the draft"
        criteria_lines = [
            "Measure vocabulary richness with MTLD, HD-D, Herdan's C, Yule's K, and optional distinct-n metrics.",
            "Flag drafts with overly repetitive wording or narrow vocabulary.",
            "Promote varied phrasing before repetition analysis triggers additional guardrails.",
        ]

        deal_breaker_text = (
            "Lexical diversity scores below configured thresholds (decision RED) or repeated AMBER/RED metrics."
            if self.decision.deal_breaker_on_red or self.decision.red_metrics_threshold
            else None
        )

        return QALayer(
            name="Lexical Diversity Guard",
            description=description,
            criteria="\n".join(criteria_lines),
            min_score=8.0,
            is_mandatory=False,
            deal_breaker_criteria=deal_breaker_text,
            concise_on_pass=True,
            order=order,
        )

class ContextDocumentRef(BaseModel):
    """Reference to a previously uploaded attachment for context injection."""

    upload_id: str = Field(..., min_length=8, description="Identifier returned by the attachment router")
    username: str = Field(..., min_length=1, description="User identifier that owns the attachment")
    intended_usage: Optional[str] = Field(default="context", description="Usage hint stored alongside the attachment")


class ImageRef(BaseModel):
    """Reference to an image attachment for vision-enabled generation."""

    upload_id: str = Field(..., min_length=8, description="Attachment upload_id containing the image")
    username: str = Field(..., min_length=1, description="Owner of the attachment")
    detail: Optional[str] = Field(
        default=None,
        description="Detail level for OpenAI: 'low', 'high', 'auto'. None = provider default"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "upload_id": "abc123def456",
                "username": "user1",
                "detail": "auto"
            }
        }


class ImageData(BaseModel):
    """Resolved image data ready for API calls."""

    base64_data: str = Field(..., description="Base64-encoded image content")
    mime_type: str = Field(..., description="Image MIME type (e.g., image/jpeg, image/png)")
    original_filename: str = Field(..., description="Original filename of the uploaded image")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    width: Optional[int] = Field(default=None, ge=1, description="Image width in pixels")
    height: Optional[int] = Field(default=None, ge=1, description="Image height in pixels")
    estimated_tokens: Optional[int] = Field(default=None, ge=0, description="Estimated token cost for this image")
    detail: Optional[str] = Field(default=None, description="Detail level applied for OpenAI")


class ProjectInitRequest(BaseModel):
    """Optional payload used when allocating or reserving a project identifier."""

    project_id: Optional[str] = Field(
        default=None,
        description="Client-supplied project identifier to validate and reserve; leave empty to let the API generate one.",
    )

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("project_id cannot be blank or whitespace only")
        if len(normalized) > 128:
            raise ValueError("project_id must be 128 characters or fewer")
        return normalized


class ContentRequest(BaseModel):
    """Request model for content generation"""
    prompt: str = Field(..., min_length=10, description="The content generation prompt")
    request_name: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional descriptive name for this request (e.g., 'LinkedIn Article', 'School Essay')"
    )
    content_type: str = Field(default="other", description="Type of content to generate (acts as a preset for system prompts)")
    json_output: bool = Field(
        default=False,
        description="When true, enforce JSON validation/payload handling while keeping the declared content_type.",
    )
    username: Optional[str] = Field(default=None, description="User identifier required when referencing attachments")
    project_id: Optional[str] = Field(
        default=None,
        description="Existing project identifier that should group this request with related sessions",
    )
    request_project_id: bool = Field(
        default=False,
        validation_alias=AliasChoices("request_project_id", "get_project_id"),
        description="When true, allocate and return a project_id if one is not provided.",
    )

    # Context attachments configuration
    context_documents: Optional[List[ContextDocumentRef]] = Field(
        default=None,
        description="List of attachments to expose as contextual documents",
    )

    # Image inputs for vision-enabled generation
    images: Optional[List[ImageRef]] = Field(
        default=None,
        description="List of image references for vision-enabled models (GPT-4o, Claude, Gemini)"
    )
    image_detail: Optional[str] = Field(
        default=None,
        description="Default detail level for all images: 'low', 'high', 'auto' (OpenAI-specific)"
    )

    # Generator configuration
    generator_model: str = Field(
        default="gpt-4o",
        validation_alias="model",  # Accept both 'model' and 'generator_model'
        description="AI model for content generation"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=4000, gt=0, description="Maximum tokens for generation (ignored if max_tokens_percentage is specified)")
    max_tokens_percentage: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=100.0,
        description="Use X% of model's maximum available tokens (1-100). Takes precedence over max_tokens"
    )
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt to override default editorial prompt")
    min_words: Optional[int] = Field(default=None, gt=0, description="Minimum words for generation")
    max_words: Optional[int] = Field(default=None, gt=0, description="Maximum words for generation (takes precedence over max_tokens)")
    language: Optional[str] = Field(
        default=None,
        description="Language hint for generation and QA modules (ISO 639-1 or locale code, e.g., 'es', 'en-US')",
    )
    
    # Reasoning/Thinking tokens configuration
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for GPT-5 models (none, low, medium, high)"
    )
    thinking_budget_tokens: Optional[int] = Field(
        default=None, 
        ge=1024, 
        description="Budget tokens for thinking/reasoning (minimum 1024, for Claude 3.7 and GPT-5 reasoning models)"
    )

    # Source text for QA validation (optional)
    source_text: Optional[str] = Field(
        default=None,
        description="Original source text for QA validation (e.g., interview text for biographical analysis)"
    )

    # Cumulative context for repetition analysis (optional)
    cumulative_text: Optional[str] = Field(
        default=None,
        description="Previously generated chapters for cumulative repetition analysis"
    )
    cumulative_word_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total word count of cumulative_text for ratio-based repetition analysis"
    )

    # QA configuration
    qa_models: Union[List[str], List[QAModelConfig]] = Field(
        default=["gpt-5-mini", "claude-sonnet-4-20250514", "gemini-2.5-flash"],
        description="AI models for QA evaluation (strings for simple config, QAModelConfig objects for advanced)"
    )
    qa_global_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Global configuration applied to all QA models (max_tokens, reasoning_effort, thinking_budget_tokens, temperature)"
    )
    qa_models_config: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Per-model QA configuration by model name. Example: {'gpt-5-mini': {'reasoning_effort': 'high', 'max_tokens': 12000}}"
    )
    qa_layers: List[QALayer] = Field(default=[], description="QA evaluation layers - use empty list [] to bypass QA evaluation")
    qa_with_vision: bool = Field(
        default=False,
        description="Enable vision support in QA evaluation. When true, input images are passed to QA layers that have include_input_images=True. Only works when images are provided in the request."
    )
    
    # Scoring configuration
    min_global_score: float = Field(default=8.0, ge=0.0, le=10.0, description="Minimum global average score")
    
    # Iteration configuration
    max_iterations: int = Field(default=5, gt=0, le=50, description="Maximum generation iterations")
    json_retry_without_iteration: bool = Field(
        default=False,
        description="When true, JSON validation errors do not consume iterations (retries are handled separately)"
    )
    max_consecutive_smart_edits: int = Field(
        default_factory=lambda: config.DEFAULT_MAX_CONSECUTIVE_SMART_EDITS,
        ge=1,
        le=50,
        description="Maximum consecutive smart edit passes before forcing full regeneration (configurable via DEFAULT_MAX_CONSECUTIVE_SMART_EDITS in .env)"
    )

    smart_editing_mode: Literal["auto", "always", "never"] = Field(
        default="auto",
        description=(
            "Control smart editing behavior: "
            "'auto' - System decides based on content type and QA feedback (default for factual content), "
            "'always' - Always request specific edit ranges (forces structured QA responses), "
            "'never' - Never request edit ranges (for opinion/voting/evaluation scenarios)"
        )
    )

    # Gran Sabio configuration
    gran_sabio_model: str = Field(default_factory=_default_gran_sabio_model, description="Model for Gran Sabio escalation")
    gran_sabio_fallback: bool = Field(default=False, description="Allow Gran Sabio to regenerate content when iterations are exhausted")

    gran_sabio_call_limit_per_iteration: int = Field(
        default=-1,
        ge=-1,
        description="Max Gran Sabio escalations per iteration. Use -1 for unlimited (testing/debug)."
    )

    gran_sabio_call_limit_per_session: int = Field(
        default=-1,
        ge=-1,
        description="Max Gran Sabio escalations per entire session. Use -1 for unlimited (testing/debug)."
    )

    # Word count enforcement configuration
    word_count_enforcement: Optional[WordCountEnforcement] = Field(
        default=None,
        description="Word count enforcement configuration"
    )

    # Phrase frequency QA configuration
    phrase_frequency: Optional[PhraseFrequencyConfig] = Field(
        default=None,
        description="Phrase frequency QA configuration"
    )
    lexical_diversity: Optional[LexicalDiversityConfig] = Field(
        default=None,
        description="Lexical diversity QA configuration"
    )
    
    # Cost tracking configuration
    show_query_costs: int = Field(
        default=0,
        ge=0,
        le=2,
        description=(
            "When >0, include token/cost usage summary in final outputs. "
            "1=aggregate summary, 2=full detailed breakdown."
        ),
    )

    # JSON Schema validation configuration
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema to validate generated JSON output against. Provides structural validation and clear error messages."
    )

    json_expectations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Path-based expectations for JSON validation. Each expectation is a dict with 'path', 'required', 'type', etc."
    )

    # Output configuration
    verbose: bool = Field(default=True, description="Enable verbose progress tracking")
    extra_verbose: bool = Field(
        default_factory=lambda: os.getenv("EXTRA_VERBOSE", "false").lower() == "true",
        description="Enable extra verbose logging including full AI prompts (can be set globally via EXTRA_VERBOSE environment variable)"
    )

    # Runtime context fields (internal use only, not part of public API)
    # These fields are set during processing to provide context to IAs
    _current_iteration: Optional[int] = None
    _total_iterations: Optional[int] = None
    _generation_mode: Optional[str] = None  # "normal" | "smart_edit"
    _smart_edit_metadata: Optional[Dict[str, Any]] = None

    @field_validator('max_tokens_percentage')
    @classmethod
    def validate_max_tokens_percentage(cls, v: Optional[float]) -> Optional[float]:
        """Validate max_tokens_percentage is within valid range"""
        if v is not None:
            if v < 1.0 or v > 100.0:
                raise ValueError(f"max_tokens_percentage must be between 1.0 and 100.0, got {v}")
        return v

    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content_type against allowed values (including hidden 'other' type)"""
        if v in VALID_CONTENT_TYPES or v == "other":
            return v

        # Provide helpful error message listing only public types
        valid_types_list = ", ".join(sorted(VALID_CONTENT_TYPES))
        raise ValueError(
            f"Invalid content_type '{v}'. Must be one of: {valid_types_list}"
        )

    @field_validator('gran_sabio_call_limit_per_iteration')
    @classmethod
    def validate_iteration_limit(cls, v: int) -> int:
        """Validate Gran Sabio escalation limit per iteration"""
        if v == 0:
            raise ValueError(
                "gran_sabio_call_limit_per_iteration cannot be 0. "
                "Use -1 for unlimited or positive integer."
            )
        if v < -1:
            raise ValueError(
                "gran_sabio_call_limit_per_iteration must be -1 (unlimited) or positive integer."
            )
        return v

    @field_validator('gran_sabio_call_limit_per_session')
    @classmethod
    def validate_session_limit(cls, v: int) -> int:
        """Validate Gran Sabio escalation limit per session"""
        if v == 0:
            raise ValueError(
                "gran_sabio_call_limit_per_session cannot be 0. "
                "Use -1 for unlimited or positive integer."
            )
        if v < -1:
            raise ValueError(
                "gran_sabio_call_limit_per_session must be -1 (unlimited) or positive integer."
            )
        return v

    @model_validator(mode="after")
    def _propagate_language(self):
        """Ensure nested QA configs inherit the request language when unspecified."""
        lang = (self.language or "").strip()
        if not lang:
            return self
        if self.lexical_diversity and not self.lexical_diversity.language:
            self.lexical_diversity.language = lang
        if self.phrase_frequency and not self.phrase_frequency.language:
            self.phrase_frequency.language = lang
        return self

    class Config:
        populate_by_name = True  # Allow both 'model' and 'generator_model'
        json_schema_extra = {
            "example": {
                "prompt": "Escribe una biografía de 2000 palabras sobre Albert Einstein, enfocándote en sus contribuciones científicas y su impacto en la física moderna.",
                "content_type": "biography",
                "json_output": False,
                "generator_model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4000,
                "min_words": 800,
                "max_words": 1200,
                "reasoning_effort": "medium",
                "thinking_budget_tokens": 2000,
                "qa_models": ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
                "qa_models_config": {
                    "gpt-4o": {
                        "max_tokens": 10000,
                        "reasoning_effort": "medium"
                    },
                    "claude-sonnet-4-20250514": {
                        "max_tokens": 12000,
                        "thinking_budget_tokens": 5000
                    }
                },
                "qa_layers": [
                    {
                        "name": "Accuracy",
                        "description": "Factual accuracy and truth verification",
                        "criteria": "Check for factual errors, invented information, or contradictions with known facts",
                        "min_score": 8.0,
                        "is_deal_breaker": True,
                        "order": 1
                    },
                    {
                        "name": "Literary Quality",
                        "description": "Writing style and literary expression",
                        "criteria": "Evaluate narrative flow, prose quality, and engaging writing style",
                        "min_score": 7.5,
                        "is_deal_breaker": False,
                        "order": 2
                    },
                    {
                        "name": "Structure",
                        "description": "Content organization and structure",
                        "criteria": "Assess logical organization, clear sections, and coherent progression",
                        "min_score": 7.0,
                        "is_deal_breaker": False,
                        "order": 3
                    }
                ],
                "min_global_score": 8.0,
                "max_iterations": 5,
                "gran_sabio_model": _default_gran_sabio_model(),
                "word_count_enforcement": {
                    "enabled": True,
                    "flexibility_percent": 15,
                    "direction": "both",
                    "severity": "deal_breaker"
                },
                "phrase_frequency": {
                    "enabled": True,
                    "rules": [
                        {
                            "name": "short_phrase_guard",
                            "min_length": 3,
                            "max_length": 6,
                            "max_repetitions": 2,
                            "severity": "warn"
                        },
                        {
                            "name": "specific_phrase",
                            "phrase": "entonces se fue a",
                            "min_length": 3,
                            "max_repetitions": 1,
                            "severity": "deal_breaker"
                        }
                    ]
                },
                "lexical_diversity": {
                    "enabled": True,
                    "metrics": "auto",
                    "top_words_k": 50,
                    "decision": {
                        "deal_breaker_on_red": True,
                        "deal_breaker_on_amber": False
                    }
                },
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "word_count": {"type": "integer"}
                    },
                    "required": ["title", "summary"],
                    "additionalProperties": False
                },
                "json_retry_without_iteration": False,
                "verbose": True
            }
        }


class QAEvaluation(BaseModel):
    """Result of a QA evaluation"""
    model: str = Field(..., description="AI model that performed the evaluation")
    layer: str = Field(..., description="QA layer name")
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Evaluation score (None when the response was invalid or missing)"
    )
    feedback: str = Field(..., description="Detailed feedback")
    deal_breaker: bool = Field(default=False, description="Whether a specific deal-breaker criteria was detected")
    deal_breaker_reason: Optional[str] = Field(None, description="Specific deal-breaker reason if detected")
    passes_score: bool = Field(..., description="Whether the score meets the minimum threshold")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for downstream processing")
    identified_issues: Optional[List["TextEditRange"]] = Field(
        default=None,
        description="Issue ranges identified during evaluation (if available)"
    )
    structured_response: Optional[Any] = Field(
        default=None,
        description="Structured payload (e.g., JSON issues block) returned by the evaluator when requested"
    )

    # Deprecated field for backward compatibility
    reason: Optional[str] = Field(None, description="DEPRECATED: Use deal_breaker_reason instead")

QAEvaluation.model_rebuild()


class ConsensusResult(BaseModel):
    """Result of consensus calculation"""

    model_config = {"populate_by_name": True}

    average_score: float = Field(..., description="Overall average score")
    layer_averages: Dict[str, float] = Field(..., description="Average score per layer")
    per_model_averages: Dict[str, float] = Field(
        ...,
        description="Average score per model",
        validation_alias=AliasChoices("model_averages", "per_model_averages"),
        serialization_alias="model_averages",
    )
    total_evaluations: int = Field(..., description="Total number of evaluations")
    approved: bool = Field(..., description="Whether content passes all criteria")
    deal_breakers: List[str] = Field(default=[], description="List of deal-breaker issues found")
    feedback_by_layer: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Aggregated feedback grouped by QA layer for iteration guidance"
    )
    actionable_feedback: List[str] = Field(
        default_factory=list,
        description="Flattened list of actionable feedback strings derived from QA evaluations"
    )


class GranSabioResult(BaseModel):
    """Result of Gran Sabio review"""
    approved: bool = Field(..., description="Whether Gran Sabio approves the content")
    final_content: str = Field(..., description="Final content (original or modified)")
    final_score: float = Field(..., description="Gran Sabio's final score")
    reason: str = Field(..., description="Explanation of the decision")
    modifications_made: bool = Field(default=False, description="Whether content was modified")
    error: Optional[str] = Field(default=None, description="Error message when Gran Sabio could not complete the review")
    timestamp: datetime = Field(default_factory=datetime.now)


class GranSabioEscalation(BaseModel):
    """Record of a Gran Sabio escalation event"""

    model_config = {"populate_by_name": True}

    escalation_id: str = Field(..., description="Unique ID for this escalation")
    session_id: str = Field(..., description="Parent session ID")
    iteration: int = Field(..., description="Iteration number when escalated")

    # Context
    layer_name: str = Field(..., description="QA layer where deal-breaker was detected")
    trigger_type: str = Field(
        ...,
        description="Type of escalation: 'minority_deal_breaker', '50_50_tie', 'max_iterations'"
    )

    # Deal-breaker details
    triggering_model: str = Field(
        ...,
        description="Model that detected the deal-breaker",
        validation_alias=AliasChoices("model_that_triggered", "triggering_model"),
        serialization_alias="model_that_triggered",
    )
    deal_breaker_reason: str = Field(..., description="Reason for the deal-breaker")
    total_models_evaluated: int = Field(..., description="Total models in this layer")
    deal_breaker_count: int = Field(..., description="Number of models that detected deal-breaker")

    # Gran Sabio decision
    gran_sabio_model_used: str = Field(..., description="Model used for Gran Sabio decision")
    decision: str = Field(default="pending", description="Decision: 'real', 'false_positive', or 'error'")
    reasoning: str = Field(default="", description="Gran Sabio's reasoning")

    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Result
    was_real_deal_breaker: Optional[bool] = Field(default=None, description="True if Gran Sabio confirmed it as real, False if false positive, None if unresolved due to error")


class ModelReliabilityStats(BaseModel):
    """Reliability statistics for a QA model"""

    model_config = {"populate_by_name": True}

    qa_model_name: str = Field(
        ...,
        description="Name of the QA model",
        validation_alias=AliasChoices("model_name", "qa_model_name"),
        serialization_alias="model_name",
    )

    # Counters
    total_deal_breakers_raised: int = Field(default=0, description="Total deal-breakers flagged by this model")
    confirmed_real: int = Field(default=0, description="Deal-breakers confirmed as real by Gran Sabio")
    confirmed_false_positive: int = Field(default=0, description="Deal-breakers rejected as false positives")

    # Rates
    false_positive_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Rate of false positives (0-1)")
    precision: float = Field(default=0.0, ge=0.0, le=1.0, description="Precision (true positives / total positives)")

    # Badge
    reliability_badge: str = Field(
        default="UNKNOWN",
        description="Reliability level: HIGH, MEDIUM, LOW, or UNKNOWN"
    )

    # Timestamps
    first_seen: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    def calculate_metrics(self):
        """Recalculate rates and badge"""
        total_confirmed = self.confirmed_real + self.confirmed_false_positive

        if total_confirmed > 0:
            self.false_positive_rate = self.confirmed_false_positive / total_confirmed
            self.precision = self.confirmed_real / total_confirmed if total_confirmed > 0 else 0.0
        else:
            self.false_positive_rate = 0.0
            self.precision = 0.0

        # Assign badge (need at least 5 evaluations for confidence)
        if total_confirmed >= 5:
            if self.precision >= 0.80:
                self.reliability_badge = "HIGH"
            elif self.precision >= 0.50:
                self.reliability_badge = "MEDIUM"
            else:
                self.reliability_badge = "LOW"
        else:
            self.reliability_badge = "UNKNOWN"

        self.last_updated = datetime.now()


class ProgressUpdate(BaseModel):
    """Progress update for streaming"""
    session_id: str = Field(..., description="Session identifier")
    project_id: Optional[str] = Field(
        default=None,
        description="Associated project identifier when available",
    )
    request_name: Optional[str] = Field(
        default=None,
        description="Descriptive name for this request when provided",
    )
    status: GenerationStatus = Field(..., description="Current status")
    current_iteration: int = Field(..., description="Current iteration number")
    max_iterations: int = Field(default=3, description="Maximum iteration number")
    verbose_log: List[str] = Field(default=[], description="Recent verbose log entries")
    generated_content: Optional[str] = Field(default=None, description="Current generated content (if available)")
    timestamp: datetime = Field(default_factory=datetime.now)


class PreflightIssue(BaseModel):
    """Issue detected during preflight validation"""
    code: str = Field(default="unspecified", description="Short identifier for the issue")
    severity: Literal["critical", "warning", "info"] = Field(default="critical", description="Severity level for the issue")
    message: str = Field(..., description="Human readable explanation of the issue")
    blockers: bool = Field(default=True, description="Whether this issue blocks the generation process")
    related_requirements: List[str] = Field(default_factory=list, description="QA layers or constraints linked to this issue")


class WordCountAnalysis(BaseModel):
    """Analysis of word count conflicts between QA layers"""
    conflicting_layers: List[str] = Field(default_factory=list, description="Names of QA layers that conflict with word count enforcement")
    recommended_removals: List[str] = Field(default_factory=list, description="Names of QA layers recommended for removal to prevent conflicts")
    analysis_reason: str = Field(..., description="Explanation of conflicts found by AI analysis")


class PreflightResult(BaseModel):
    """Structured output from the preflight validator"""
    decision: Literal["proceed", "reject", "manual_review"] = Field(..., description="Preflight decision outcome")
    user_feedback: str = Field(..., description="Message intended for the end user")
    summary: Optional[str] = Field(default=None, description="Short summary for logging and debugging")
    issues: List[PreflightIssue] = Field(default_factory=list, description="Issues detected during preflight validation")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score reported by the validator")

    # AI-powered word count analysis
    word_count_analysis: Optional[WordCountAnalysis] = Field(default=None, description="AI analysis of word count conflicts between QA layers")

    # Legacy word count optimization fields (deprecated - use word_count_analysis instead)
    enable_algorithmic_word_count: bool = Field(default=False, description="Enable algorithmic word count enforcement to avoid AI evaluation conflicts")
    duplicate_word_count_layers_to_remove: List[str] = Field(default_factory=list, description="Names of QA layers that duplicate word counting and should be removed to prevent conflicts")


class GenerationInitResponse(BaseModel):
    """Response returned when initializing a generation request"""
    status: str = Field(..., description="Initialization status (e.g., 'initialized', 'rejected')")
    session_id: Optional[str] = Field(default=None, description="Session identifier when generation starts")
    project_id: Optional[str] = Field(
        default=None,
        description="Project identifier associated with this session when provided or generated",
    )
    request_name: Optional[str] = Field(
        default=None,
        description="Descriptive name for this request when provided by the client",
    )
    preflight_feedback: Optional[PreflightResult] = Field(default=None, description="Preflight validation feedback when the request is rejected")
    recommended_timeout_seconds: Optional[int] = Field(
        default=None,
        description="Recommended wait time (in seconds) before assuming a timeout"
    )


class ProjectInitResponse(BaseModel):
    """Response returned when allocating or validating a project identifier."""

    project_id: str = Field(..., description="Project identifier allocated or validated for subsequent requests")


class ContentResponse(BaseModel):
    """Final response with generated content"""
    session_id: str = Field(..., description="Session identifier")
    project_id: Optional[str] = Field(default=None, description="Associated project identifier when available")
    request_name: Optional[str] = Field(default=None, description="Descriptive name for this request when provided")
    content: str = Field(..., description="Generated content")
    final_iteration: str = Field(..., description="Final iteration or 'Gran Sabio'")
    final_score: float = Field(..., description="Final approval score")
    qa_summary: Dict[str, Any] = Field(..., description="Summary of QA evaluations")
    generated_at: datetime = Field(..., description="Generation completion timestamp")
    costs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token usage and cost summary when requested by the client",
    )
    
    # Optional Gran Sabio fields
    gran_sabio_reason: Optional[str] = Field(None, description="Gran Sabio's reasoning if applicable")
    modifications_made: Optional[bool] = Field(None, description="Whether Gran Sabio modified content")


# Pre-defined QA layer templates
DEFAULT_QA_LAYERS = {
    "biography": [
        QALayer(
            name="Factual Accuracy",
            description="Source text fidelity and consistency",
            criteria="Verify that ALL information is traceable to the source text provided. Biographies can include fictional, fantasy, supernatural, or extraordinary elements - these are VALID if present in the source. Check for invented information NOT in the source text. Do NOT penalize for unusual content (fantasy characters, sci-fi scenarios, spiritual beliefs, extraordinary claims) if stated in the source. Use context to understand if content is metaphorical, humorous, spiritual belief, or literal within the biography's universe.",
            min_score=8.5,
            is_deal_breaker=True,
            order=1
        ),
        QALayer(
            name="Literary Quality",
            description="Writing style and narrative quality",
            criteria="Evaluate prose quality, narrative flow, engaging writing style, and appropriate tone for biographical content.",
            min_score=7.5,
            is_deal_breaker=False,
            order=2
        ),
        QALayer(
            name="Structure & Organization",
            description="Content structure and logical organization",
            criteria="Assess chronological organization, clear sections, coherent progression from birth to major achievements.",
            min_score=7.0,
            is_deal_breaker=False,
            order=3
        ),
        QALayer(
            name="Depth & Coverage",
            description="Comprehensiveness and depth of coverage",
            criteria="Evaluate whether all major life events, achievements, and impacts are adequately covered with sufficient detail.",
            min_score=7.0,
            is_deal_breaker=False,
            order=4
        )
    ],
    "script": [
        QALayer(
            name="Dialogue Quality",
            description="Natural and engaging dialogue",
            criteria="Evaluate dialogue naturalness, character voice distinctiveness, and conversational flow.",
            min_score=8.0,
            is_deal_breaker=False,
            order=1
        ),
        QALayer(
            name="Format Compliance",
            description="Proper script formatting",
            criteria="Check adherence to industry-standard script formatting, proper scene headings, character names, and action descriptions.",
            min_score=7.5,
            is_deal_breaker=True,
            order=2
        ),
        QALayer(
            name="Story Structure",
            description="Narrative structure and pacing",
            criteria="Assess three-act structure, character development, conflict progression, and pacing.",
            min_score=7.0,
            is_deal_breaker=False,
            order=3
        )
    ],
    "novel": [
        QALayer(
            name="Character Development",
            description="Character depth and consistency",
            criteria="Evaluate character development, consistency, and realistic character arcs throughout the narrative.",
            min_score=7.5,
            is_deal_breaker=False,
            order=1
        ),
        QALayer(
            name="Plot Coherence",
            description="Plot logic and consistency",
            criteria="Check for plot holes, logical inconsistencies, and ensure cause-and-effect relationships are clear.",
            min_score=8.0,
            is_deal_breaker=True,
            order=2
        ),
        QALayer(
            name="Prose Quality",
            description="Writing style and prose quality",
            criteria="Evaluate writing style, sentence variety, descriptive language, and overall prose quality.",
            min_score=7.0,
            is_deal_breaker=False,
            order=3
        ),
        QALayer(
            name="Engagement",
            description="Reader engagement and pacing",
            criteria="Assess whether the content maintains reader interest, appropriate pacing, and compelling narrative tension.",
            min_score=7.0,
            is_deal_breaker=False,
            order=4
        )
    ]
}


def normalize_qa_models_config(
    qa_models: Union[List[str], List[QAModelConfig]],
    qa_global_config: Optional[Dict[str, Any]] = None,
    qa_models_config: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[QAModelConfig]:
    """
    Normalize different QA configuration formats to a unified list of QAModelConfig

    Priority order:
    1. Explicit QAModelConfig in qa_models
    2. qa_models_config (per-model configuration)
    3. qa_global_config (global configuration)
    4. System defaults

    Args:
        qa_models: List of model names (strings) or QAModelConfig objects
        qa_global_config: Global configuration applied to all models
        qa_models_config: Per-model configuration by model name

    Returns:
        List of QAModelConfig objects with normalized configuration
    """
    result = []

    for model in qa_models:
        if isinstance(model, str):
            # Create QAModelConfig from string with defaults
            config = QAModelConfig(model=model)

            # Apply per-model configuration if exists (highest priority)
            if qa_models_config and model in qa_models_config:
                model_specific = qa_models_config[model]
                for key, value in model_specific.items():
                    if hasattr(config, key) and value is not None:
                        setattr(config, key, value)

            # Apply global configuration if no specific override (lower priority)
            elif qa_global_config:
                for key, value in qa_global_config.items():
                    if hasattr(config, key) and value is not None:
                        # Only set if still at default value
                        current = getattr(config, key)
                        if current is None or (key == 'max_tokens' and current == 8000):
                            setattr(config, key, value)
        else:
            # Already a QAModelConfig object
            config = model

        result.append(config)

    return result
