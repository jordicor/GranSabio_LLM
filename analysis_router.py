"""
FastAPI router for text analysis tools (Lexical Diversity & Repetition Analyzer)
================================================================================

Provides HTTP API endpoints for:
- Lexical Diversity Analysis: Vocabulary richness metrics (MTLD, HD-D, Yule's K, etc.)
- Repetition Analysis: N-gram repetition patterns with clustering and diagnostics
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Import analysis tools
from tools.lexical_diversity_utils import (
    LexicalDiversitySettings,
    LexicalDiversityDecisionPolicy,
    LexicalDiversityScorePolicy,
    LexicalDiversityWindowPolicy,
    analyze_text_lexical_diversity,
)
from tools.repetition_analyzer import AnalysisConfig, analyze_text
from word_count_utils import extract_target_field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ============================================================================
# LEXICAL DIVERSITY MODELS
# ============================================================================

class LexicalDiversityThresholdsRequest(BaseModel):
    """Threshold configuration for lexical diversity grades (GREEN/AMBER/RED)"""

    herdan_green_min: Optional[float] = Field(default=0.80, description="Herdan's C: min for GREEN")
    herdan_amber_min: Optional[float] = Field(default=0.65, description="Herdan's C: min for AMBER")
    yulek_green_max: Optional[float] = Field(default=100.0, description="Yule's K: max for GREEN")
    yulek_amber_max: Optional[float] = Field(default=150.0, description="Yule's K: max for AMBER")
    mtld_green_min: Optional[float] = Field(default=70.0, description="MTLD: min for GREEN")
    mtld_amber_min: Optional[float] = Field(default=50.0, description="MTLD: min for AMBER")
    hdd_green_min: Optional[float] = Field(default=0.70, description="HD-D: min for GREEN")
    hdd_amber_min: Optional[float] = Field(default=0.55, description="HD-D: min for AMBER")
    brunet_green_max: Optional[float] = Field(default=None, description="Brunet's W: max for GREEN (None=disabled)")
    brunet_amber_max: Optional[float] = Field(default=None, description="Brunet's W: max for AMBER (None=disabled)")


class LexicalDiversityRequest(BaseModel):
    """Request model for lexical diversity analysis endpoint"""

    # Input (required)
    text: str = Field(..., description="Text to analyze", min_length=1)

    # Tokenization
    tokenizer: Literal["word_punct", "alnum"] = Field(
        default="word_punct",
        description="Tokenization mode: 'word_punct' includes punctuation, 'alnum' only alphanumeric"
    )
    lowercase: bool = Field(default=True, description="Convert tokens to lowercase")
    strip_accents: bool = Field(default=False, description="Strip diacritics via NFKD normalization")

    # Metrics selection
    metrics: str = Field(
        default="auto",
        description="Metrics to compute: 'auto' (auto-select by length), 'all', or CSV like 'mtld,hdd,yulek'"
    )
    include_ttr: bool = Field(default=False, description="Include TTR when metrics='auto'")
    distinct_max_n: int = Field(default=0, ge=0, description="Compute distinct-n for n=1..N (0=disabled)")

    # Metric-specific parameters
    mtld_threshold: float = Field(default=0.72, description="MTLD factor threshold")
    mtld_min_factor_len: int = Field(default=10, ge=1, description="Minimum factor length for MTLD")
    hdd_sample_size: int = Field(default=42, ge=1, description="HD-D sample size")
    brunet_alpha: float = Field(default=0.165, description="Brunet's W alpha parameter")

    # Thresholds for grades
    thresholds: Optional[LexicalDiversityThresholdsRequest] = Field(
        default=None,
        description="Custom thresholds for GREEN/AMBER/RED grades"
    )

    # Decision logic
    decision_mode: Literal["consensus"] = Field(default="consensus", description="Decision aggregation mode")
    require_majority: int = Field(default=2, ge=1, description="Majority requirement for final decision")
    deal_breaker_on_red: bool = Field(default=False, description="Treat RED grade as deal-breaker")

    # Top words analysis
    top_words: int = Field(default=0, ge=0, description="Top K most frequent words (0=disabled)")
    include_positions: bool = Field(default=False, description="Include token positions for top words")

    # Target field extraction
    target_field: Optional[str] = Field(
        default=None,
        description="Specific JSON field to analyze (e.g., 'generated_text', 'data.content'). If None, analyzes full text."
    )
    language: Optional[str] = Field(
        default=None,
        description="Language hint for stop-word filtering (ISO 639-1 code or locale, e.g., 'es', 'en-US')"
    )
    filter_stop_words: bool = Field(
        default=True,
        description="Filter stop words from top-word summaries when the language is recognized"
    )

    # Window analysis
    analyze_windows: bool = Field(default=False, description="Enable window-based analysis")
    window_mode: Literal["tokens", "paragraphs"] = Field(
        default="tokens",
        description="Window mode: 'tokens' for sliding windows, 'paragraphs' for paragraph-based"
    )
    window_size: int = Field(default=200, ge=50, description="Token window size (tokens mode only)")
    window_step: int = Field(default=100, ge=10, description="Token window step (tokens mode only)")
    window_top_k: int = Field(default=10, ge=1, description="Top words per window")
    window_preview_chars: int = Field(default=160, ge=0, description="Preview snippet chars per window")
    include_window_metrics: bool = Field(default=False, description="Compute metrics per window (slower)")
    window_include_positions: bool = Field(default=False, description="Include positions for window top words")

    # Auto window on large text
    auto_window_on_large_text: bool = Field(
        default=True,
        description="Automatically enable window analysis for large texts"
    )
    auto_window_token_threshold: int = Field(
        default=1200,
        ge=500,
        description="Token threshold to trigger auto window analysis"
    )

    # Output
    output_mode: Literal["full", "compact"] = Field(
        default="full",
        description="Output verbosity: 'full' includes all details, 'compact' is minimal"
    )


class LexicalDiversityResponse(BaseModel):
    """Response model for lexical diversity analysis"""

    version: str = Field(..., description="Tool version")
    meta: Dict[str, Any] = Field(..., description="Analysis metadata (tokens, tokenizer, etc.)")
    metrics: Dict[str, Any] = Field(..., description="Computed metric values")
    grades: Dict[str, str] = Field(..., description="Metric grades (GREEN/AMBER/RED)")
    decision: Dict[str, Any] = Field(..., description="Final decision with reasoning")
    top_words: Optional[List[Any]] = Field(None, description="Most frequent words (if enabled)")
    windows: Optional[List[Any]] = Field(None, description="Window analysis results (if enabled)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================================
# REPETITION ANALYZER MODELS
# ============================================================================

class RepetitionAnalyzerRequest(BaseModel):
    """Request model for repetition analyzer endpoint"""

    # Input (required)
    text: str = Field(..., description="Text to analyze", min_length=1)

    # N-gram settings
    min_n: int = Field(default=2, ge=1, description="Minimum n-gram length")
    max_n: int = Field(default=5, ge=1, description="Maximum n-gram length")
    min_count: int = Field(default=2, ge=1, description="Report phrases with at least this frequency")

    # Tokenization
    tokenizer: Literal["word_punct", "alnum"] = Field(
        default="word_punct",
        description="Tokenization mode: 'word_punct' includes punctuation, 'alnum' only alphanumeric"
    )
    lowercase: bool = Field(default=True, description="Convert tokens to lowercase")
    strip_accents: bool = Field(default=False, description="Strip diacritics via NFKD normalization")

    # Sentence handling
    respect_sentences: bool = Field(
        default=False,
        description="Don't allow n-grams to cross sentence boundaries"
    )
    sentence_terminators: str = Field(
        default=".,?!;:…",
        description="Sentence terminator characters"
    )
    sentence_mode: Literal["chars", "list"] = Field(
        default="chars",
        description="Interpretation mode for terminators"
    )
    sentence_multis: Optional[str] = Field(
        default=None,
        description="Extra multi-char terminators (CSV, e.g., '...,!!,??')"
    )

    # Algorithm & performance
    algo_mode: Literal["auto", "counter", "multiprocess"] = Field(
        default="auto",
        description="Counting algorithm: 'auto' selects based on text size"
    )
    mp_threshold_tokens: int = Field(
        default=50000,
        ge=1000,
        description="Switch to multiprocess at this token count"
    )
    workers: int = Field(default=0, ge=0, le=16, description="Worker count for multiprocess (0=auto, max=16)")
    core_policy: Literal["physical", "logical", "auto"] = Field(
        default="auto",
        description="CPU core policy for auto workers"
    )

    # Punctuation filtering
    punct_policy: Literal["keep", "drop-edge", "drop-any"] = Field(
        default="drop-edge",
        description="Filter n-grams by punctuation: 'keep', 'drop-edge', or 'drop-any'"
    )

    # Output shaping
    summary_mode: Literal["counts", "ratio", "both", "none"] = Field(
        default="counts",
        description="Summary sections to include"
    )
    details: Literal["none", "top_count", "top_ratio", "all"] = Field(
        default="top_count",
        description="Which phrases to enrich with details"
    )
    details_top_k: int = Field(default=50, ge=0, description="Phrases per n to enrich")
    details_ratios: bool = Field(default=True, description="Include ratio fields in details")
    positions_preview: int = Field(
        default=10,
        ge=0,
        description="Occurrence start indices to expose"
    )
    summary_top_k: int = Field(default=50, ge=0, description="Phrases per n in summaries")
    counts_only_limit_per_n: int = Field(
        default=0,
        ge=0,
        description="Cap candidates per n before building summaries (0=unlimited)"
    )

    # Clustering
    enable_clusters: bool = Field(default=False, description="Enable cluster computation")
    cluster_gap_tokens: int = Field(default=50, ge=0, description="Max token gap to group occurrences")
    clusters_top_k: int = Field(default=0, ge=0, description="Cap clusters per phrase (0=unlimited)")
    clusters_top_by: Literal["count", "density"] = Field(default="count", description="Cluster ranking key")

    # Dense windows
    enable_windows: bool = Field(default=False, description="Enable dense window search")
    window_size_tokens: int = Field(default=200, ge=10, description="Sliding window size")
    top_windows_k: int = Field(default=3, ge=0, description="Top dense windows per phrase")

    # Diagnostics
    diagnostics: Literal["off", "basic", "full"] = Field(
        default="off",
        description="Diagnostics report level"
    )
    diag_len_bins: str = Field(
        default="3-5:5,6-10:3,11-1000:2",
        description="Length bins policy (e.g., '3-5:5,6-10:3,11-1000:2')"
    )
    diag_max_repeat_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Max allowed repeat ratio (0..1)"
    )
    diag_min_distance_tokens: int = Field(
        default=50,
        ge=0,
        description="Min allowed distance for 'too_close' rule"
    )
    diag_cluster_gap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token gap for cluster detection in diagnostics"
    )
    diag_cluster_min_count: int = Field(
        default=3,
        ge=1,
        description="Min occurrences within cluster to flag"
    )
    diag_cluster_max_span_tokens: int = Field(
        default=300,
        ge=1,
        description="Max span for cluster to be considered dense"
    )
    diag_top_k: int = Field(default=100, ge=0, description="Max violations returned")
    diag_digest_k: int = Field(default=10, ge=0, description="Max digest entries per list")

    # Positional bias analysis
    enable_position_metrics: bool = Field(
        default=False,
        description="Enable positional bias analysis (sentence/paragraph/block)"
    )
    pos_bias_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Bias threshold (Wilson lower bound)"
    )
    pos_min_count: int = Field(
        default=2,
        ge=1,
        description="Min occurrences for positional bias"
    )
    pos_conf_z: float = Field(
        default=1.96,
        ge=0.0,
        description="Z-value for Wilson interval (1.96 ~ 95%)"
    )
    pos_report_top_k: int = Field(
        default=100,
        ge=0,
        description="Cap positional violations reported (0=unlimited)"
    )
    pos_candidates_top_k: int = Field(
        default=0,
        ge=0,
        description="Cap phrases per n evaluated for position (0=all)"
    )
    paragraph_break_min_blank_lines: int = Field(
        default=1,
        ge=1,
        description="Blank lines to split paragraphs"
    )
    block_break_min_blank_lines: int = Field(
        default=2,
        ge=1,
        description="Blank lines to split blocks"
    )

    # Output
    output_mode: Literal["full", "compact"] = Field(
        default="full",
        description="Output verbosity: 'full' includes all phrases, 'compact' is minimal"
    )


class RepetitionAnalyzerResponse(BaseModel):
    """Response model for repetition analyzer"""

    model_config = {"extra": "allow"}

    version: str = Field(..., description="Tool version")
    meta: Dict[str, Any] = Field(..., description="Analysis metadata")
    settings: Dict[str, Any] = Field(..., description="Configuration used")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary of top phrases")
    diagnostics: Optional[Dict[str, Any]] = Field(None, description="Diagnostic information")
    phrases: Optional[Any] = Field(None, description="Detailed phrase analysis (dict with integer keys)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/lexical-diversity",
    response_model=LexicalDiversityResponse,
    summary="Analyze Lexical Diversity",
    description=(
        "Analyzes text for vocabulary richness and lexical diversity using multiple "
        "statistical metrics including MTLD, HD-D, Yule's K, Herdan's C, and more. "
        "Returns GREEN/AMBER/RED grades and optional window-based analysis."
    )
)
async def analyze_lexical_diversity_endpoint(request: LexicalDiversityRequest) -> LexicalDiversityResponse:
    """
    Analyze lexical diversity of input text.

    This endpoint provides comprehensive vocabulary richness analysis with:
    - Multiple metrics (MTLD, HD-D, Yule's K, Herdan's C, Brunet's W, TTR)
    - Auto metric selection based on text length
    - Configurable thresholds for GREEN/AMBER/RED grades
    - Top words frequency analysis
    - Optional window-based analysis for detecting local repetition zones
    """
    try:
        start_time = time.time()

        # Build thresholds dict
        thresholds_dict = {}
        if request.thresholds:
            thresholds_dict = {
                "herdan_green_min": request.thresholds.herdan_green_min,
                "herdan_amber_min": request.thresholds.herdan_amber_min,
                "yulek_green_max": request.thresholds.yulek_green_max,
                "yulek_amber_max": request.thresholds.yulek_amber_max,
                "mtld_green_min": request.thresholds.mtld_green_min,
                "mtld_amber_min": request.thresholds.mtld_amber_min,
                "hdd_green_min": request.thresholds.hdd_green_min,
                "hdd_amber_min": request.thresholds.hdd_amber_min,
                "brunet_green_max": request.thresholds.brunet_green_max,
                "brunet_amber_max": request.thresholds.brunet_amber_max,
            }
            # Remove None values
            thresholds_dict = {k: v for k, v in thresholds_dict.items() if v is not None}

        # Extract target field if specified (standalone API handles extraction here)
        text_to_analyze = request.text
        if request.target_field:
            text_to_analyze, was_extracted = extract_target_field(request.text, request.target_field)
            if was_extracted:
                logger.info(f"Extracted field '{request.target_field}' for lexical diversity analysis")
            else:
                logger.warning(
                    f"Field '{request.target_field}' not found or JSON parse failed, "
                    f"analyzing full content as fallback"
                )

        # Build settings object
        settings = LexicalDiversitySettings(
            enabled=True,
            language=request.language,
            filter_stop_words=request.filter_stop_words,
            metrics=request.metrics,
            include_ttr=request.include_ttr,
            distinct_max_n=request.distinct_max_n,
            mtld_threshold=request.mtld_threshold,
            mtld_min_factor_len=request.mtld_min_factor_len,
            hdd_sample_size=request.hdd_sample_size,
            brunet_alpha=request.brunet_alpha,
            tokenizer=request.tokenizer,
            lowercase=request.lowercase,
            strip_accents=request.strip_accents,
            thresholds_overrides=thresholds_dict,
            top_words_k=request.top_words,
            include_positions=request.include_positions,
            decision_policy=LexicalDiversityDecisionPolicy(
                require_majority=request.require_majority,
                deal_breaker_on_red=request.deal_breaker_on_red,
            ),
            score_policy=LexicalDiversityScorePolicy(),
            window_policy=LexicalDiversityWindowPolicy(
                auto_window_on_large_text=request.auto_window_on_large_text,
                auto_window_token_threshold=request.auto_window_token_threshold,
                window_mode=request.window_mode,
                window_size=request.window_size,
                window_step=request.window_step,
                window_top_k=request.window_top_k,
                window_preview_chars=request.window_preview_chars,
                window_include_positions=request.window_include_positions,
                include_window_metrics=request.include_window_metrics,
                analyze_windows=request.analyze_windows,
            ),
        )

        # Run analysis
        result = analyze_text_lexical_diversity(text_to_analyze, settings)

        # Extract result
        analysis = result.analysis
        processing_time_ms = (time.time() - start_time) * 1000

        return LexicalDiversityResponse(
            version=analysis.get("version", "0.2.0"),
            meta=analysis.get("meta", {}),
            metrics=analysis.get("metrics", {}),
            grades=analysis.get("grades", {}),
            decision=analysis.get("decision", {}),
            top_words=analysis.get("top_words"),
            windows=analysis.get("windows"),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Lexical diversity analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post(
    "/repetition",
    response_model=RepetitionAnalyzerResponse,
    summary="Analyze Text Repetition",
    description=(
        "Analyzes exact n-gram repetition patterns in text with clustering, "
        "diagnostics, and positional bias detection. Useful for detecting "
        "over-repetition, dense clusters, and phrase positioning anomalies."
    )
)
async def analyze_repetition_endpoint(request: RepetitionAnalyzerRequest) -> RepetitionAnalyzerResponse:
    """
    Analyze repetition patterns in input text.

    This endpoint provides comprehensive repetition analysis with:
    - N-gram counting (configurable min/max n and min count)
    - Distance metrics between repetitions
    - Cluster detection for grouped repetitions
    - Dense window identification
    - Diagnostics for over-repetition detection
    - Positional bias analysis (sentence/paragraph/block starts/ends)
    - Multi-process support for large texts
    """
    try:
        start_time = time.time()

        # Build configuration object
        config = AnalysisConfig(
            # Counting
            min_n=request.min_n,
            max_n=request.max_n,
            min_count=request.min_count,
            # Tokenization
            tokenizer=request.tokenizer,
            lowercase=request.lowercase,
            strip_accents=request.strip_accents,
            # Sentence handling
            respect_sentences=request.respect_sentences,
            sentence_terminators=request.sentence_terminators,
            sentence_mode=request.sentence_mode,
            sentence_multis=request.sentence_multis or "",
            # Algorithm
            algo_mode=request.algo_mode,
            mp_threshold_tokens=request.mp_threshold_tokens,
            workers=request.workers,
            core_policy=request.core_policy,
            # Punctuation
            punct_policy=request.punct_policy,
            # Output shaping
            summary_mode=request.summary_mode,
            details_ratios=request.details_ratios,
            output_mode=request.output_mode,
            # Details
            details=request.details,
            details_top_k=request.details_top_k,
            positions_preview=request.positions_preview,
            summary_top_k=request.summary_top_k,
            counts_only_limit_per_n=request.counts_only_limit_per_n,
            # Clustering
            enable_clusters=request.enable_clusters,
            cluster_gap_tokens=request.cluster_gap_tokens,
            clusters_top_k=request.clusters_top_k,
            clusters_top_by=request.clusters_top_by,
            # Windows
            enable_windows=request.enable_windows,
            window_size_tokens=request.window_size_tokens,
            top_windows_k=request.top_windows_k,
            # Diagnostics
            diagnostics=request.diagnostics,
            diag_len_bins=request.diag_len_bins,
            diag_max_repeat_ratio=request.diag_max_repeat_ratio,
            diag_min_distance_tokens=request.diag_min_distance_tokens,
            diag_cluster_gap_tokens=request.diag_cluster_gap_tokens,
            diag_cluster_min_count=request.diag_cluster_min_count,
            diag_cluster_max_span_tokens=request.diag_cluster_max_span_tokens,
            diag_top_k=request.diag_top_k,
            diag_digest_k=request.diag_digest_k,
            # Positional
            enable_position_metrics=request.enable_position_metrics,
            pos_bias_threshold=request.pos_bias_threshold,
            pos_min_count=request.pos_min_count,
            pos_conf_z=request.pos_conf_z,
            pos_report_top_k=request.pos_report_top_k,
            pos_candidates_top_k=request.pos_candidates_top_k,
            paragraph_break_min_blank_lines=request.paragraph_break_min_blank_lines,
            block_break_min_blank_lines=request.block_break_min_blank_lines,
        )

        # Run analysis
        result = analyze_text(request.text, config)

        processing_time_ms = (time.time() - start_time) * 1000

        return RepetitionAnalyzerResponse(
            version=result.get("version", "2.3.2"),
            meta=result.get("meta", {}),
            settings=result.get("settings", {}),
            summary=result.get("summary"),
            diagnostics=result.get("diagnostics"),
            phrases=result.get("phrases"),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Repetition analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health Check",
    description="Verify that the analysis endpoints are operational"
)
async def health_check():
    """Health check endpoint for analysis router"""
    return {
        "status": "healthy",
        "endpoints": [
            "/analysis/lexical-diversity",
            "/analysis/repetition"
        ]
    }
