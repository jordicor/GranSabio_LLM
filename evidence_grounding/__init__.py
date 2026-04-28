"""
Evidence Grounding Module for Gran Sabio LLM Engine
====================================================

This module implements the Strawberry evidence grounding system for detecting
procedural hallucination (confabulation) by measuring whether the model actually
relied on cited evidence.

Phase 2 Implementation:
- ClaimExtractor: Extracts atomic, verifiable claims from generated content

Phase 3 Implementation:
- EvidenceMatcher: Matches claims to evidence spans in context
- spanize_context: Divides context into labeled spans (S0, S1, etc.)
- Span classification (assertion/question/instruction/empty)
- Span scrubbing for pseudo-prior calculation

Phase 4 Implementation:
- BudgetScorer: Calculates budget gaps using logprobs
- kl_bernoulli: KL divergence for Bernoulli distributions
- extract_yes_probability: Extract P(YES) from logprobs

Phase 5 Implementation:
- GroundingEngine: Orchestrates full pipeline, integrates with QA engine
- get_effective_order: Auto-calculates execution order based on on_flag
"""

from .budget_scorer import (
    ENTAILMENT_SYSTEM_PROMPT,
    BudgetScorer,
    build_entailment_prompt,
    calculate_budget_metrics,
    extract_yes_probability,
    kl_bernoulli,
    score_claims,
)
from .claim_extractor import ClaimExtractor, extract_claims
from .evidence_matcher import (
    EvidenceMatcher,
    classify_span,
    find_matching_spans,
    format_spans_for_prompt,
    match_claims_to_spans,
    scrub_spans,
    spanize_context,
)
from .grounding_engine import (
    GroundingEngine,
    get_effective_order,
    run_grounding_check,
)

__all__ = [
    # Phase 2: Claim Extraction
    "ClaimExtractor",
    "extract_claims",
    # Phase 3: Evidence Matching
    "EvidenceMatcher",
    "spanize_context",
    "classify_span",
    "match_claims_to_spans",
    "scrub_spans",
    "find_matching_spans",
    "format_spans_for_prompt",
    # Phase 4: Budget Scoring
    "BudgetScorer",
    "score_claims",
    "kl_bernoulli",
    "calculate_budget_metrics",
    "extract_yes_probability",
    "build_entailment_prompt",
    "ENTAILMENT_SYSTEM_PROMPT",
    # Phase 5: Grounding Engine
    "GroundingEngine",
    "run_grounding_check",
    "get_effective_order",
]
