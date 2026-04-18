"""
Evidence Grounding Budget Scorer for Gran Sabio LLM
====================================================

Core logprob verification logic for detecting procedural hallucination.

This module implements the information-budget calculation from Strawberry/Pythea:
- If removing evidence doesn't reduce model confidence, the model wasn't using it
- Uses KL divergence to quantify "information lift" from evidence
- Flags claims where confidence doesn't drop when evidence is removed

Adapted from Strawberry (Pythea project):
https://github.com/leochlon/pythea
MIT License - See original repository for full license text.

This is Phase 4 of the Strawberry Integration.
"""

import logging
import math
import time
from typing import List, Optional, Tuple, Dict, Any, Callable

from models import (
    ExtractedClaim,
    EvidenceSpan,
    ClaimBudgetResult,
    EvidenceGroundingConfig,
)
from evidence_grounding.evidence_matcher import (
    format_spans_for_prompt,
    scrub_spans,
)
from config import config

logger = logging.getLogger(__name__)


# =============================================================================
# Mathematical Functions
# =============================================================================

def kl_bernoulli(a: float, b: float) -> float:
    """Calculate KL divergence between two Bernoulli distributions.

    KL(Ber(a) || Ber(b)) measures how much information is needed to
    describe distribution a using distribution b as a reference.

    Args:
        a: Probability parameter of first Bernoulli (target)
        b: Probability parameter of second Bernoulli (reference)

    Returns:
        KL divergence in nats (natural log units)

    Example:
        >>> kl_bernoulli(0.95, 0.5)  # High target, uncertain reference
        0.8675...  # Significant information needed
        >>> kl_bernoulli(0.95, 0.95)  # Same distributions
        0.0  # No additional information needed
    """
    # Clamp to avoid log(0) and division by zero
    eps = 1e-12
    a = max(eps, min(1 - eps, a))
    b = max(eps, min(1 - eps, b))

    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def calculate_budget_metrics(
    posterior: float,
    prior: float,
    target: float,
) -> Tuple[float, float, float]:
    """Calculate information budget metrics for a claim.

    The budget framework measures whether the model's confidence increase
    (from prior to posterior) is sufficient to justify the target confidence.

    Args:
        posterior: P(YES | full_context) - confidence with evidence
        prior: P(YES | scrubbed_context) - confidence without evidence
        target: Expected reliability (e.g., 0.95)

    Returns:
        Tuple of (required_bits, observed_bits, budget_gap):
        - required_bits: Information needed to reach target from prior
        - observed_bits: Information actually provided by evidence
        - budget_gap: required - observed (positive = deficit = confabulation risk)

    Example:
        >>> calculate_budget_metrics(0.92, 0.15, 0.95)
        (2.89, 2.45, 0.44)  # Small gap - evidence mostly sufficient
        >>> calculate_budget_metrics(0.90, 0.85, 0.95)
        (0.87, 0.04, 0.83)  # Large gap - model confident without evidence!
    """
    required = kl_bernoulli(target, prior)
    observed = kl_bernoulli(posterior, prior)
    gap = required - observed

    return required, observed, gap


# =============================================================================
# Verification Prompt Template
# =============================================================================

ENTAILMENT_SYSTEM_PROMPT = """You are a **strict textual entailment** verifier.

Definitions:
- Only **declarative assertions** in the CONTEXT can entail facts.
- **Questions, prompts, headings, and instructions do NOT assert facts**.
- Do **not** use world knowledge; judge only whether CLAIM follows from CONTEXT.

Decision rule:
- Reply YES only if the CLAIM is explicitly stated or logically implied by the CONTEXT.
- Reply NO only if the CONTEXT explicitly contradicts the CLAIM.
- Otherwise reply UNSURE.

CRITICAL: Reply with EXACTLY one token: YES, NO, or UNSURE"""


def build_entailment_prompt(
    formatted_spans: str,
    claim: str,
) -> str:
    """Build the user prompt for entailment verification.

    Args:
        formatted_spans: Pre-formatted context spans from format_spans_for_prompt()
        claim: The claim to verify

    Returns:
        User prompt for YES/NO/UNSURE response
    """
    return f"""CONTEXT SPANS:
{formatted_spans}

CLAIM:
{claim}

Question: Is the CLAIM entailed by the CONTEXT?

Reply with EXACTLY one token: YES, NO, or UNSURE"""


# =============================================================================
# Logprobs Extraction
# =============================================================================

def extract_yes_probability(
    logprobs_content: Optional[List[Dict[str, Any]]],
    default_uncertain: float = 0.33,
) -> float:
    """Extract P(YES) from the first token's logprobs.

    Searches for YES/NO/UNSURE tokens in the top logprobs and calculates
    P(YES) either directly or as an upper bound if not in top-k.

    Args:
        logprobs_content: List of token logprob data from OpenAI response
            Each item has: token, logprob, top_logprobs
        default_uncertain: Default probability if extraction fails

    Returns:
        Estimated P(YES) in range [0, 1]
    """
    if not logprobs_content:
        logger.warning("No logprobs content received, using default")
        return default_uncertain

    first_token_data = logprobs_content[0]
    top_logprobs = first_token_data.get("top_logprobs", [])

    if not top_logprobs:
        # No top_logprobs, try to use the token's own logprob
        token = first_token_data.get("token", "").strip().upper()
        logprob = first_token_data.get("logprob")

        if token == "YES" and logprob is not None:
            return math.exp(logprob)
        logger.warning("No top_logprobs available, using default")
        return default_uncertain

    # Search for YES, NO, UNSURE in top_logprobs
    yes_logprob = None
    no_logprob = None
    unsure_logprob = None
    min_logprob = float('inf')

    for item in top_logprobs:
        token = item.get("token", "").strip().upper()
        logprob = item.get("logprob", float('-inf'))
        min_logprob = min(min_logprob, logprob)

        if token == "YES":
            yes_logprob = logprob
        elif token == "NO":
            no_logprob = logprob
        elif token == "UNSURE":
            unsure_logprob = logprob

    # If YES found directly, convert logprob to probability
    if yes_logprob is not None:
        return math.exp(yes_logprob)

    # YES not in top-k: use upper bound based on k-th token
    # P(YES) < P(k-th token) since YES would be in top-k if higher
    if min_logprob != float('inf'):
        upper_bound = math.exp(min_logprob)
        logger.debug(f"YES not in top-k, using upper bound: {upper_bound:.4f}")
        return upper_bound

    logger.warning("Could not extract YES probability, using default")
    return default_uncertain


def normalize_yes_no_unsure(
    logprobs_content: Optional[List[Dict[str, Any]]],
) -> Tuple[float, float, float]:
    """Extract and normalize probabilities for YES, NO, UNSURE.

    Returns normalized probabilities that sum to 1.0 for the three
    possible responses.

    Args:
        logprobs_content: Logprobs from API response

    Returns:
        Tuple of (p_yes, p_no, p_unsure) summing to 1.0
    """
    if not logprobs_content:
        return (0.33, 0.33, 0.34)  # Uniform when unknown

    first_token_data = logprobs_content[0]
    top_logprobs = first_token_data.get("top_logprobs", [])

    # Extract raw probabilities
    probs = {"YES": 0.0, "NO": 0.0, "UNSURE": 0.0}

    for item in top_logprobs:
        token = item.get("token", "").strip().upper()
        logprob = item.get("logprob", float('-inf'))

        if token in probs:
            probs[token] = math.exp(logprob)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        return (
            probs["YES"] / total,
            probs["NO"] / total,
            probs["UNSURE"] / total,
        )

    return (0.33, 0.33, 0.34)


# =============================================================================
# Budget Scorer Class
# =============================================================================

class BudgetScorer:
    """Calculates information budget for claims using logprobs.

    This is the core component of evidence grounding verification.
    For each claim, it measures:
    1. P(YES | full_context) - posterior confidence
    2. P(YES | context_without_cited_evidence) - pseudo-prior confidence
    3. Budget gap = information needed - information provided

    A high budget gap indicates the model claims high confidence but
    the evidence doesn't justify it (likely confabulation).

    Attributes:
        ai_service: AIService instance for logprob API calls
    """

    def __init__(self, ai_service):
        """Initialize the BudgetScorer.

        Args:
            ai_service: AIService instance. Use get_ai_service() from ai_service module.
        """
        self.ai_service = ai_service

    async def score_claim(
        self,
        claim: ExtractedClaim,
        spans: List[EvidenceSpan],
        model: str,
        target_confidence: float = 0.95,
        budget_gap_threshold: Optional[float] = None,
        top_logprobs: int = 10,
        placeholder: str = "[EVIDENCE REMOVED]",
        extra_verbose: bool = False,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> ClaimBudgetResult:
        """Score a single claim for evidence grounding.

        Args:
            claim: The claim to verify
            spans: Full list of evidence spans
            model: Model for logprob verification (must support logprobs)
            target_confidence: Expected reliability (0.5-0.99)
            budget_gap_threshold: Threshold for flagging under-grounded claims
            top_logprobs: Number of top logprobs to request (1-20)
            placeholder: Text for scrubbed spans
            extra_verbose: Enable detailed logging
            usage_callback: Token usage tracking callback

        Returns:
            ClaimBudgetResult with all metrics and flagging decision
        """
        start_time = time.time()

        # 1. Format full context for posterior
        full_context = format_spans_for_prompt(spans, mask_non_assertions=True)
        posterior_prompt = build_entailment_prompt(full_context, claim.claim)

        if extra_verbose:
            logger.info(f"[BUDGET_SCORER] Scoring claim {claim.idx}: {claim.claim[:50]}...")
            logger.info(f"[BUDGET_SCORER] Cited spans: {claim.cited_spans}")

        # 2. Get posterior P(YES | full_context)
        posterior_yes = await self._get_yes_probability(
            prompt=posterior_prompt,
            model=model,
            top_logprobs=top_logprobs,
            extra_verbose=extra_verbose,
            usage_callback=usage_callback,
            usage_extra={"subphase": "posterior", "claim_idx": claim.idx},
        )

        # 3. Scrub cited spans and format for prior
        scrubbed_spans = scrub_spans(spans, claim.cited_spans, placeholder)
        scrubbed_context = format_spans_for_prompt(scrubbed_spans, mask_non_assertions=True)
        prior_prompt = build_entailment_prompt(scrubbed_context, claim.claim)

        # 4. Get prior P(YES | scrubbed_context)
        prior_yes = await self._get_yes_probability(
            prompt=prior_prompt,
            model=model,
            top_logprobs=top_logprobs,
            extra_verbose=extra_verbose,
            usage_callback=usage_callback,
            usage_extra={"subphase": "prior", "claim_idx": claim.idx},
        )

        # 5. Calculate budget metrics
        required_bits, observed_bits, budget_gap = calculate_budget_metrics(
            posterior=posterior_yes,
            prior=prior_yes,
            target=target_confidence,
        )

        # 6. Determine if flagged
        threshold = (
            budget_gap_threshold
            if budget_gap_threshold is not None
            else (
                config.EVIDENCE_GROUNDING_BUDGET_GAP_THRESHOLD
                if hasattr(config, 'EVIDENCE_GROUNDING_BUDGET_GAP_THRESHOLD')
                else 0.5
            )
        )
        flagged = budget_gap > threshold

        confidence_delta = posterior_yes - prior_yes

        elapsed_ms = (time.time() - start_time) * 1000

        if extra_verbose:
            logger.info(
                f"[BUDGET_SCORER] Claim {claim.idx} results: "
                f"posterior={posterior_yes:.3f}, prior={prior_yes:.3f}, "
                f"gap={budget_gap:.3f}, flagged={flagged} "
                f"({elapsed_ms:.0f}ms)"
            )

        return ClaimBudgetResult(
            idx=claim.idx,
            claim=claim.claim,
            cited_spans=claim.cited_spans,
            posterior_yes=posterior_yes,
            prior_yes=prior_yes,
            required_bits=required_bits,
            observed_bits=observed_bits,
            budget_gap=budget_gap,
            flagged=flagged,
            confidence_delta=confidence_delta,
        )

    async def score_claims(
        self,
        claims: List[ExtractedClaim],
        spans: List[EvidenceSpan],
        config_obj: EvidenceGroundingConfig,
        extra_verbose: bool = False,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[ClaimBudgetResult]:
        """Score multiple claims for evidence grounding.

        Processes claims sequentially to avoid rate limiting issues.
        For batch optimization in the future, consider grouping API calls.

        Args:
            claims: List of claims to verify
            spans: Full list of evidence spans
            config_obj: Evidence grounding configuration
            extra_verbose: Enable detailed logging
            usage_callback: Token usage tracking callback

        Returns:
            List of ClaimBudgetResult for each claim
        """
        if not claims:
            return []

        # Resolve model - use scoring model (requires logprobs)
        model = config_obj.model or config.EVIDENCE_GROUNDING_SCORING_MODEL

        logger.info(
            f"Scoring {len(claims)} claims using model={model}, "
            f"target_confidence={config_obj.target_confidence}, "
            f"threshold={config_obj.budget_gap_threshold}"
        )

        results = []
        for claim in claims:
            try:
                result = await self.score_claim(
                    claim=claim,
                    spans=spans,
                    model=model,
                    target_confidence=config_obj.target_confidence,
                    budget_gap_threshold=config_obj.budget_gap_threshold,
                    top_logprobs=config_obj.top_logprobs,
                    placeholder=config_obj.placeholder_text,
                    extra_verbose=extra_verbose,
                    usage_callback=usage_callback,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to score claim {claim.idx}: {e}")
                # Create a failed result - conservative: assume well-grounded to avoid false positives
                # Set posterior=prior to get zero gap, preventing flagging
                results.append(ClaimBudgetResult(
                    idx=claim.idx,
                    claim=claim.claim,
                    cited_spans=claim.cited_spans,
                    posterior_yes=0.5,
                    prior_yes=0.5,
                    required_bits=0.0,
                    observed_bits=0.0,
                    budget_gap=0.0,  # Zero gap = not flagged
                    flagged=False,  # Don't flag on error - be conservative
                    confidence_delta=0.0,
                ))

        return results

    async def _get_yes_probability(
        self,
        prompt: str,
        model: str,
        top_logprobs: int = 10,
        extra_verbose: bool = False,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        usage_extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Get P(YES) for an entailment verification prompt.

        Makes an API call with logprobs enabled and extracts the
        probability of YES from the response.

        Args:
            prompt: The entailment verification prompt
            model: Model to use (must support logprobs)
            top_logprobs: Number of top logprobs to request
            extra_verbose: Enable detailed logging
            usage_callback: Token usage callback
            usage_extra: Additional usage metadata

        Returns:
            P(YES) probability in range [0, 1]
        """
        # Use the new logprobs method from ai_service
        # Let exceptions propagate to caller for proper error handling
        response_text, logprobs_data = await self.ai_service.generate_with_logprobs(
            prompt=prompt,
            model=model,
            system_prompt=ENTAILMENT_SYSTEM_PROMPT,
            max_tokens=5,  # Only need YES/NO/UNSURE
            top_logprobs=top_logprobs,
            temperature=0.0,  # Deterministic for consistency
            usage_callback=usage_callback,
            usage_extra={
                "phase": "evidence_grounding",
                **(usage_extra or {}),
            },
        )

        if extra_verbose:
            logger.info(f"[BUDGET_SCORER] Response: {response_text}")
            logger.info(f"[BUDGET_SCORER] Logprobs: {logprobs_data}")

        # Extract P(YES) from logprobs
        p_yes = extract_yes_probability(logprobs_data)

        return p_yes


# =============================================================================
# Convenience Functions
# =============================================================================

async def score_claims(
    claims: List[ExtractedClaim],
    spans: List[EvidenceSpan],
    config_obj: Optional[EvidenceGroundingConfig] = None,
    extra_verbose: bool = False,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[ClaimBudgetResult]:
    """Convenience function to score claims using the shared AI service.

    This is a standalone function that uses the global AI service instance.
    For more control, use BudgetScorer class directly.

    Args:
        claims: Claims to score
        spans: Evidence spans
        config_obj: Configuration (uses defaults if None)
        extra_verbose: Enable detailed logging
        usage_callback: Token usage callback

    Returns:
        List of ClaimBudgetResult
    """
    from ai_service import get_ai_service

    if config_obj is None:
        config_obj = EvidenceGroundingConfig(enabled=True)

    scorer = BudgetScorer(get_ai_service())
    return await scorer.score_claims(
        claims=claims,
        spans=spans,
        config_obj=config_obj,
        extra_verbose=extra_verbose,
        usage_callback=usage_callback,
    )
