"""
Evidence Grounding Engine for Gran Sabio LLM
=============================================

Orchestrates the complete evidence grounding verification pipeline:
1. Claim extraction (ClaimExtractor)
2. Evidence matching (EvidenceMatcher)
3. Budget scoring (BudgetScorer)
4. Result aggregation (EvidenceGroundingResult)

This is Phase 5 of the Strawberry Integration.

The grounding engine integrates with the QA pipeline as a special layer,
executing alongside semantic QA layers with configurable ordering based
on the on_flag setting.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from ai_service import is_content_filter_stop
from context_budget import ContextAdmissionError
from evidence_grounding.budget_scorer import BudgetScorer
from evidence_grounding.claim_extractor import ClaimExtractor
from evidence_grounding.evidence_matcher import (
    EvidenceMatcher,
    match_claims_to_spans,
)
from llm_routing import resolve_call
from local_execution_policy import LocalExecutionPolicyError
from models import (
    EvidenceGroundingConfig,
    EvidenceGroundingResult,
)

logger = logging.getLogger(__name__)


def get_effective_order(grounding_config: EvidenceGroundingConfig) -> int:
    """Calculate the effective execution order for evidence grounding.

    If order is explicitly set, use that value.
    Otherwise, auto-calculate based on on_flag:
    - deal_breaker/regenerate -> 0 (fail-fast, run first)
    - warn -> 999 (verification-only, run last)

    Args:
        grounding_config: The evidence grounding configuration

    Returns:
        Effective order value (integer)
    """
    if grounding_config.order is not None:
        return grounding_config.order

    # Auto-order based on on_flag
    if grounding_config.on_flag in ("deal_breaker", "regenerate"):
        return 0  # Fail-fast: run before semantic QA
    else:
        return 999  # Verification-only: run after semantic QA


class GroundingEngine:
    """
    Orchestrates the complete evidence grounding verification pipeline.

    This engine combines:
    - ClaimExtractor: Extracts verifiable claims from generated content
    - EvidenceMatcher: Maps claims to evidence spans in context
    - BudgetScorer: Calculates budget gaps using logprobs

    The result integrates with the QA pipeline through QAEvaluation conversion,
    supporting deal-breaker detection and consensus integration.

    Usage:
        engine = GroundingEngine(ai_service)
        result = await engine.run_grounding_check(
            content=generated_content,
            context=original_prompt,
            config=request.evidence_grounding,
        )
    """

    def __init__(self, ai_service):
        """
        Initialize the GroundingEngine.

        Args:
            ai_service: AIService instance (use get_ai_service())
        """
        self.ai_service = ai_service
        self.claim_extractor = ClaimExtractor(ai_service)
        self.evidence_matcher = EvidenceMatcher(ai_service=ai_service)
        self.budget_scorer = BudgetScorer(ai_service)

    async def run_grounding_check(
        self,
        content: str,
        context: str,
        grounding_config: EvidenceGroundingConfig,
        progress_callback: Optional[Callable[[str], Any]] = None,
        stream_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        extra_verbose: bool = False,
        original_request: Optional[Any] = None,
        cancellation_token: Optional[Any] = None,
        local_execution: Optional[Any] = None,
    ) -> EvidenceGroundingResult:
        """
        Run complete evidence grounding verification pipeline.

        Executes the full Strawberry-inspired verification:
        1. Extract substantive claims from content
        2. Match claims to evidence spans in context
        3. Calculate budget gaps using logprobs
        4. Aggregate results and determine pass/fail

        Args:
            content: Generated content to verify
            context: Original context/evidence (prompt + attachments)
            grounding_config: Configuration from request
            progress_callback: Callback for progress updates (for verbose logs)
            stream_callback: Callback for streaming events
            usage_callback: Callback for token usage tracking
            extra_verbose: Enable detailed logging

        Returns:
            EvidenceGroundingResult with all metrics and pass/fail status
        """
        start_time = time.time()
        tokens_used = 0  # Accumulated across all API calls via usage payloads

        def _tracking_usage_callback(payload: Dict[str, Any]) -> None:
            """Accumulate reported tokens and forward the payload downstream."""
            nonlocal tokens_used
            total = payload.get("total_tokens")
            if not isinstance(total, (int, float)):
                total = (payload.get("input_tokens") or 0) + (payload.get("output_tokens") or 0)
            tokens_used += max(0, int(total))
            if usage_callback is not None:
                usage_callback(payload)

        # Providers only report usage when a callback is present, so the
        # tracker is always passed down even without an external callback.
        tracked_usage_callback = _tracking_usage_callback

        extraction_route = resolve_call("evidence.extract_claims", request=original_request)
        classification_route = resolve_call("evidence.classify_spans", request=original_request)
        scoring_route = resolve_call("evidence.score_logprobs", request=original_request)
        # If user specifies a legacy model in config, use it for both phases.
        extraction_model = grounding_config.model or extraction_route.model
        classification_model = grounding_config.model or classification_route.model
        scoring_model = grounding_config.model or scoring_route.model

        if extra_verbose:
            logger.info(f"[GROUNDING] Starting evidence grounding check")
            logger.info(f"[GROUNDING] Extraction model: {extraction_model}")
            logger.info(f"[GROUNDING] Scoring model: {scoring_model}")

        # ===== STEP 1: Extract claims =====
        if progress_callback:
            await self._safe_callback(progress_callback, "Extracting verifiable claims from content...")
        if stream_callback:
            await self._safe_callback(stream_callback, {
                "type": "grounding_phase",
                "phase": "claim_extraction",
                "status": "started"
            })

        try:
            claims = await self.claim_extractor.extract_claims(
                content=content,
                context=context,
                model=extraction_model,
                request=original_request,
                max_claims=grounding_config.max_claims,
                filter_trivial=grounding_config.filter_trivial,
                min_importance=grounding_config.min_claim_importance,
                extra_verbose=extra_verbose,
                usage_callback=tracked_usage_callback,
                cancellation_token=cancellation_token,
                local_execution=local_execution,
            )
        except asyncio.CancelledError:
            raise
        except ContextAdmissionError:
            # A context overflow must fail the grounding check visibly, never
            # become a conservative pass that masks unverified content.
            raise
        except LocalExecutionPolicyError:
            raise
        except Exception as e:
            if is_content_filter_stop(e):
                raise
            logger.error(f"[GROUNDING] Claim extraction failed: {e}")
            # Degrade per on_flag policy: never mask the error as a real pass.
            return self._create_error_result(
                model=extraction_model,
                error_message=f"Claim extraction failed: {e}",
                elapsed_ms=(time.time() - start_time) * 1000,
                on_flag=grounding_config.on_flag,
                tokens_used=tokens_used,
            )

        total_extracted = len(claims)

        # Filter by importance (may already be done by extractor, but ensure)
        filtered_claims = [
            c for c in claims
            if c.importance >= grounding_config.min_claim_importance
        ]
        claims_after_filter = len(filtered_claims)

        if stream_callback:
            await self._safe_callback(stream_callback, {
                "type": "grounding_phase",
                "phase": "claim_extraction",
                "status": "completed",
                "total_extracted": total_extracted,
                "after_filter": claims_after_filter,
            })

        if extra_verbose:
            logger.info(f"[GROUNDING] Extracted {total_extracted} claims, {claims_after_filter} after filter")

        # Short-circuit if no claims to verify
        if not filtered_claims:
            if progress_callback:
                await self._safe_callback(progress_callback, "No verifiable claims found in content")
            return EvidenceGroundingResult(
                enabled=True,
                model_used=f"{extraction_model} (extraction)",
                total_claims_extracted=total_extracted,
                claims_after_filter=0,
                claims_verified=0,
                claims=[],
                flagged_claims=0,
                max_budget_gap=0.0,
                passed=True,  # No claims = nothing to fail
                triggered_action=None,
                verification_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens_used,
            )

        # ===== STEP 2: Spanize context and match claims =====
        if progress_callback:
            await self._safe_callback(progress_callback, "Matching claims to evidence spans...")

        spans = self.evidence_matcher.spanize(context)
        try:
            spans = await self.evidence_matcher.classify_spans(
                model=classification_model,
                request=original_request,
                usage_callback=tracked_usage_callback,
                cancellation_token=cancellation_token,
                local_execution=local_execution,
            )
        except asyncio.CancelledError:
            raise
        except ContextAdmissionError:
            # A context overflow must fail the grounding check visibly, never
            # become a conservative pass that masks unverified content.
            raise
        except LocalExecutionPolicyError:
            raise
        except Exception as e:
            if is_content_filter_stop(e):
                raise
            logger.error(f"[GROUNDING] Span classification failed: {e}")
            return self._create_error_result(
                model=classification_model,
                error_message=f"Span classification failed: {e}",
                elapsed_ms=(time.time() - start_time) * 1000,
                on_flag=grounding_config.on_flag,
                tokens_used=tokens_used,
            )
        matched_claims = match_claims_to_spans(filtered_claims, spans)

        if extra_verbose:
            logger.info(f"[GROUNDING] Context divided into {len(spans)} spans")

        # ===== STEP 3: Score each claim with logprobs =====
        if progress_callback:
            await self._safe_callback(progress_callback, f"Verifying {len(matched_claims)} claims with logprobs...")
        if stream_callback:
            await self._safe_callback(stream_callback, {
                "type": "grounding_phase",
                "phase": "budget_scoring",
                "status": "started",
                "claims_to_verify": len(matched_claims),
            })

        try:
            scored_claims = await self.budget_scorer.score_claims(
                claims=matched_claims,
                spans=spans,
                config_obj=grounding_config,
                request=original_request,
                extra_verbose=extra_verbose,
                usage_callback=tracked_usage_callback,
                cancellation_token=cancellation_token,
                local_execution=local_execution,
            )
        except asyncio.CancelledError:
            raise
        except ContextAdmissionError:
            # A context overflow must fail the grounding check visibly, never
            # become a conservative pass that masks unverified content.
            raise
        except LocalExecutionPolicyError:
            raise
        except Exception as e:
            if is_content_filter_stop(e):
                raise
            logger.error(f"[GROUNDING] Budget scoring failed: {e}")
            return self._create_error_result(
                model=scoring_model,
                error_message=f"Budget scoring failed: {e}",
                elapsed_ms=(time.time() - start_time) * 1000,
                on_flag=grounding_config.on_flag,
                tokens_used=tokens_used,
            )

        # ===== STEP 4: Aggregate results =====
        flagged_count = sum(1 for r in scored_claims if r.flagged)
        max_gap = max((r.budget_gap for r in scored_claims), default=0.0)

        # Determine pass/fail based on max_flagged_claims threshold
        passed = flagged_count < grounding_config.max_flagged_claims
        triggered_action = None if passed else grounding_config.on_flag

        elapsed_ms = (time.time() - start_time) * 1000

        if stream_callback:
            await self._safe_callback(stream_callback, {
                "type": "grounding_phase",
                "phase": "budget_scoring",
                "status": "completed",
                "flagged_claims": flagged_count,
                "max_budget_gap": max_gap,
                "passed": passed,
            })

        if progress_callback:
            status = "PASSED" if passed else f"FAILED ({flagged_count} claims flagged)"
            await self._safe_callback(progress_callback, f"Evidence grounding check: {status}")

        if extra_verbose:
            logger.info(
                f"[GROUNDING] Complete: {flagged_count}/{len(scored_claims)} flagged, "
                f"max_gap={max_gap:.3f}, passed={passed}, time={elapsed_ms:.0f}ms"
            )

        return EvidenceGroundingResult(
            enabled=True,
            model_used=f"{extraction_model} (extraction) + {scoring_model} (scoring)",
            total_claims_extracted=total_extracted,
            claims_after_filter=claims_after_filter,
            claims_verified=len(scored_claims),
            claims=scored_claims,
            flagged_claims=flagged_count,
            max_budget_gap=max_gap,
            passed=passed,
            triggered_action=triggered_action,
            verification_time_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

    def _create_error_result(
        self,
        model: str,
        error_message: str,
        elapsed_ms: float,
        on_flag: str = "warn",
        tokens_used: int = 0,
    ) -> EvidenceGroundingResult:
        """Create the result for a verification that could not run.

        The degradation policy follows ``on_flag``:
        - ``deal_breaker``/``regenerate``: the error counts as a verification
          failure (``passed=False`` with the configured action), so unverified
          content is blocked or regenerated instead of silently passing.
        - ``warn``: keeps the historical conservative pass so an infrastructure
          error never blocks a generation the user only wanted to be warned
          about; the error is logged and ``claims_verified=0`` distinguishes
          it from a real verification.

        Either way the result is marked ``degraded=True`` with the original
        error in ``verification_error`` so the QA pipeline can surface the
        failure instead of confusing it with a real pass.

        Args:
            model: Model that was in use when the error happened
            error_message: Human-readable error description
            elapsed_ms: Elapsed verification time in milliseconds
            on_flag: Configured action policy from EvidenceGroundingConfig
            tokens_used: Tokens consumed before the error

        Returns:
            EvidenceGroundingResult following the policy above
        """
        if on_flag in ("deal_breaker", "regenerate"):
            logger.warning(
                f"[GROUNDING] Verification error treated as '{on_flag}' failure: {error_message}"
            )
            return EvidenceGroundingResult(
                enabled=True,
                model_used=model,
                total_claims_extracted=0,
                claims_after_filter=0,
                claims_verified=0,
                claims=[],
                flagged_claims=0,
                max_budget_gap=0.0,
                passed=False,
                triggered_action=on_flag,
                degraded=True,
                verification_error=error_message,
                verification_time_ms=elapsed_ms,
                tokens_used=tokens_used,
            )
        logger.warning(f"[GROUNDING] Returning conservative pass due to error: {error_message}")
        return EvidenceGroundingResult(
            enabled=True,
            model_used=model,
            total_claims_extracted=0,
            claims_after_filter=0,
            claims_verified=0,
            claims=[],
            flagged_claims=0,
            max_budget_gap=0.0,
            passed=True,  # Conservative: warn mode never blocks on errors
            triggered_action=None,
            degraded=True,
            verification_error=error_message,
            verification_time_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

    async def _safe_callback(
        self,
        callback: Callable,
        data: Any,
    ) -> None:
        """Safely invoke a callback, handling both sync and async functions."""
        try:
            result = callback(data)
            # If it's a coroutine, await it
            if hasattr(result, '__await__'):
                await result
        except Exception as e:
            logger.warning(f"[GROUNDING] Callback error (non-fatal): {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_grounding_check(
    content: str,
    context: str,
    grounding_config: Optional[EvidenceGroundingConfig] = None,
    extra_verbose: bool = False,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> EvidenceGroundingResult:
    """
    Convenience function to run evidence grounding with shared AI service.

    This is a standalone function for testing or direct usage outside
    the QA pipeline. For production use, the GroundingEngine is typically
    invoked through QAEngine.

    Args:
        content: Generated content to verify
        context: Original context/evidence
        grounding_config: Configuration (uses defaults if None)
        extra_verbose: Enable detailed logging
        usage_callback: Token usage tracking callback

    Returns:
        EvidenceGroundingResult
    """
    from ai_service import get_ai_service

    if grounding_config is None:
        grounding_config = EvidenceGroundingConfig(enabled=True)

    engine = GroundingEngine(get_ai_service())
    return await engine.run_grounding_check(
        content=content,
        context=context,
        grounding_config=grounding_config,
        extra_verbose=extra_verbose,
        usage_callback=usage_callback,
    )
