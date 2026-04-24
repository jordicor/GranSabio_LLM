"""
QA Engine Module for Gran Sabio LLM Engine
==========================================

Multi-layer quality assurance evaluation system that processes content
through multiple evaluation criteria using different AI models.

Changes (2025-11-06):
- Make QA decide the strategy (edit vs regenerate): always request ranges in 'auto'.
- Simplify and normalize the structured JSON returned by QA for edit ranges.
- Prefer paragraph-level scopes; remove keyword heuristics.
- Propagate QA 'edit_strategy_recommendation' via evaluation.metadata.
- Fix bug: undefined 'qa_model_names' in evaluate_all_layers.
- Load dotenv to ensure API keys are available to downstream services.
"""

import asyncio
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple, TYPE_CHECKING
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    from logging_utils import PhaseLogger
    from models import ImageData

from ai_service import AIService, get_ai_service, AIRequestError
from qa_evaluation_service import QAEvaluationService, QAResponseParseError
from qa_result_utils import (
    apply_gran_sabio_false_positive_override,
    build_qa_counts,
    is_technical_qa_failure,
    semantic_deal_breakers,
)
from qa_scheduler import (
    QAScheduler,
    QASchedulerPolicy,
    QASchedulerResult,
    QASchedulerSlot,
    QASchedulerTechnicalFailure,
    QASchedulerUnavailableError,
)
from usage_tracking import UsageTracker
from models import QALayer, QAEvaluation, EvidenceGroundingConfig, EvidenceGroundingResult
from config import EDITABLE_CONTENT_TYPES, config
from model_aliasing import ModelAliasRegistry, get_evaluator_alias
from qa_bypass_engine import QABypassEngine
from gran_sabio import GranSabioInvocationError, GranSabioProcessCancelled
from evidence_grounding import GroundingEngine, get_effective_order


logger = logging.getLogger(__name__)


class QAProcessCancelled(Exception):
    """Raised when a QA evaluation flow is cancelled by the user."""


class QAModelUnavailableError(RuntimeError):
    """Raised when a QA model cannot be queried reliably anymore."""


CancelCallback = Optional[Callable[[], Awaitable[bool]]]


def _qa_model_result_key(
    model_name: str,
    index: int,
    qa_model_names: List[str],
    alias_registry: Optional[ModelAliasRegistry],
) -> str:
    """Use slot keys only when repeated real model names would collide."""

    if alias_registry and qa_model_names.count(model_name) > 1:
        return alias_registry.qa_slot_id(index)
    return model_name


def _attach_evaluator_identity(
    evaluation: QAEvaluation,
    model_name: str,
    index: int,
    alias_registry: Optional[ModelAliasRegistry],
) -> QAEvaluation:
    """Attach prompt-facing slot identity while preserving the real model field."""

    if alias_registry:
        alias_registry.apply_to_evaluation(evaluation, slot_id=alias_registry.qa_slot_id(index))
    else:
        current_alias = getattr(evaluation, "evaluator_alias", None)
        if not isinstance(current_alias, str) or not current_alias.strip():
            evaluation.evaluator_alias = model_name
    return evaluation


def calculate_qa_timeout_for_model(
    model: str,
    reasoning_effort: Optional[str] = None,
    thinking_budget_tokens: Optional[int] = None
) -> float:
    """
    Calculate timeout for a single QA model evaluation.

    QA needs more time than generation because it must:
    - Read and analyze generated content
    - Read and understand evaluation criteria
    - Read context from original request
    - Reason about compliance and quality
    """
    # Get base reasoning timeout from config
    reasoning_timeout = config.get_reasoning_timeout_seconds(
        model, reasoning_effort, thinking_budget_tokens
    )

    if reasoning_timeout and reasoning_timeout > 0:
        # Apply multiplier for QA complexity
        timeout = reasoning_timeout * config.QA_TIMEOUT_MULTIPLIER
        logger.info(
            f"QA timeout for {model}: {timeout}s "
            f"(base reasoning: {reasoning_timeout}s × multiplier: {config.QA_TIMEOUT_MULTIPLIER})"
        )
        return timeout
    else:
        # Non-reasoning model: use base timeout
        timeout = float(config.QA_BASE_TIMEOUT)
        logger.info(f"QA timeout for {model}: {timeout}s (base timeout for non-reasoning model)")
        return timeout


def calculate_comprehensive_qa_timeout(
    layers: List[Any],
    qa_models: List[Any]
) -> float:
    """
    Calculate total timeout for comprehensive QA evaluation.
    """
    from models import QAModelConfig

    # Calculate the max timeout needed per layer (models run in parallel)
    max_timeout_per_layer = 0.0

    for model in qa_models:
        # Normalize model to extract configuration
        if isinstance(model, str):
            model_name = model
            reasoning_effort = None
            thinking_budget_tokens = None
        else:
            model_name = model.model
            reasoning_effort = getattr(model, 'reasoning_effort', None)
            thinking_budget_tokens = getattr(model, 'thinking_budget_tokens', None)

        individual_timeout = calculate_qa_timeout_for_model(
            model_name,
            reasoning_effort,
            thinking_budget_tokens
        )
        max_timeout_per_layer = max(max_timeout_per_layer, individual_timeout)

    num_layers = len(layers)
    comprehensive_timeout = (max_timeout_per_layer * num_layers) + config.QA_COMPREHENSIVE_TIMEOUT_MARGIN

    logger.info(
        f"Comprehensive QA timeout: {comprehensive_timeout}s "
        f"({num_layers} layers × max {max_timeout_per_layer}s per layer + {config.QA_COMPREHENSIVE_TIMEOUT_MARGIN}s margin)"
    )

    return comprehensive_timeout


class QAEngine:
    """Multi-layer Quality Assurance Engine"""
    
    def __init__(self, ai_service: Optional[AIService] = None, bypass_engine: Optional[QABypassEngine] = None):
        """Initialize QA Engine with optional shared AI service and bypass engine."""
        self.ai_service = ai_service if ai_service is not None else get_ai_service()
        self.qa_evaluator = QAEvaluationService(self.ai_service)
        self.bypass_engine = bypass_engine if bypass_engine is not None else QABypassEngine()
        self.grounding_engine = GroundingEngine(self.ai_service)
        self._qa_failure_tracker: Dict[str, Dict[str, int]] = defaultdict(dict)

    def clear_session_state(self, session_id: str) -> None:
        """Release QA runtime state scoped to an expired session."""
        self._qa_failure_tracker.pop(session_id, None)

    def _should_request_edit_info(
        self,
        mode: str,
        content_type: str
    ) -> bool:
        """
        Decide whether to request edit info (editable, edit_strategy, edit_groups) from QA.

        Simplified logic:
        - 'never' -> False
        - 'always' -> True
        - 'auto' -> True only for editable content types (articles, biographies, etc.)
        """
        if mode == "never":
            return False
        if mode == "always":
            return True

        # Auto mode: only for narrative content
        return content_type in EDITABLE_CONTENT_TYPES

    def _increment_model_failure(self, session_id: Optional[str], model_name: str) -> int:
        if not session_id:
            return 1
        tracker = self._qa_failure_tracker.setdefault(session_id, {})
        tracker[model_name] = tracker.get(model_name, 0) + 1
        return tracker[model_name]

    def _reset_model_failure(self, session_id: Optional[str], model_name: str) -> None:
        if not session_id:
            return
        tracker = self._qa_failure_tracker.get(session_id)
        if not tracker:
            return
        if model_name in tracker:
            tracker.pop(model_name, None)
        if not tracker:
            self._qa_failure_tracker.pop(session_id, None)

    def _qa_failure_threshold(self) -> int:
        try:
            threshold = int(getattr(config, "QA_MODEL_FAILURE_THRESHOLD", 5))
            return threshold if threshold > 0 else 5
        except Exception:
            return 5

    def _estimate_fast_global_tokens(self, text: str) -> int:
        """Estimate prompt tokens conservatively enough for fast_global guardrails."""
        if not text:
            return 1
        byte_estimate = (len(text.encode("utf-8")) + 2) // 3
        char_estimate = (len(text) + 2) // 3
        word_estimate = int((len(text.split()) * 3 + 1) // 2)
        return max(1, byte_estimate, char_estimate, word_estimate)

    def _fast_global_token_limit(self) -> int:
        try:
            limit = int(getattr(config, "QA_FAST_GLOBAL_MAX_ESTIMATED_TOKENS", 12000))
            return limit if limit > 0 else 12000
        except Exception:
            return 12000

    async def evaluate_content(
        self,
        content: str,
        layer: QALayer,
        model: Any,  # Can be str or QAModelConfig
        original_request: Optional[Any] = None,
        extra_verbose: bool = False,
        stream_callback: Optional[callable] = None,
        usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        request_edit_info: bool = True,
        phase_logger: Optional["PhaseLogger"] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        edit_history: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> QAEvaluation:
        """
        Evaluate content using a specific QA layer and AI model

        Args:
            marker_mode: "ids", "phrase" for text markers, or "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-64)
            word_map_formatted: Formatted word map string for word_index mode
            draft_map_formatted: Formatted paragraph/sentence ID map for ids mode
            input_images: Optional list of ImageData for vision-enabled QA evaluation.
                         Only passed if both the layer has include_input_images=True
                         and the QA model supports vision.
        """
        from models import QAModelConfig

        # Normalize model to QAModelConfig
        if isinstance(model, str):
            model_config = QAModelConfig(model=model)
        else:
            model_config = model

        # Determine if we should pass images to this evaluation
        # Images are only passed if:
        # 1. input_images is provided
        # 2. The layer has include_input_images=True
        # 3. The QA model supports vision
        images_for_eval = None
        if input_images and getattr(layer, 'include_input_images', False):
            try:
                model_info = config.get_model_info(model_config.model)
                model_capabilities = model_info.get("capabilities", [])
                model_supports_vision = "vision" in [
                    c.lower() for c in model_capabilities if isinstance(c, str)
                ]
                if model_supports_vision:
                    images_for_eval = input_images
                    if extra_verbose:
                        logger.info(
                            f"[QA VISION] Passing {len(input_images)} images to {model_config.model} "
                            f"for layer '{layer.name}'"
                        )
                else:
                    logger.warning(
                        f"[QA VISION] Layer '{layer.name}' has include_input_images=True but "
                        f"model {model_config.model} doesn't support vision. Evaluating without images."
                    )
            except Exception as exc:
                logger.warning(
                    f"[QA VISION] Could not determine vision support for {model_config.model}: {exc}. "
                    f"Evaluating without images."
                )

        try:
            return await self.qa_evaluator.evaluate_content(
                content=content,
                criteria=layer.criteria,
                model=model_config.model,
                layer_name=layer.name,
                min_score=layer.min_score,
                deal_breaker_criteria=layer.deal_breaker_criteria,
                concise_on_pass=layer.concise_on_pass,
                original_request=original_request,
                extra_verbose=extra_verbose,
                stream_callback=stream_callback,
                usage_callback=usage_callback,
                max_tokens=model_config.max_tokens,
                reasoning_effort=model_config.reasoning_effort,
                thinking_budget_tokens=model_config.thinking_budget_tokens,
                temperature=model_config.temperature,
                request_edit_info=request_edit_info,
                phase_logger=phase_logger,
                marker_mode=marker_mode,
                marker_length=marker_length,
                word_map_formatted=word_map_formatted,
                draft_map_formatted=draft_map_formatted,
                input_images=images_for_eval,
                edit_history=edit_history,
                model_alias_registry=model_alias_registry,
                layer=layer,
                bypass_engine=self.bypass_engine,
                session_id=session_id,
                project_id=project_id,
                tool_event_callback=tool_event_callback,
            )
        except Exception as e:
            logger.error(f"QA evaluation failed for layer {layer.name} with model {model_config.model}: {str(e)}")
            raise

    async def evaluate_all_layers(
        self,
        content: str,
        layers: List[QALayer],
        qa_models: List[str],
        input_images: Optional[List["ImageData"]] = None,
    ) -> Dict[str, Dict[str, QAEvaluation]]:
        """
        Evaluate content through all QA layers with all specified models.
        (Non-progress variant)

        Args:
            input_images: Optional list of ImageData for vision-enabled QA.
                         Passed to layers with include_input_images=True.
        """
        qa_results = await self.evaluate_all_layers_with_progress(
            content=content,
            layers=layers,
            qa_models=qa_models,
            input_images=input_images,
        )
        if isinstance(qa_results, dict) and "qa_results" in qa_results:
            return qa_results["qa_results"]
        return qa_results

    async def _evaluate_evidence_grounding(
        self,
        content: str,
        context: str,
        grounding_config: EvidenceGroundingConfig,
        progress_callback: Optional[Callable] = None,
        stream_callback: Optional[Callable] = None,
        usage_tracker: Optional[UsageTracker] = None,
        extra_verbose: bool = False,
        phase_logger: Optional["PhaseLogger"] = None,
    ) -> tuple:
        """
        Run evidence grounding check and convert result to QAEvaluation.

        This method is called as part of the QA layer execution flow when
        evidence grounding is enabled. It treats grounding as a special
        QA layer that uses logprobs instead of semantic evaluation.

        Args:
            content: Generated content to verify
            context: Original context/evidence (prompt + attachments)
            grounding_config: Evidence grounding configuration
            progress_callback: Progress update callback
            stream_callback: Streaming events callback
            usage_tracker: Token usage tracker
            extra_verbose: Enable detailed logging
            phase_logger: Phase logger for structured logging

        Returns:
            Tuple of (QAEvaluation, EvidenceGroundingResult)
            - QAEvaluation: For consensus integration with other QA layers
            - EvidenceGroundingResult: Full detailed result for reporting
        """
        if phase_logger:
            phase_logger.info("Running evidence grounding verification...")

        # Create usage callback if tracker provided
        usage_callback = None
        if usage_tracker:
            usage_callback = usage_tracker.create_callback(
                phase="evidence_grounding",
                role="grounding_verifier",
                operation="logprob_verification",
            )

        # Run the grounding check
        grounding_result = await self.grounding_engine.run_grounding_check(
            content=content,
            context=context,
            grounding_config=grounding_config,
            progress_callback=progress_callback,
            stream_callback=stream_callback,
            usage_callback=usage_callback,
            extra_verbose=extra_verbose,
        )

        # Convert to QAEvaluation for consensus integration
        # Score mapping:
        #   - passed=True -> 10.0
        #   - passed=False with warn -> 5.0 (contributes to average but not deal_breaker)
        #   - passed=False with deal_breaker/regenerate -> 0.0
        if grounding_result.passed:
            score = 10.0
            feedback = (
                f"Evidence grounding PASSED. Verified {grounding_result.claims_verified} claims, "
                f"{grounding_result.flagged_claims} flagged (threshold: {grounding_config.max_flagged_claims})."
            )
        elif grounding_result.triggered_action == "warn":
            score = 5.0
            feedback = (
                f"Evidence grounding WARNING. {grounding_result.flagged_claims} claims lack sufficient "
                f"evidence grounding (max gap: {grounding_result.max_budget_gap:.2f} bits)."
            )
        else:
            score = 0.0
            feedback = (
                f"Evidence grounding FAILED. {grounding_result.flagged_claims} claims flagged "
                f"(>= threshold {grounding_config.max_flagged_claims}). "
                f"Max budget gap: {grounding_result.max_budget_gap:.2f} bits."
            )

        is_deal_breaker = (
            not grounding_result.passed and
            grounding_result.triggered_action in ("deal_breaker", "regenerate")
        )

        qa_evaluation = QAEvaluation(
            model="evidence_grounding_logprobs",
            layer="Evidence Grounding",
            score=score,
            feedback=feedback,
            deal_breaker=is_deal_breaker,
            deal_breaker_reason=(
                f"{grounding_result.flagged_claims} claims lack evidence grounding"
                if is_deal_breaker else None
            ),
            passes_score=grounding_result.passed,
            metadata={
                "grounding_result": grounding_result.model_dump(),
                "prompt_facing_grounding_result": {
                    **{
                        key: value
                        for key, value in grounding_result.model_dump().items()
                        if key != "model_used"
                    },
                    "verifier": "GroundingVerifier",
                },
                "verification_time_ms": grounding_result.verification_time_ms,
                "flagged_claims_detail": [
                    {"idx": c.idx, "claim": c.claim, "budget_gap": c.budget_gap}
                    for c in grounding_result.claims if c.flagged
                ],
            }
        )

        if phase_logger:
            phase_logger.info(
                f"Evidence grounding: score={score}, passed={grounding_result.passed}, "
                f"flagged={grounding_result.flagged_claims}, deal_breaker={is_deal_breaker}"
            )

        return qa_evaluation, grounding_result

    async def evaluate_all_layers_with_progress(
        self,
        content: str,
        layers: List[QALayer],
        qa_models: List[Any],  # Can be List[str] or List[QAModelConfig]
        progress_callback: Optional[callable] = None,
        original_request: Optional[Any] = None,
        stream_callback: Optional[callable] = None,
        gran_sabio_engine: Optional[Any] = None,
        session_id: Optional[str] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        iteration: Optional[int] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        gran_sabio_stream_callback: Optional[callable] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        evidence_grounding_config: Optional[EvidenceGroundingConfig] = None,
        context_for_grounding: Optional[str] = None,
        content_for_bypass: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
        project_id: Optional[str] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Dict[str, QAEvaluation]]:
        """
        Evaluate content through all QA layers with detailed progress tracking.

        Args:
            marker_mode: "ids", "phrase" for text markers, or "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-64)
            word_map_formatted: Formatted word map string for word_index mode
            draft_map_formatted: Formatted paragraph/sentence ID map for ids mode
            input_images: Optional list of ImageData for vision-enabled QA.
                         Passed to layers with include_input_images=True.
            evidence_grounding_config: Optional evidence grounding configuration.
                         If enabled, grounding runs as a special layer with auto-order.
            context_for_grounding: Context string for grounding (prompt + attachments).
            content_for_bypass: Optional text for algorithmic bypass evaluation.
                         When provided, bypass engine uses this instead of `content`.
                         Used when `content` is JSON but bypass needs extracted text.
        """
        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        qa_model_names = [get_model_name(m) for m in qa_models]

        # Build unified execution plan including semantic layers and evidence grounding
        execution_plan: List[Dict[str, Any]] = []

        # Add semantic QA layers
        for layer in layers:
            execution_plan.append({
                "type": "semantic_layer",
                "layer": layer,
                "order": layer.order,
            })

        # Add evidence grounding if enabled
        grounding_result_holder: Dict[str, Any] = {"result": None}
        if evidence_grounding_config and evidence_grounding_config.enabled:
            grounding_order = get_effective_order(evidence_grounding_config)
            execution_plan.append({
                "type": "evidence_grounding",
                "config": evidence_grounding_config,
                "order": grounding_order,
            })
            logger.info(f"Evidence grounding enabled with order={grounding_order}")

        # Sort execution plan by order
        execution_plan.sort(key=lambda x: x["order"])

        escalations_this_evaluation = 0
        iteration_limit = getattr(original_request, 'gran_sabio_call_limit_per_iteration', -1)
        results: Dict[str, Dict[str, QAEvaluation]] = {}
        extra_verbose = getattr(original_request, 'extra_verbose', False) if original_request else False

        async def _abort_if_cancelled(message: str) -> None:
            if cancel_callback and await cancel_callback():
                if progress_callback:
                    await progress_callback(message)
                raise QAProcessCancelled()

        for plan_item in execution_plan:
            # Handle evidence grounding as special layer
            if plan_item["type"] == "evidence_grounding":
                await _abort_if_cancelled("Cancelled before evidence grounding.")

                if phase_logger:
                    phase_logger.info("Executing evidence grounding layer...")

                if progress_callback:
                    await progress_callback("Evidence grounding verification...")

                grounding_config = plan_item["config"]
                context = context_for_grounding or getattr(original_request, 'prompt', '')

                qa_eval, full_result = await self._evaluate_evidence_grounding(
                    content=content,
                    context=context,
                    grounding_config=grounding_config,
                    progress_callback=progress_callback,
                    stream_callback=stream_callback,
                    usage_tracker=usage_tracker,
                    extra_verbose=extra_verbose,
                    phase_logger=phase_logger,
                )

                grounding_result_holder["result"] = full_result

                # Store as layer result (single "model")
                results["Evidence Grounding"] = {
                    "evidence_grounding_logprobs": qa_eval
                }

                # Handle deal-breaker
                if qa_eval.deal_breaker:
                    deal_breaker_consensus = {
                        "immediate_stop": True,
                        "deal_breaker_count": 1,
                        "total_evaluated": 1,
                        "total_models": 1,
                        "deal_breaker_details": [{
                            "model": "evidence_grounding",
                            "reason": qa_eval.deal_breaker_reason
                        }],
                        "majority_threshold": 0.5
                    }

                    if grounding_config.on_flag == "deal_breaker":
                        logger.warning("Evidence grounding deal-breaker triggered. Stopping evaluation.")
                        if progress_callback:
                            await progress_callback("Evidence grounding FAILED. Forcing iteration.")
                        return self._create_iteration_stop_result(results, deal_breaker_consensus)

                    elif grounding_config.on_flag == "regenerate":
                        logger.warning("Evidence grounding failed with regenerate action. Stopping evaluation.")
                        if progress_callback:
                            await progress_callback("Evidence grounding FAILED. Triggering regeneration.")
                        return self._create_iteration_stop_result(results, deal_breaker_consensus)

                if progress_callback:
                    status = "PASSED" if full_result.passed else "WARNING"
                    await progress_callback(f"Evidence grounding: {status} ({full_result.flagged_claims} claims flagged)")

                continue  # Move to next item in execution plan

            # Handle semantic QA layer - delegate to extracted function
            layer = plan_item["layer"]

            # Evaluate layer using extracted function
            layer_results, majority_deal_breaker = await self._evaluate_single_semantic_layer(
                content=content,
                layer=layer,
                qa_models=qa_models,
                qa_model_names=qa_model_names,
                progress_callback=progress_callback,
                original_request=original_request,
                stream_callback=stream_callback,
                session_id=session_id,
                cancel_callback=cancel_callback,
                usage_tracker=usage_tracker,
                iteration=iteration,
                phase_logger=phase_logger,
                marker_mode=marker_mode,
                marker_length=marker_length,
                word_map_formatted=word_map_formatted,
                draft_map_formatted=draft_map_formatted,
                input_images=input_images,
                content_for_bypass=content_for_bypass,
                model_alias_registry=model_alias_registry,
                project_id=project_id,
                tool_event_callback=tool_event_callback,
            )

            # If majority deal-breaker detected, stop immediately
            if majority_deal_breaker:
                results[layer.name] = layer_results
                return self._create_iteration_stop_result(results, majority_deal_breaker)

            results[layer.name] = layer_results

            # ===== Minority deal-breakers & Gran Sabio logic =====
            deal_breakers = semantic_deal_breakers(layer_results)

            if deal_breakers:
                consensus_counts = build_qa_counts(layer_results, len(qa_model_names))
                total_models = consensus_counts["valid"]
                deal_breaker_count = len(deal_breakers)

                is_majority = (
                    total_models >= consensus_counts["required_valid"]
                    and deal_breaker_count >= consensus_counts["required_majority"]
                )
                is_tie = total_models > 0 and (total_models % 2 == 0) and (deal_breaker_count * 2 == total_models)
                is_minority = total_models > 0 and deal_breaker_count < (total_models / 2)

                if is_majority:
                    logger.warning(
                        f"Majority deal-breaker consensus in {layer.name}: "
                        f"{deal_breaker_count}/{total_models} models"
                    )
                    if progress_callback:
                        await progress_callback(
                            f"Majority deal-breaker detected in {layer.name}. Forcing iteration."
                        )
                    return self._create_iteration_stop_result(results, {
                        "immediate_stop": True,
                        "deal_breaker_count": deal_breaker_count,
                        "total_evaluated": total_models,
                        "total_models": len(qa_model_names),
                        "deal_breaker_details": [
                            {
                                "model": eval.model,
                                "evaluator": get_evaluator_alias(eval, fallback=eval.model),
                                "reason": eval.deal_breaker_reason or eval.reason or "",
                            }
                            for eval in deal_breakers
                        ],
                        "majority_threshold": total_models / 2,
                        **consensus_counts,
                    })

                elif is_tie or is_minority:
                    # (Gran Sabio escalation block preserved)
                    if iteration_limit == -1:
                        can_escalate = True
                    else:
                        can_escalate = escalations_this_evaluation < iteration_limit

                    if can_escalate and gran_sabio_engine:
                        escalations_this_evaluation += 1
                        escalation_type = "50_50_tie" if is_tie else "minority_deal_breaker"

                        separator = "=" * 80
                        logger.info(separator)
                        logger.info(f"ESCALATING TO GRAN SABIO - {escalation_type.upper().replace('_', ' ')}")
                        logger.info(f"Layer: {layer.name}")
                        logger.info(f"Deal-breakers: {deal_breaker_count}/{total_models}")
                        logger.info(f"Escalations this evaluation: {escalations_this_evaluation}/{iteration_limit if iteration_limit != -1 else 'unlimited'}")
                        logger.info(separator)

                        if progress_callback:
                            await progress_callback(
                                f"Deal-breaker detected in {layer.name} ({deal_breaker_count}/{total_models}). Escalating to Gran Sabio for review..."
                            )

                        await _abort_if_cancelled(
                            f"🛑 Cancelled before Gran Sabio escalation for layer {layer.name}."
                        )

                        from deal_breaker_tracker import get_tracker
                        tracker = get_tracker()

                        first_deal_breaker = deal_breakers[0]
                        escalation_id = tracker.record_escalation(
                            session_id=session_id or "unknown",
                            iteration=getattr(original_request, '_current_iteration', 0),
                            layer_name=layer.name,
                            trigger_type=escalation_type,
                            triggering_model=first_deal_breaker.model,
                            deal_breaker_reason=first_deal_breaker.deal_breaker_reason or first_deal_breaker.reason or "",
                            total_models=total_models,
                            deal_breaker_count=deal_breaker_count,
                            gran_sabio_model=getattr(original_request, "gran_sabio_model", None),
                            deal_breaker_evaluations=deal_breakers,
                        )

                        minority_data = {
                            "has_minority_deal_breakers": True,
                            "deal_breaker_count": deal_breaker_count,
                            "total_evaluations": total_models,
                            "total_models_configured": len(qa_model_names),
                            "qa_quorum": consensus_counts,
                            "details": [
                                {
                                    "layer": layer.name,
                                    "model": eval.model,
                                    "evaluator": get_evaluator_alias(eval, fallback=eval.model),
                                    "reason": eval.deal_breaker_reason or eval.reason or "",
                                    "score_given": getattr(eval, "score", None),
                                    "layer_criteria": getattr(layer, "criteria", None),
                                    "layer_min_score": getattr(layer, "min_score", None),
                                    "layer_deal_breaker_criteria": getattr(layer, "deal_breaker_criteria", None),
                                }
                                for eval in deal_breakers
                            ],
                            "qa_configuration": {
                                "layer_name": layer.name,
                                "description": getattr(layer, "description", None),
                                "criteria": getattr(layer, "criteria", None),
                                "min_score": getattr(layer, "min_score", None),
                                "deal_breaker_criteria": getattr(layer, "deal_breaker_criteria", None),
                                "concise_on_pass": getattr(layer, "concise_on_pass", None),
                                "order": getattr(layer, "order", None),
                                "is_deal_breaker": getattr(layer, "is_deal_breaker", None),
                                "is_mandatory": getattr(layer, "is_mandatory", None),
                            },
                            "summary": f"{deal_breaker_count} deal-breakers from {total_models} evaluations in {layer.name}"
                        }

                        escalation_completed = False
                        try:
                            gs_result = await gran_sabio_engine.review_minority_deal_breakers(
                                session_id=session_id or "unknown",
                                content=content,
                                minority_deal_breakers=minority_data,
                                original_request=original_request,
                                stream_callback=gran_sabio_stream_callback,
                                cancel_callback=cancel_callback
                            )

                            if getattr(gs_result, "error", None):
                                error_reason = gs_result.reason or gs_result.error
                                tracker.complete_escalation(
                                    escalation_id=escalation_id,
                                    decision="error",
                                    reasoning=error_reason,
                                    was_real=None
                                )
                                escalation_completed = True
                                logger.error("Gran Sabio review failed for session %s: %s", session_id or "unknown", gs_result.error)
                                if progress_callback:
                                    await progress_callback(f"Gran Sabio review failed: {gs_result.error}")
                                raise GranSabioInvocationError(gs_result.error)

                            tracker.complete_escalation(
                                escalation_id=escalation_id,
                                decision="real" if not gs_result.approved else "false_positive",
                                reasoning=gs_result.reason,
                                was_real=not gs_result.approved
                            )
                            escalation_completed = True

                            if not gs_result.approved:
                                logger.warning(
                                    f"Gran Sabio confirmed deal-breaker in {layer.name} as REAL. Forcing iteration."
                                )
                                if progress_callback:
                                    await progress_callback(
                                        f"Gran Sabio confirmed deal-breaker in {layer.name}. Forcing iteration."
                                    )
                                return self._create_iteration_stop_result(results, {
                                    "immediate_stop": True,
                                    "deal_breaker_count": deal_breaker_count,
                                    "total_evaluated": total_models,
                                    "total_models": total_models,
                                    "deal_breaker_details": minority_data["details"],
                                    "majority_threshold": total_models / 2,
                                    "gran_sabio_confirmed": True
                                })
                            else:
                                logger.info(
                                    f"Gran Sabio determined deal-breaker in {layer.name} as FALSE POSITIVE. Continuing evaluation."
                                )
                                if progress_callback:
                                    await progress_callback(
                                        f"Gran Sabio: FALSE POSITIVE in {layer.name}. Continuing."
                                    )

                                for eval in deal_breakers:
                                    original_reason = eval.deal_breaker_reason or eval.reason or ""
                                    override = apply_gran_sabio_false_positive_override(
                                        eval,
                                        final_score=getattr(gs_result, "final_score", None),
                                        layer_min_score=getattr(layer, "min_score", 0.0),
                                        original_reason=original_reason,
                                    )
                                    logger.debug(
                                        "Gran Sabio override updated %s score from %s to %s (raw=%s, clamped_to_min=%s)",
                                        eval.model,
                                        override["previous_score"],
                                        override["effective_score"],
                                        override["raw_score"],
                                        override["clamped_to_min"],
                                    )
                                    if progress_callback:
                                        await progress_callback(
                                            f"Gran Sabio override: {eval.model} score "
                                            f"{override['previous_score']} -> {override['effective_score']}"
                                        )

                                # Check if Gran Sabio provided modified content
                                gs_modified_content = getattr(gs_result, "final_content", None)
                                gs_modifications_made = getattr(gs_result, "modifications_made", False)

                                if gs_modifications_made and gs_modified_content and gs_modified_content.strip():
                                    logger.info(
                                        f"Gran Sabio approved with modifications in {layer.name}. Content will be re-evaluated."
                                    )
                                    if progress_callback:
                                        await progress_callback(f"Gran Sabio approved {layer.name} with modifications. Re-evaluating...")

                                    # Return special result to trigger re-evaluation with modified content
                                    results[layer.name] = layer_results
                                    return self._create_gran_sabio_modified_result(
                                        partial_results=results,
                                        modified_content=gs_modified_content,
                                        reason=gs_result.reason,
                                        score=gs_result.final_score
                                    )
                                else:
                                    if progress_callback:
                                        await progress_callback(f"Gran Sabio reviewed {layer.name}: False positive. Continuing...")

                        except GranSabioProcessCancelled:
                            if not escalation_completed:
                                tracker.complete_escalation(
                                    escalation_id=escalation_id,
                                    decision="cancelled",
                                    reasoning="Cancelled by user request",
                                    was_real=None
                                )
                            raise QAProcessCancelled()
                        except Exception as e:
                            logger.error(f"Gran Sabio escalation failed: {e}")
                            if not escalation_completed:
                                tracker.complete_escalation(
                                    escalation_id=escalation_id,
                                    decision="error",
                                    reasoning=str(e),
                                    was_real=None
                                )
                            if progress_callback:
                                await progress_callback(f"Gran Sabio review failed: {str(e)[:100]}")
                            raise
                    else:
                        if not can_escalate:
                            limit_text = "unlimited" if iteration_limit == -1 else iteration_limit
                            logger.warning(
                                f"Gran Sabio escalation limit reached for this evaluation ({escalations_this_evaluation}/{limit_text}). Continuing."
                            )
                            if progress_callback:
                                await progress_callback(
                                    f"Escalation limit reached ({escalations_this_evaluation}/{limit_text}). Continuing evaluation."
                                )

            await _abort_if_cancelled(f"🛑 Cancelled before moving to next QA layer after {layer.name}.")
            # ===== END =====

        return results
    
    async def evaluate_content_comprehensive(
        self,
        content: str,
        layers: List[QALayer],
        qa_models: List[Any],
        progress_callback: Optional[callable] = None,
        original_request: Optional[Any] = None,
        stream_callback: Optional[callable] = None,
        gran_sabio_engine: Optional[Any] = None,
        session_id: Optional[str] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        iteration: Optional[int] = None,
        gran_sabio_stream_callback: Optional[callable] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        evidence_grounding_config: Optional[EvidenceGroundingConfig] = None,
        context_for_grounding: Optional[str] = None,
        content_for_bypass: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
        project_id: Optional[str] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive content evaluation with progress tracking

        Args:
            marker_mode: "ids", "phrase" for text markers, or "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-64)
            word_map_formatted: Formatted word map string for word_index mode
            draft_map_formatted: Formatted paragraph/sentence ID map for ids mode
            input_images: Optional list of ImageData for vision-enabled QA.
                         Passed to layers with include_input_images=True.
            evidence_grounding_config: Optional evidence grounding configuration.
                         If enabled, grounding runs as a special layer with auto-order.
            context_for_grounding: Context string for grounding (prompt + attachments).
            content_for_bypass: Optional text for algorithmic bypass evaluation.
                         Propagated to evaluate_all_layers_with_progress.
        """
        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        qa_model_names = [get_model_name(m) for m in qa_models]

        start_time = datetime.now()
        qa_results = await self.evaluate_all_layers_with_progress(
            content,
            layers,
            qa_models,
            progress_callback,
            original_request,
            stream_callback,
            gran_sabio_engine,
            session_id,
            cancel_callback,
            usage_tracker=usage_tracker,
            iteration=iteration,
            gran_sabio_stream_callback=gran_sabio_stream_callback,
            marker_mode=marker_mode,
            marker_length=marker_length,
            word_map_formatted=word_map_formatted,
            draft_map_formatted=draft_map_formatted,
            input_images=input_images,
            evidence_grounding_config=evidence_grounding_config,
            context_for_grounding=context_for_grounding,
            content_for_bypass=content_for_bypass,
            model_alias_registry=model_alias_registry,
            project_id=project_id,
            tool_event_callback=tool_event_callback,
        )

        if isinstance(qa_results, dict) and qa_results.get("summary", {}).get("force_iteration"):
            evidence_grounding_result = None
            partial_results = qa_results.get("qa_results", {})
            if "Evidence Grounding" in partial_results:
                eg_eval = partial_results["Evidence Grounding"].get("evidence_grounding_logprobs")
                if eg_eval and eg_eval.metadata:
                    evidence_grounding_result = eg_eval.metadata.get("grounding_result")

            return {
                **qa_results,
                "evidence_grounding": evidence_grounding_result,
            }

        summary = self._calculate_summary(qa_results, layers)
        critical_issues = self._identify_critical_issues(qa_results)
        layer_stats = self._calculate_layer_statistics(qa_results, layers)
        model_stats = self._calculate_model_statistics(qa_results, qa_model_names)

        # Extract full grounding result from metadata if present
        evidence_grounding_result = None
        if "Evidence Grounding" in qa_results:
            eg_eval = qa_results["Evidence Grounding"].get("evidence_grounding_logprobs")
            if eg_eval and eg_eval.metadata:
                evidence_grounding_result = eg_eval.metadata.get("grounding_result")

        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()

        if progress_callback:
            await progress_callback(f"Evaluation completed in {evaluation_time:.2f} seconds")

        return {
            "qa_results": qa_results,
            "summary": summary,
            "critical_issues": critical_issues,
            "layer_statistics": layer_stats,
            "model_statistics": model_stats,
            "evidence_grounding": evidence_grounding_result,
            "evaluation_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": evaluation_time,
                "total_evaluations": len(layers) * len(qa_models)
            }
        }

    def _build_fast_global_verification_layer(
        self,
        source_layers: List[QALayer],
        original_request: Optional[Any],
    ) -> QALayer:
        """Build one synthetic global QA layer from the configured semantic layers."""

        ordered_layers = sorted(source_layers, key=lambda layer: getattr(layer, "order", 0))
        source_blocks = []
        deal_breaker_blocks = []

        for index, layer in enumerate(ordered_layers, start=1):
            deal_breaker_criteria = getattr(layer, "deal_breaker_criteria", None)
            if deal_breaker_criteria:
                deal_breaker_blocks.append(f"- {layer.name}: {deal_breaker_criteria}")

            source_blocks.append(
                "\n".join(
                    [
                        f"Source layer {index}: {layer.name}",
                        f"Description: {getattr(layer, 'description', '')}",
                        f"Criteria: {getattr(layer, 'criteria', '')}",
                        f"Minimum score: {getattr(layer, 'min_score', None)}",
                        f"Mandatory: {getattr(layer, 'is_mandatory', False)}",
                        f"Deal-breaker criteria: {deal_breaker_criteria or 'None'}",
                    ]
                )
            )

        min_global_score = float(getattr(original_request, "min_global_score", 8.0) or 8.0)
        criteria = f"""
Perform a read-only final global QA verification of the content.

This is a fast global verification pass, not a per-layer re-evaluation. Check
whether the final content satisfies the whole semantic QA contract in one global
review. Use the source layer definitions below as the contract.

Approval threshold:
- Give one global score from 0 to 10.
- The content passes this synthetic global review only when the score is at
  least {min_global_score:.2f}.
- Do not request or propose edits.
- Do not claim that each source layer passed individually. Report any concerns
  by mapping them back to the relevant source layer names.

Source QA layers:
{chr(10).join(source_blocks)}
""".strip()

        deal_breaker_criteria = None
        if deal_breaker_blocks:
            deal_breaker_criteria = (
                "Any issue that truly satisfies one of these source deal-breaker criteria:\n"
                + "\n".join(deal_breaker_blocks)
            )

        synthetic_layer = QALayer(
            name="Final Global QA Verification",
            description="Fast global read-only verification across all semantic QA criteria",
            criteria=criteria,
            min_score=min_global_score,
            is_mandatory=True,
            deal_breaker_criteria=deal_breaker_criteria,
            concise_on_pass=True,
            order=max((getattr(layer, "order", 0) for layer in ordered_layers), default=0) + 1,
            include_input_images=any(getattr(layer, "include_input_images", False) for layer in ordered_layers),
        )
        estimated_tokens = self._estimate_fast_global_tokens(
            "\n".join(
                value
                for value in [synthetic_layer.criteria, synthetic_layer.deal_breaker_criteria]
                if value
            )
        )
        token_limit = self._fast_global_token_limit()
        if estimated_tokens > token_limit:
            raise ValueError(
                "fast_global final verification synthetic criteria exceeds "
                f"the safe token budget ({estimated_tokens} estimated tokens > {token_limit}). "
                "Use full_parallel or full_sequential for this QA contract."
            )

        return synthetic_layer

    def _resolve_final_verification_layers(
        self,
        layers: List[QALayer],
        strategy: str,
        original_request: Optional[Any],
    ) -> Tuple[List[QALayer], List[QALayer], str]:
        """Resolve real/synthetic layers and approval contract for final verification."""

        if strategy != "fast_global":
            return list(layers), list(layers), "per_layer"

        deterministic_layers: List[QALayer] = []
        semantic_layers: List[QALayer] = []
        for layer in layers:
            if self.bypass_engine.can_bypass_layer(layer, original_request):
                deterministic_layers.append(layer)
            else:
                semantic_layers.append(layer)

        final_layers = list(deterministic_layers)
        if semantic_layers:
            final_layers.append(
                self._build_fast_global_verification_layer(semantic_layers, original_request)
            )

        return final_layers, semantic_layers, "fast_global"

    async def evaluate_final_verification(
        self,
        content: str,
        layers: List[QALayer],
        qa_models: List[Any],
        *,
        strategy: str,
        progress_callback: Optional[callable] = None,
        original_request: Optional[Any] = None,
        stream_callback: Optional[callable] = None,
        session_id: Optional[str] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        iteration: Optional[int] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        evidence_grounding_config: Optional[EvidenceGroundingConfig] = None,
        context_for_grounding: Optional[str] = None,
        content_for_bypass: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
    ) -> Dict[str, Any]:
        """
        Run a read-only final QA verification pass.

        Unlike the normal comprehensive path, this method collects all layer
        results before approval. It never asks QA for smart-edit ranges because
        callers pass a request clone with smart_editing_mode='never'.
        """

        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        async def _abort_if_cancelled(message: str) -> None:
            if cancel_callback and await cancel_callback():
                if progress_callback:
                    await progress_callback(message)
                raise QAProcessCancelled()

        strategy = strategy if strategy in {"full_parallel", "full_sequential", "fast_global"} else "full_parallel"
        qa_model_names = [get_model_name(model) for model in qa_models]
        evaluation_layers, source_semantic_layers, approval_contract = self._resolve_final_verification_layers(
            layers,
            strategy,
            original_request,
        )
        ordered_layers = sorted(evaluation_layers, key=lambda layer: getattr(layer, "order", 0))
        start_time = datetime.now()

        async def evaluate_grounding() -> Tuple[str, Dict[str, QAEvaluation], Optional[EvidenceGroundingResult]]:
            await _abort_if_cancelled("Cancelled before final evidence grounding.")
            grounding_config = evidence_grounding_config
            if not grounding_config or not grounding_config.enabled:
                return "Evidence Grounding", {}, None
            if progress_callback:
                await progress_callback("Final verification: Evidence Grounding")
            qa_eval, full_result = await self._evaluate_evidence_grounding(
                content=content,
                context=context_for_grounding or getattr(original_request, "prompt", ""),
                grounding_config=grounding_config,
                progress_callback=progress_callback,
                stream_callback=stream_callback,
                usage_tracker=usage_tracker,
                extra_verbose=getattr(original_request, "extra_verbose", False) if original_request else False,
                phase_logger=phase_logger,
            )
            return "Evidence Grounding", {"evidence_grounding_logprobs": qa_eval}, full_result

        async def evaluate_layer(layer: QALayer) -> Tuple[str, Dict[str, QAEvaluation], Optional[Any]]:
            await _abort_if_cancelled(f"Cancelled before final verification layer {layer.name}.")
            if progress_callback:
                await progress_callback(f"Final verification: {layer.name}")
            layer_results, majority_deal_breaker = await self._evaluate_single_semantic_layer(
                content=content,
                layer=layer,
                qa_models=qa_models,
                qa_model_names=qa_model_names,
                progress_callback=progress_callback,
                original_request=original_request,
                stream_callback=stream_callback,
                session_id=session_id,
                cancel_callback=cancel_callback,
                usage_tracker=usage_tracker,
                iteration=iteration,
                phase_logger=phase_logger,
                marker_mode=marker_mode,
                marker_length=marker_length,
                word_map_formatted=word_map_formatted,
                draft_map_formatted=draft_map_formatted,
                input_images=input_images,
                content_for_bypass=content_for_bypass,
                model_alias_registry=model_alias_registry,
            )
            return layer.name, layer_results, majority_deal_breaker

        execution_items: List[Tuple[str, Any]] = [("layer", layer) for layer in ordered_layers]
        if evidence_grounding_config and evidence_grounding_config.enabled:
            execution_items.append(("evidence_grounding", None))

        results: Dict[str, Dict[str, QAEvaluation]] = {}
        majority_deal_breakers: List[Dict[str, Any]] = []
        evidence_grounding_result: Optional[EvidenceGroundingResult] = None

        if strategy == "full_parallel":
            max_requests = max(1, int(getattr(config, "MAX_CONCURRENT_REQUESTS", 10) or 10))
            models_per_layer = max(1, len(qa_model_names))
            max_layer_concurrency = max(1, max_requests // models_per_layer)
            semaphore = asyncio.Semaphore(max_layer_concurrency)

            async def run_item(item: Tuple[str, Any]) -> Tuple[str, Dict[str, QAEvaluation], Optional[Any]]:
                async with semaphore:
                    item_type, payload = item
                    if item_type == "evidence_grounding":
                        return await evaluate_grounding()
                    return await evaluate_layer(payload)

            item_results = await asyncio.gather(*(run_item(item) for item in execution_items))
        else:
            item_results = []
            for item_type, payload in execution_items:
                if item_type == "evidence_grounding":
                    item_results.append(await evaluate_grounding())
                else:
                    item_results.append(await evaluate_layer(payload))

        for name, layer_results, extra in item_results:
            if layer_results:
                results[name] = layer_results
            if name == "Evidence Grounding":
                evidence_grounding_result = extra
            elif extra:
                majority_deal_breakers.append({"layer": name, "info": extra})

        summary = self._calculate_summary(results, evaluation_layers)
        layers_summary: Dict[str, Dict[str, Any]] = {}
        for layer in evaluation_layers:
            layer_results = results.get(layer.name, {})
            scores = [
                evaluation.score
                for evaluation in layer_results.values()
                if getattr(evaluation, "score", None) is not None
            ]
            average_score = sum(scores) / len(scores) if scores else 0.0
            layers_summary[layer.name] = {
                "passed": bool(scores) and average_score >= layer.min_score,
                "score": average_score,
                "min_score": layer.min_score,
                "deal_breaker": any(
                    bool(getattr(evaluation, "deal_breaker", False))
                    for evaluation in layer_results.values()
                ),
                "order": getattr(layer, "order", 0),
            }

        if "Evidence Grounding" in results:
            grounding_eval = results["Evidence Grounding"].get("evidence_grounding_logprobs")
            layers_summary["Evidence Grounding"] = {
                "passed": bool(getattr(evidence_grounding_result, "passed", True)) if evidence_grounding_result else True,
                "score": getattr(grounding_eval, "score", 0.0) if grounding_eval else 0.0,
                "min_score": None,
                "deal_breaker": bool(getattr(grounding_eval, "deal_breaker", False)) if grounding_eval else False,
                "order": get_effective_order(evidence_grounding_config) if evidence_grounding_config else 0,
            }

        summary.update(
            {
                "final_verification": True,
                "strategy": strategy,
                "approval_contract": approval_contract,
                "layers_summary": layers_summary,
                "force_iteration": bool(majority_deal_breakers)
                or bool(
                    evidence_grounding_result
                    and not evidence_grounding_result.passed
                    and evidence_grounding_result.triggered_action in ("deal_breaker", "regenerate")
                ),
                "majority_deal_breakers": majority_deal_breakers,
            }
        )

        end_time = datetime.now()
        return {
            "qa_results": results,
            "summary": summary,
            "critical_issues": self._identify_critical_issues(results),
            "layer_statistics": self._calculate_layer_statistics(results, evaluation_layers),
            "model_statistics": self._calculate_model_statistics(results, qa_model_names),
            "evidence_grounding": evidence_grounding_result,
            "evaluated_layers": evaluation_layers,
            "source_semantic_layers": source_semantic_layers,
            "approval_contract": approval_contract,
            "evaluation_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - start_time).total_seconds(),
                "total_evaluations": sum(len(layer_results) for layer_results in results.values()),
            },
        }
    
    def _calculate_summary(self, qa_results: Dict, layers: List[QALayer]) -> Dict[str, Any]:
        """Calculate summary statistics from QA results"""
        all_scores = []
        deal_breakers = []
        
        for layer_name, layer_results in qa_results.items():
            for model, evaluation in layer_results.items():
                if evaluation.score is not None:
                    all_scores.append(evaluation.score)
                if evaluation.deal_breaker:
                    deal_breakers.append({
                        "layer": layer_name,
                        "model": model,
                        "reason": evaluation.reason
                    })
        
        return {
            "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "min_score": min(all_scores) if all_scores else 0.0,
            "max_score": max(all_scores) if all_scores else 0.0,
            "total_evaluations": len(all_scores),
            "deal_breakers_count": len(deal_breakers),
            "deal_breakers": deal_breakers,
            "has_deal_breakers": len(deal_breakers) > 0
        }
    
    def _identify_critical_issues(self, qa_results: Dict) -> List[Dict[str, Any]]:
        """Identify critical issues from QA results"""
        critical_issues = []
        
        for layer_name, layer_results in qa_results.items():
            invalid_models = [model for model, evaluation in layer_results.items() if evaluation.score is None]
            if invalid_models:
                critical_issues.append({
                    "type": "missing_model_scores",
                    "layer": layer_name,
                    "models": invalid_models,
                    "description": f"QA models with invalid JSON responses: {', '.join(invalid_models)}"
                })

            layer_scores = [evaluation.score for evaluation in layer_results.values() if evaluation.score is not None]
            if not layer_scores:
                continue

            layer_avg = sum(layer_scores) / len(layer_scores)
            
            if layer_avg < 4.0:
                critical_issues.append({
                    "type": "low_layer_score",
                    "layer": layer_name,
                    "average_score": layer_avg,
                    "description": f"Layer {layer_name} average score is critically low: {layer_avg:.2f}"
                })
            
            if len(layer_scores) > 1:
                score_range = max(layer_scores) - min(layer_scores)
                if score_range > 5.0:
                    critical_issues.append({
                        "type": "model_disagreement",
                        "layer": layer_name,
                        "score_range": score_range,
                        "description": f"Models disagree significantly in {layer_name}: range of {score_range:.2f} points"
                    })
            
            deal_breaker_models = [model for model, eval in layer_results.items() if eval.deal_breaker]
            if deal_breaker_models:
                critical_issues.append({
                    "type": "deal_breaker",
                    "layer": layer_name,
                    "models": deal_breaker_models,
                    "description": f"Deal-breaker detected in {layer_name} by: {', '.join(deal_breaker_models)}"
                })
        
        return critical_issues
    
    def _calculate_layer_statistics(self, qa_results: Dict, layers: List[QALayer]) -> Dict[str, Dict]:
        """Calculate statistics for each layer"""
        layer_stats: Dict[str, Dict[str, Any]] = {}
        
        for layer in layers:
            layer_name = layer.name
            if layer_name in qa_results:
                layer_results = qa_results[layer_name]
                scores = [evaluation.score for evaluation in layer_results.values() if evaluation.score is not None]
                invalid_models = [model for model, evaluation in layer_results.items() if evaluation.score is None]
                average_score = sum(scores) / len(scores) if scores else None
                
                layer_stats[layer_name] = {
                    "average_score": average_score,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None,
                    "required_score": layer.min_score,
                    "passes_requirement": (average_score is not None) and (average_score >= layer.min_score),
                    "model_count": len(layer_results),
                    "valid_model_count": len(scores),
                    "invalid_models": invalid_models,
                    "deal_breaker_count": sum(1 for eval in layer_results.values() if eval.deal_breaker),
                    "is_deal_breaker_layer": layer.is_deal_breaker
                }
        
        return layer_stats
    
    def _calculate_model_statistics(self, qa_results: Dict, qa_models: List[str]) -> Dict[str, Dict]:
        """Calculate statistics for each model"""
        model_stats: Dict[str, Dict[str, Any]] = {}
        
        for model in qa_models:
            model_scores = []
            model_deal_breakers = 0
            invalid_layers = []
            
            for layer_name, layer_results in qa_results.items():
                if model in layer_results:
                    evaluation = layer_results[model]
                    if evaluation.score is not None:
                        model_scores.append(evaluation.score)
                    else:
                        invalid_layers.append(layer_name)
                    if evaluation.deal_breaker:
                        model_deal_breakers += 1
            
            evaluation_count = len(model_scores)
            reliability = (
                (evaluation_count - model_deal_breakers) / evaluation_count
                if evaluation_count else None
            )

            if model_scores or invalid_layers:
                model_stats[model] = {
                    "average_score": sum(model_scores) / len(model_scores) if model_scores else None,
                    "min_score": min(model_scores) if model_scores else None,
                    "max_score": max(model_scores) if model_scores else None,
                    "evaluation_count": evaluation_count,
                    "deal_breaker_count": model_deal_breakers,
                    "invalid_layers": invalid_layers,
                    "reliability": reliability
                }
        
        return model_stats
    
    async def validate_layers(self, layers: List[QALayer]) -> Dict[str, Any]:
        """
        Validate QA layer configuration
        """
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        if not layers:
            validation_results["is_valid"] = False
            validation_results["errors"].append("No QA layers provided")
            return validation_results
        
        layer_names = [layer.name for layer in layers]
        if len(layer_names) != len(set(layer_names)):
            validation_results["is_valid"] = False
            validation_results["errors"].append("Duplicate layer names found")
        
        layer_orders = [layer.order for layer in layers]
        if len(layer_orders) != len(set(layer_orders)):
            validation_results["warnings"].append("Duplicate layer orders found - execution order may be unpredictable")
        
        for layer in layers:
            if layer.min_score < 0 or layer.min_score > 10:
                validation_results["errors"].append(f"Layer {layer.name} has invalid min_score: {layer.min_score}")
                validation_results["is_valid"] = False
        
        deal_breaker_layers = [layer for layer in layers if layer.is_deal_breaker]
        if not deal_breaker_layers:
            validation_results["warnings"].append("No deal-breaker layers configured - content quality may be inconsistent")
        
        return validation_results
    
    def _resolve_qa_scheduler_policy(self, original_request: Optional[Any], configured_count: int) -> QASchedulerPolicy:
        """Resolve public request knobs into the internal scheduler policy."""

        max_concurrency = max(1, int(getattr(config, "MAX_CONCURRENT_REQUESTS", 10) or 10))
        timeout_retries = max(0, int(getattr(config, "MAX_QA_TIMEOUT_RETRIES", 2) or 0))
        min_valid_models = (
            getattr(original_request, "min_valid_qa_models", None)
            if original_request
            else None
        )
        if not isinstance(min_valid_models, int):
            min_valid_models = None
        min_valid_ratio = (
            getattr(original_request, "min_valid_qa_model_ratio", None)
            if original_request
            else None
        )
        if not isinstance(min_valid_ratio, (int, float)):
            min_valid_ratio = None
        execution_mode = getattr(original_request, "qa_execution_mode", "auto") if original_request else "auto"
        if execution_mode not in {"auto", "sequential", "parallel", "progressive_quorum"}:
            execution_mode = "auto"
        unavailable_policy = (
            getattr(original_request, "on_qa_model_unavailable", "skip_if_quorum")
            if original_request
            else "skip_if_quorum"
        )
        if unavailable_policy not in {"fail", "skip_if_quorum", "skip"}:
            unavailable_policy = "skip_if_quorum"
        timeout_policy = (
            getattr(original_request, "on_qa_timeout", "retry_then_skip_if_quorum")
            if original_request
            else "retry_then_skip_if_quorum"
        )
        if timeout_policy not in {
            "fail",
            "skip_as_technical_failure",
            "retry_then_fail",
            "retry_then_skip_if_quorum",
        }:
            timeout_policy = "retry_then_skip_if_quorum"
        return QASchedulerPolicy(
            execution_mode=execution_mode,
            on_model_unavailable=unavailable_policy,
            on_timeout=timeout_policy,
            min_valid_models=min_valid_models,
            min_valid_model_ratio=min_valid_ratio,
            max_concurrency=min(max_concurrency, max(1, configured_count)),
            timeout_retries=timeout_retries,
        )

    async def _evaluate_ai_layer_with_scheduler(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[Any],
        qa_model_names: List[str],
        progress_callback: Optional[Callable] = None,
        original_request: Optional[Any] = None,
        stream_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        cancel_callback: Optional[Callable[[], Awaitable[bool]]] = None,
        usage_tracker: Optional[UsageTracker] = None,
        iteration: Optional[int] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        edit_history: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
        project_id: Optional[str] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> QASchedulerResult:
        """Evaluate a semantic QA layer through the shared scheduler primitive."""

        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        async def _abort_if_cancelled(message: str) -> None:
            if cancel_callback and await cancel_callback():
                if progress_callback:
                    await progress_callback(message)
                raise QAProcessCancelled()

        slots: List[QASchedulerSlot] = []
        for index, model in enumerate(qa_models):
            model_name = get_model_name(model)
            if isinstance(model, QAModelConfig):
                timeout = calculate_qa_timeout_for_model(
                    model.model,
                    model.reasoning_effort,
                    model.thinking_budget_tokens,
                )
            else:
                timeout = calculate_qa_timeout_for_model(model)

            slot_id = model_alias_registry.qa_slot_id(index) if model_alias_registry else None
            slot_meta = model_alias_registry.slots.get(slot_id) if model_alias_registry and slot_id else None
            slots.append(
                QASchedulerSlot(
                    index=index,
                    model=model,
                    model_name=model_name,
                    result_key=_qa_model_result_key(model_name, index, qa_model_names, model_alias_registry),
                    evaluator_label=(
                        model_alias_registry.qa_alias(index)
                        if model_alias_registry
                        else model_name
                    ),
                    timeout_seconds=timeout,
                    slot_id=slot_id,
                    config_fingerprint=getattr(slot_meta, "config_fingerprint", None),
                )
            )

        smart_editing_mode = getattr(original_request, 'smart_editing_mode', 'auto') if original_request else 'auto'
        content_type = getattr(original_request, 'content_type', 'other') if original_request else 'other'
        request_edit_info = self._should_request_edit_info(
            mode=smart_editing_mode,
            content_type=content_type,
        )
        extra_verbose = getattr(original_request, 'extra_verbose', False) if original_request else False

        async def evaluate_slot(slot: QASchedulerSlot, attempt: int) -> QAEvaluation:
            retry_suffix = f" (retry {attempt})" if attempt > 1 else ""
            await _abort_if_cancelled(
                f"Cancelled before querying {slot.model_name} for layer {layer.name}."
            )

            if phase_logger:
                phase_logger.info(f"Querying model: {slot.model_name}{retry_suffix}")

            if progress_callback:
                await progress_callback(
                    f"Querying {slot.evaluator_label} for {layer.name}{retry_suffix}..."
                )

            usage_callback_fn = None
            if usage_tracker:
                usage_callback_fn = usage_tracker.create_callback(
                    phase="qa",
                    role="evaluation",
                    iteration=iteration,
                    layer=layer.name,
                    metadata={"qa_model": slot.model_name, "session_id": session_id},
                )

            try:
                evaluation = await asyncio.wait_for(
                    self.evaluate_content(
                        content,
                        layer,
                        slot.model,
                        original_request,
                        extra_verbose=extra_verbose,
                        stream_callback=stream_callback,
                        usage_callback=usage_callback_fn,
                        request_edit_info=request_edit_info,
                        phase_logger=phase_logger,
                        marker_mode=marker_mode,
                        marker_length=marker_length,
                        word_map_formatted=word_map_formatted,
                        draft_map_formatted=draft_map_formatted,
                        input_images=input_images,
                        edit_history=edit_history,
                        model_alias_registry=model_alias_registry,
                        session_id=session_id,
                        project_id=project_id,
                        tool_event_callback=tool_event_callback,
                    ),
                    timeout=slot.timeout_seconds,
                )
            except QAResponseParseError as parse_error:
                logger.warning(
                    "QA model %s returned invalid JSON for layer %s.",
                    slot.model_name,
                    layer.name,
                )
                if progress_callback:
                    await progress_callback(f"{slot.evaluator_label}: invalid JSON response. Skipping.")
                raise QASchedulerTechnicalFailure(
                    "parse_error",
                    str(parse_error),
                    metadata={"parse_error": "invalid_json"},
                    original_exception=parse_error,
                ) from parse_error
            except AIRequestError as api_err:
                failure_count = self._increment_model_failure(session_id, slot.model_name)
                cause = getattr(api_err, "cause", None)
                logger.error(
                    "QA model %s failed due to provider error in layer %s: %s",
                    slot.model_name,
                    layer.name,
                    api_err,
                    exc_info=True,
                )
                if phase_logger:
                    phase_logger.info(
                        f"Model {slot.model_name} provider error (failure {failure_count})."
                    )
                if progress_callback:
                    await progress_callback(
                        f"{slot.evaluator_label}: API error after "
                        f"{getattr(api_err, 'attempts', 0)} attempts."
                    )
                error_type = (
                    "model_unavailable"
                    if failure_count >= self._qa_failure_threshold()
                    else "api_failure"
                )
                raise QASchedulerTechnicalFailure(
                    error_type,
                    str(api_err),
                    metadata={
                        "attempts": getattr(api_err, "attempts", None),
                        "provider": getattr(api_err, "provider", None),
                        "failure_count": failure_count,
                        "exception_class": type(api_err).__name__,
                        "cause_class": type(cause).__name__ if cause is not None else None,
                        "cause_message": str(cause)[:500] if cause is not None else None,
                    },
                    original_exception=api_err,
                ) from api_err
            except asyncio.TimeoutError as timeout_error:
                logger.error("Timeout for %s with %s", layer.name, slot.model_name)
                if progress_callback:
                    await progress_callback(f"Timeout in {layer.name} with {slot.evaluator_label}")
                raise QASchedulerTechnicalFailure(
                    "timeout",
                    f"Timeout during evaluation with {slot.model_name}",
                    retryable=True,
                    metadata={"timeout_seconds": slot.timeout_seconds},
                    original_exception=timeout_error,
                ) from timeout_error
            except Exception as exc:
                logger.error(
                    "Evaluation failed for %s with %s: %s",
                    layer.name,
                    slot.model_name,
                    exc,
                    exc_info=True,
                )
                if progress_callback:
                    await progress_callback(
                        f"Error in {layer.name} with {slot.evaluator_label}: {str(exc)[:50]}"
                    )
                raise QASchedulerTechnicalFailure(
                    "unexpected",
                    f"Unexpected error during evaluation: {str(exc)}",
                    metadata={"exception_class": type(exc).__name__},
                    original_exception=exc,
                ) from exc

            _attach_evaluator_identity(evaluation, slot.model_name, slot.index, model_alias_registry)
            self._reset_model_failure(session_id, slot.model_name)

            score_text = f"{evaluation.score:.2f}" if evaluation.score is not None else "N/A"
            logger.info("Layer %s - Model %s: Score %s", layer.name, slot.model_name, score_text)

            if phase_logger:
                phase_logger.log_qa_result(
                    model=slot.model_name,
                    score=evaluation.score if evaluation.score is not None else 0.0,
                    is_deal_breaker=evaluation.deal_breaker,
                    feedback=evaluation.feedback,
                )

            if progress_callback:
                if evaluation.score is not None:
                    await progress_callback(f"{slot.evaluator_label}: {evaluation.score:.1f}/10")
                else:
                    await progress_callback(f"{slot.evaluator_label}: invalid QA response. Skipped.")

            if evaluation.deal_breaker:
                logger.warning(
                    "Deal-breaker detected in %s by %s: %s",
                    layer.name,
                    slot.model_name,
                    evaluation.deal_breaker_reason,
                )
                if progress_callback:
                    await progress_callback(
                        f"Deal-breaker detected in {layer.name} by {slot.evaluator_label}"
                    )

            await _abort_if_cancelled(
                f"Cancelled after evaluation from {slot.model_name} for layer {layer.name}."
            )
            return evaluation

        scheduler = QAScheduler(self._resolve_qa_scheduler_policy(original_request, len(slots)))
        try:
            scheduler_result = await scheduler.evaluate_layer(
                layer_name=layer.name,
                has_deal_breaker_criteria=bool(
                    getattr(layer, "deal_breaker_criteria", None)
                    or getattr(layer, "is_deal_breaker", False)
                ),
                slots=slots,
                evaluate_slot=evaluate_slot,
            )
            self._record_technical_failures_for_supervisor(
                session_id=session_id,
                iteration=iteration,
                layer_name=layer.name,
                layer_results=scheduler_result.layer_results,
            )
            return scheduler_result
        except QASchedulerUnavailableError as unavailable:
            if isinstance(unavailable.original_exception, QAResponseParseError):
                model_name = unavailable.slot.model_name if unavailable.slot else "QA model"
                raise QAResponseParseError(
                    f"QA model {model_name} returned invalid JSON response for layer {layer.name}"
                ) from unavailable.original_exception
            raise QAModelUnavailableError(str(unavailable)) from unavailable.original_exception

    def _record_technical_failures_for_supervisor(
        self,
        *,
        session_id: Optional[str],
        iteration: Optional[int],
        layer_name: str,
        layer_results: Dict[str, QAEvaluation],
    ) -> None:
        """Record technical QA failures as separate tracker events."""

        if not session_id:
            return
        technical_failures = [
            evaluation for evaluation in layer_results.values() if is_technical_qa_failure(evaluation)
        ]
        if not technical_failures:
            return
        try:
            from deal_breaker_tracker import get_tracker

            tracker = get_tracker()
            for evaluation in technical_failures:
                metadata = getattr(evaluation, "metadata", None) or {}
                tracker.record_technical_failure(
                    session_id=session_id,
                    layer_name=layer_name,
                    model_name=getattr(evaluation, "model", "unknown"),
                    slot_id=getattr(evaluation, "slot_id", None),
                    error_type=metadata.get("error_type") if isinstance(metadata, dict) else None,
                    reason=getattr(evaluation, "reason", None),
                    iteration=iteration,
                )
        except Exception:
            logger.debug("Failed to record QA technical failure tracker events.", exc_info=True)

    async def _evaluate_single_semantic_layer(
        self,
        content: str,
        layer: QALayer,
        qa_models: List[Any],
        qa_model_names: List[str],
        progress_callback: Optional[Callable] = None,
        original_request: Optional[Any] = None,
        stream_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        cancel_callback: Optional[Callable[[], Awaitable[bool]]] = None,
        usage_tracker: Optional[UsageTracker] = None,
        iteration: Optional[int] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        marker_mode: str = "phrase",
        marker_length: Optional[int] = None,
        word_map_formatted: Optional[str] = None,
        draft_map_formatted: Optional[str] = None,
        input_images: Optional[List["ImageData"]] = None,
        content_for_bypass: Optional[str] = None,
        edit_history: Optional[str] = None,
        model_alias_registry: Optional[ModelAliasRegistry] = None,
        project_id: Optional[str] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Tuple[Dict[str, QAEvaluation], Optional[Dict[str, Any]]]:
        """
        Evaluate a single semantic QA layer.

        This is the core evaluation logic extracted for reuse in per-layer smart-edit flow.
        Does NOT include Gran Sabio escalation - that is handled by the caller.

        Args:
            content: The content to evaluate
            layer: The QA layer configuration
            qa_models: List of QA models (strings or QAModelConfig)
            qa_model_names: Pre-extracted model names
            progress_callback: Optional callback for progress updates
            original_request: The original content request
            stream_callback: Optional callback for streaming
            session_id: Session identifier
            cancel_callback: Optional async callback to check for cancellation
            usage_tracker: Optional usage tracker
            iteration: Current iteration number
            phase_logger: Optional phase logger
            marker_mode: "ids", "phrase", or "word_index"
            marker_length: Number of words for phrase markers
            word_map_formatted: Formatted word map for word_index mode
            draft_map_formatted: Formatted paragraph/sentence ID map for ids mode
            input_images: Optional images for vision-enabled QA
            content_for_bypass: Optional extracted text for bypass evaluation

        Returns:
            Tuple of:
            - layer_results: Dict mapping model names to QAEvaluation objects
            - majority_deal_breaker_info: If majority deal-breaker detected, dict with consensus info; else None
        """
        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        async def _abort_if_cancelled(message: str) -> None:
            if cancel_callback and await cancel_callback():
                if progress_callback:
                    await progress_callback(message)
                raise QAProcessCancelled()

        extra_verbose = getattr(original_request, 'extra_verbose', False) if original_request else False

        await _abort_if_cancelled(f"Cancelled before evaluating layer {layer.name}.")

        # Log layer evaluation start
        if phase_logger:
            phase_logger.info(f"Evaluating layer: {layer.name}")
            if phase_logger.extra_verbose and layer.criteria:
                phase_logger.info(f"Criteria: {layer.criteria[:200]}...")

        if progress_callback:
            await progress_callback(f"Evaluating layer: {layer.name}")

        layer_results: Dict[str, QAEvaluation] = {}

        used_algorithmic_bypass = False

        # Algorithmic bypass check
        if self.bypass_engine.should_bypass_qa_layer(
            layer, original_request,
            extra_verbose=extra_verbose
        ):
            if progress_callback:
                await progress_callback(f"Using algorithmic bypass for {layer.name}...")

            bypass_content = content_for_bypass if content_for_bypass is not None else content
            bypass_results = self.bypass_engine.bypass_layer_evaluation(
                bypass_content,
                layer,
                qa_model_names,
                original_request,
                content_already_prepared=content_for_bypass is not None,
            )
            if bypass_results:
                used_algorithmic_bypass = True
                for index, model in enumerate(qa_models):
                    model_name = get_model_name(model)
                    evaluation = bypass_results.get(model_name)
                    if evaluation is None:
                        continue
                    if hasattr(evaluation, "model_copy"):
                        evaluation = evaluation.model_copy(deep=True)
                    evaluation.model = model_name
                    _attach_evaluator_identity(evaluation, model_name, index, model_alias_registry)
                    result_key = _qa_model_result_key(model_name, index, qa_model_names, model_alias_registry)
                    layer_results[result_key] = evaluation
                logger.info(f"Layer {layer.name} evaluated algorithmically with bypass engine")

                first_evaluation = next(iter(bypass_results.values()))
                if first_evaluation.deal_breaker:
                    logger.warning(f"Algorithmic deal-breaker detected in {layer.name}.")
                    if progress_callback:
                        await progress_callback(f"Algorithmic deal-breaker in {layer.name}.")
                    # Return with deal-breaker info (all models agree since it's algorithmic)
                    consensus_info = {
                        "immediate_stop": True,
                        "deal_breaker_count": len(qa_model_names),
                        "total_evaluated": len(qa_model_names),
                        "total_models": len(qa_model_names),
                        "deal_breaker_details": [{
                            "model": model,
                            "evaluator": model_alias_registry.qa_alias(index) if model_alias_registry else f"Algorithmic ({model})",
                            "reason": first_evaluation.deal_breaker_reason
                        } for index, model in enumerate(qa_model_names)],
                        "majority_threshold": len(qa_model_names) / 2
                    }
                    return layer_results, consensus_info

                if progress_callback:
                    for index, model in enumerate(qa_models):
                        model_name = get_model_name(model)
                        result_key = _qa_model_result_key(model_name, index, qa_model_names, model_alias_registry)
                        evaluation = layer_results[result_key]
                        evaluator_label = get_evaluator_alias(evaluation, fallback=model_name)
                        if evaluation.score is not None:
                            await progress_callback(f"{evaluator_label}: {evaluation.score:.1f}/10")
                        else:
                            await progress_callback(f"{evaluator_label}: invalid JSON response. Skipped.")

                await _abort_if_cancelled(f"Cancelled after bypass evaluation of layer {layer.name}.")
            else:
                logger.warning(
                    "Bypass engine returned no evaluations for %s; falling back to AI evaluation.",
                    layer.name,
                )

        if not used_algorithmic_bypass:
            scheduler_result = await self._evaluate_ai_layer_with_scheduler(
                content=content,
                layer=layer,
                qa_models=qa_models,
                qa_model_names=qa_model_names,
                progress_callback=progress_callback,
                original_request=original_request,
                stream_callback=stream_callback,
                session_id=session_id,
                cancel_callback=cancel_callback,
                usage_tracker=usage_tracker,
                iteration=iteration,
                phase_logger=phase_logger,
                marker_mode=marker_mode,
                marker_length=marker_length,
                word_map_formatted=word_map_formatted,
                draft_map_formatted=draft_map_formatted,
                input_images=input_images,
                edit_history=edit_history,
                model_alias_registry=model_alias_registry,
                project_id=project_id,
                tool_event_callback=tool_event_callback,
            )
            layer_results = scheduler_result.layer_results
            if scheduler_result.majority_deal_breaker:
                logger.warning("Majority deal-breaker consensus reached for layer %s.", layer.name)
                if progress_callback:
                    await progress_callback(f"Majority deal-breaker consensus in {layer.name}.")
                return layer_results, scheduler_result.majority_deal_breaker

        # No majority deal-breaker - return results without early stop
        return layer_results, None

    def _create_iteration_stop_result(self, partial_results: Dict[str, Any], consensus_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create result structure for stopping QA to force iteration due to majority deal-breakers"""
        
        all_scores = []
        total_evals = 0
        
        for layer_results in partial_results.values():
            for evaluation in layer_results.values():
                if hasattr(evaluation, 'score') and evaluation.score is not None:
                    all_scores.append(evaluation.score)
                    total_evals += 1
        
        return {
            "qa_results": partial_results,
            "summary": {
                "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                "min_score": min(all_scores) if all_scores else 0.0,
                "max_score": max(all_scores) if all_scores else 0.0,
                "total_evaluations": total_evals,
                "deal_breakers_count": consensus_info['deal_breaker_count'],
                "has_deal_breakers": True,
                "force_iteration": True,
                "stop_reason": "majority_deal_breaker_consensus_for_iteration"
            },
            "critical_issues": [
                {
                    "type": "majority_deal_breaker_iteration",
                    "description": f"Majority consensus ({consensus_info['deal_breaker_count']}/{consensus_info['total_evaluated']}) detected deal-breakers - forcing iteration",
                    "details": consensus_info["deal_breaker_details"]
                }
            ],
            "consensus_info": consensus_info,
            "layer_statistics": {},
            "model_statistics": {},
            "evaluation_metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,
                "total_evaluations": total_evals
            }
        }

    def _create_gran_sabio_modified_result(
        self,
        partial_results: Dict[str, Any],
        modified_content: str,
        reason: str,
        score: float
    ) -> Dict[str, Any]:
        """Create result structure when Gran Sabio approved with modifications."""

        all_scores = []
        total_evals = 0

        for layer_results in partial_results.values():
            for evaluation in layer_results.values():
                if hasattr(evaluation, 'score') and evaluation.score is not None:
                    all_scores.append(evaluation.score)
                    total_evals += 1

        return {
            "qa_results": partial_results,
            "summary": {
                "average_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
                "min_score": min(all_scores) if all_scores else 0.0,
                "max_score": max(all_scores) if all_scores else 0.0,
                "total_evaluations": total_evals,
                "deal_breakers_count": 0,
                "has_deal_breakers": False,
                "force_iteration": True,  # Force re-evaluation with new content
                "stop_reason": "gran_sabio_approved_with_modifications",
                "gran_sabio_modified": True,
            },
            "gran_sabio_modified_content": modified_content,
            "gran_sabio_modification_reason": reason,
            "gran_sabio_score": score,
            "critical_issues": [],
            "layer_statistics": {},
            "model_statistics": {},
            "evaluation_metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 0.0,
                "total_evaluations": total_evals
            }
        }
