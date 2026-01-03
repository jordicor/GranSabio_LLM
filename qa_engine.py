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
from typing import List, Dict, Any, Optional, Callable, Awaitable, TYPE_CHECKING
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    from logging_utils import PhaseLogger

from ai_service import AIService, get_ai_service, AIRequestError
from qa_evaluation_service import QAEvaluationService
from usage_tracking import UsageTracker
from models import QALayer, QAEvaluation
from config import config
from qa_bypass_engine import QABypassEngine
from gran_sabio import GranSabioInvocationError, GranSabioProcessCancelled


logger = logging.getLogger(__name__)


class QAProcessCancelled(Exception):
    """Raised when a QA evaluation flow is cancelled by the user."""


class QAModelUnavailableError(RuntimeError):
    """Raised when a QA model cannot be queried reliably anymore."""


CancelCallback = Optional[Callable[[], Awaitable[bool]]]


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
        self._qa_failure_tracker: Dict[str, Dict[str, int]] = defaultdict(dict)

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
        editable_types = ["biography", "article", "script", "story", "essay", "blog", "novel"]
        return content_type in editable_types

    def _feedback_suggests_edits(self, feedback: str) -> bool:
        """
        Kept for backward compatibility. Not used in the simplified flow.
        """
        if not feedback:
            return False
        return False

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
    ) -> QAEvaluation:
        """
        Evaluate content using a specific QA layer and AI model

        Args:
            marker_mode: "phrase" for text markers, "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-12)
            word_map_formatted: Formatted word map string for word_index mode
        """
        from models import QAModelConfig

        # Normalize model to QAModelConfig
        if isinstance(model, str):
            model_config = QAModelConfig(model=model)
        else:
            model_config = model

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
            )
        except Exception as e:
            logger.error(f"QA evaluation failed for layer {layer.name} with model {model_config.model}: {str(e)}")
            raise

    async def evaluate_all_layers(
        self,
        content: str,
        layers: List[QALayer],
        qa_models: List[str]
    ) -> Dict[str, Dict[str, QAEvaluation]]:
        """
        Evaluate content through all QA layers with all specified models.
        (Non-progress variant)
        """
        # Sort layers by order
        sorted_layers = sorted(layers, key=lambda x: x.order)
        results: Dict[str, Dict[str, QAEvaluation]] = {}

        multiple_models = len(qa_models) > 1

        # Process each layer
        for layer in sorted_layers:
            layer_results: Dict[str, QAEvaluation] = {}

            # Evaluate with each QA model concurrently with individual timeouts
            tasks = []
            for model in qa_models:
                individual_timeout = calculate_qa_timeout_for_model(model)
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.evaluate_content(content, layer, model),
                        timeout=individual_timeout
                    ),
                    name=f"{layer.name}_{model}"
                )
                tasks.append((model, task))
            
            # Wait for all evaluations for this layer
            for model, task in tasks:
                try:
                    evaluation = await task
                    layer_results[model] = evaluation
                    
                    score_text = f"{evaluation.score:.2f}" if evaluation.score is not None else "N/A"
                    logger.info(f"Layer {layer.name} - Model {model}: Score {score_text}")
                    
                    if evaluation.deal_breaker:
                        logger.warning(f"Deal-breaker detected in {layer.name} by {model}: {evaluation.reason}")
                
                except ValueError as parse_error:
                    if multiple_models:
                        logger.warning(
                            "QA model %s returned invalid JSON for layer %s. Skipping this evaluation.",
                            model,
                            layer.name
                        )
                        layer_results[model] = QAEvaluation(
                            model=model,
                            layer=layer.name,
                            score=None,
                            feedback=str(parse_error),
                            deal_breaker=False,
                            deal_breaker_reason=None,
                            passes_score=False,
                            reason=str(parse_error),
                            metadata={"parse_error": "invalid_json"}
                        )
                    else:
                        raise ValueError(
                            f"QA model {model} returned invalid JSON response for layer {layer.name}"
                        ) from parse_error
                except AIRequestError as api_err:
                    logger.error(
                        "QA model %s failed due to provider error in layer %s: %s",
                        model,
                        layer.name,
                        api_err,
                    )
                    if not multiple_models:
                        raise QAModelUnavailableError(
                            f"QA model {model} unavailable for layer {layer.name}: {api_err}"
                        ) from api_err

                    layer_results[model] = QAEvaluation(
                        model=model,
                        layer=layer.name,
                        score=None,
                        feedback=str(api_err),
                        deal_breaker=False,
                        deal_breaker_reason=None,
                        passes_score=False,
                        reason=str(api_err),
                        metadata={
                            "error_type": "api_failure",
                            "attempts": getattr(api_err, "attempts", None),
                            "provider": getattr(api_err, "provider", None),
                        },
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout for {layer.name} with {model}")
                    layer_results[model] = QAEvaluation(
                        model=model,
                        layer=layer.name,
                        score=0.0,
                        feedback=f"Timeout during evaluation with {model}",
                        deal_breaker=True,
                        reason="Timeout during evaluation"
                    )
                except Exception as e:
                    logger.error(f"Evaluation failed for {layer.name} with {model}: {str(e)}")
                    layer_results[model] = QAEvaluation(
                        model=model,
                        layer=layer.name,
                        score=0.0,
                        feedback=f"Error during evaluation: {str(e)}",
                        deal_breaker=True,
                        reason="Technical error"
                    )
            
            results[layer.name] = layer_results
            
            # Deal-breaker layers: keep evaluating remaining layers to present complete feedback
            deal_breakers = [eval for eval in layer_results.values() if eval.deal_breaker]
            if deal_breakers and layer.is_deal_breaker:
                logger.info(f"Deal-breaker layer {layer.name} failed, continuing for comprehensive feedback")
        
        return results
    
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
    ) -> Dict[str, Dict[str, QAEvaluation]]:
        """
        Evaluate content through all QA layers with detailed progress tracking.

        Args:
            marker_mode: "phrase" for text markers, "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-12)
            word_map_formatted: Formatted word map string for word_index mode
        """
        from models import QAModelConfig

        def get_model_name(model: Any) -> str:
            return model.model if isinstance(model, QAModelConfig) else model

        qa_model_names = [get_model_name(m) for m in qa_models]
        sorted_layers = sorted(layers, key=lambda x: x.order)

        escalations_this_evaluation = 0
        iteration_limit = getattr(original_request, 'gran_sabio_call_limit_per_iteration', -1)
        results: Dict[str, Dict[str, QAEvaluation]] = {}

        async def _abort_if_cancelled(message: str) -> None:
            if cancel_callback and await cancel_callback():
                if progress_callback:
                    await progress_callback(message)
                raise QAProcessCancelled()
        
        for layer in sorted_layers:
            await _abort_if_cancelled(f"🛑 Cancelled before evaluating layer {layer.name}.")

            # Log layer evaluation start with phase_logger
            if phase_logger:
                phase_logger.info(f"Evaluating layer: {layer.name}")
                if phase_logger.extra_verbose and layer.criteria:
                    phase_logger.info(f"Criteria: {layer.criteria[:200]}...")

            if progress_callback:
                await progress_callback(f"📊 Evaluating layer: {layer.name}")

            layer_results: Dict[str, QAEvaluation] = {}
            
            # Algorithmic bypass (unchanged)
            if self.bypass_engine.should_bypass_qa_layer(
                layer, original_request,
                extra_verbose=getattr(original_request, 'extra_verbose', False) if original_request else False
            ):
                if progress_callback:
                    await progress_callback(f"🔄 Using algorithmic bypass for {layer.name}...")

                bypass_results = self.bypass_engine.bypass_layer_evaluation(content, layer, qa_model_names, original_request)
                layer_results.update(bypass_results)
                logger.info(f"Layer {layer.name} evaluated algorithmically with bypass engine")

                first_evaluation = next(iter(bypass_results.values()))
                if first_evaluation.deal_breaker:
                    logger.warning(f"Algorithmic deal-breaker detected in {layer.name}. Stopping to iterate.")
                    if progress_callback:
                        await progress_callback(f"🛑 Algorithmic deal-breaker in {layer.name}. Stopping to iterate.")
                    results[layer.name] = layer_results
                    consensus_info = {
                        "immediate_stop": True,
                        "deal_breaker_count": len(qa_model_names),
                        "total_evaluated": len(qa_model_names),
                        "total_models": len(qa_model_names),
                        "deal_breaker_details": [{
                            "model": f"Algorithmic ({model})",
                            "reason": first_evaluation.deal_breaker_reason
                        } for model in qa_model_names],
                        "majority_threshold": len(qa_model_names) / 2
                    }
                    return self._create_iteration_stop_result(results, consensus_info)

                if progress_callback:
                    for model in qa_models:
                        model_name = get_model_name(model)
                        evaluation = layer_results[model_name]
                        if evaluation.score is not None:
                            await progress_callback(f"✅ {evaluation.model}: {evaluation.score:.1f}/10")
                        else:
                            await progress_callback(
                                f"⚠️ {evaluation.model}: invalid JSON response. Evaluation skipped for this model."
                            )
                await _abort_if_cancelled(f"🛑 Cancelled after bypass evaluation of layer {layer.name}.")
                
            else:
                # Normal AI-based evaluation
                for model in qa_models:
                    model_name = get_model_name(model)
                    await _abort_if_cancelled(f"🛑 Cancelled before querying {model_name} for layer {layer.name}.")

                    if phase_logger:
                        phase_logger.info(f"Querying model: {model_name}")

                    if progress_callback:
                        await progress_callback(f"🧠 Querying {model_name} for {layer.name}...")

                    if isinstance(model, QAModelConfig):
                        individual_timeout = calculate_qa_timeout_for_model(
                            model.model, model.reasoning_effort, model.thinking_budget_tokens
                        )
                    else:
                        individual_timeout = calculate_qa_timeout_for_model(model)

                    usage_callback_fn = None
                    if usage_tracker:
                        usage_callback_fn = usage_tracker.create_callback(
                            phase="qa",
                            role="evaluation",
                            iteration=iteration,
                            layer=layer.name,
                            metadata={"qa_model": model_name, "session_id": session_id},
                        )

                    # Determine if we should request edit info based on smart_editing_mode
                    smart_editing_mode = getattr(original_request, 'smart_editing_mode', 'auto') if original_request else 'auto'
                    content_type = getattr(original_request, 'content_type', 'other') if original_request else 'other'
                    request_edit_info = self._should_request_edit_info(
                        mode=smart_editing_mode,
                        content_type=content_type
                    )

                    try:
                        evaluation = await asyncio.wait_for(
                            self.evaluate_content(
                                content,
                                layer,
                                model,
                                original_request,
                                extra_verbose=getattr(original_request, 'extra_verbose', False) if original_request else False,
                                stream_callback=stream_callback,
                                usage_callback=usage_callback_fn,
                                request_edit_info=request_edit_info,
                                phase_logger=phase_logger,
                                marker_mode=marker_mode,
                                marker_length=marker_length,
                                word_map_formatted=word_map_formatted,
                            ),
                            timeout=individual_timeout
                        )

                        layer_results[model_name] = evaluation
                        self._reset_model_failure(session_id, model_name)

                        score_text = f"{evaluation.score:.2f}" if evaluation.score is not None else "N/A"
                        logger.info(f"Layer {layer.name} - Model {model_name}: Score {score_text}")

                        # Log QA result with phase_logger
                        if phase_logger:
                            phase_logger.log_qa_result(
                                model=model_name,
                                score=evaluation.score if evaluation.score is not None else 0.0,
                                is_deal_breaker=evaluation.deal_breaker,
                                feedback=evaluation.feedback
                            )

                        if progress_callback:
                            if evaluation.score is not None:
                                await progress_callback(f"✅ {model_name}: {evaluation.score:.1f}/10")
                            else:
                                await progress_callback(f"⚠️ {model_name}: invalid JSON response. Evaluation skipped for this model.")

                        if evaluation.deal_breaker:
                            logger.warning(f"Deal-breaker detected in {layer.name} by {model_name}: {evaluation.deal_breaker_reason}")
                            if progress_callback:
                                await progress_callback(f"⛔ Deal-breaker detected in {layer.name} by {model_name}")
                        
                        deal_breaker_consensus = self._check_deal_breaker_consensus(layer_results, qa_model_names)
                        if deal_breaker_consensus["immediate_stop"]:
                            logger.warning(f"Majority deal-breaker consensus reached for layer {layer.name}. Stopping evaluations to force iteration.")
                            if progress_callback:
                                await progress_callback(f"🛑 Majority deal-breaker consensus in {layer.name}. Stopping to iterate.")
                            results[layer.name] = layer_results
                            return self._create_iteration_stop_result(results, deal_breaker_consensus)
                    
                    except ValueError as parse_error:
                        if len(qa_model_names) > 1:
                            logger.warning(
                                "QA model %s returned invalid JSON for layer %s. Skipping this evaluation.",
                                model_name, layer.name
                            )
                            placeholder = QAEvaluation(
                                model=model_name,
                                layer=layer.name,
                                score=None,
                                feedback=str(parse_error),
                                deal_breaker=False,
                                deal_breaker_reason=None,
                                passes_score=False,
                                reason=str(parse_error),
                                metadata={"parse_error": "invalid_json"}
                            )
                            layer_results[model_name] = placeholder

                            if progress_callback:
                                await progress_callback(f"⚠️ {model_name}: invalid JSON response. Skipping this QA response.")
                        else:
                            raise ValueError(
                                f"QA model {model_name} returned invalid JSON response for layer {layer.name}"
                            ) from parse_error
                    except AIRequestError as api_err:
                        logger.error(
                            "QA model %s failed due to provider error in layer %s: %s",
                            model_name,
                            layer.name,
                            api_err,
                        )

                        if len(qa_model_names) == 1:
                            raise QAModelUnavailableError(
                                f"QA model {model_name} unavailable for layer {layer.name}: {api_err}"
                            ) from api_err

                        failure_count = self._increment_model_failure(session_id, model_name)
                        placeholder = QAEvaluation(
                            model=model_name,
                            layer=layer.name,
                            score=None,
                            feedback=str(api_err),
                            deal_breaker=False,
                            deal_breaker_reason=None,
                            passes_score=False,
                            reason=str(api_err),
                            metadata={
                                "error_type": "api_failure",
                                "attempts": getattr(api_err, "attempts", None),
                                "provider": getattr(api_err, "provider", None),
                                "failure_count": failure_count,
                            }
                        )
                        layer_results[model_name] = placeholder

                        if phase_logger:
                            phase_logger.info(
                                f"Model {model_name} skipped due to provider error (failure {failure_count})."
                            )
                        if progress_callback:
                            await progress_callback(
                                f"⚠️ {model_name}: API error tras {getattr(api_err, 'attempts', 0)} intentos. Saltando (fallo {failure_count})."
                            )

                        if failure_count >= self._qa_failure_threshold():
                            raise QAModelUnavailableError(
                                f"QA model {model_name} failed {failure_count} times in this session"
                            ) from api_err
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout for {layer.name} with {model_name}")
                        if progress_callback:
                            await progress_callback(f"⏰ Timeout in {layer.name} with {model_name}")
                        layer_results[model_name] = QAEvaluation(
                            model=model_name,
                            layer=layer.name,
                            score=0.0,
                            feedback=f"Timeout during evaluation with {model_name}",
                            deal_breaker=True,
                            deal_breaker_reason="Timeout during evaluation",
                            passes_score=False,
                            reason="Timeout during evaluation",
                            identified_issues=None
                        )
                    except Exception as e:
                        logger.error(f"Evaluation failed for {layer.name} with {model_name}: {str(e)}")
                        if progress_callback:
                            await progress_callback(f"❌ Error in {layer.name} with {model_name}: {str(e)[:50]}")
                        layer_results[model_name] = QAEvaluation(
                            model=model_name,
                            layer=layer.name,
                            score=0.0,
                            feedback=f"Error during evaluation: {str(e)}",
                            deal_breaker=True,
                            deal_breaker_reason="Technical error during evaluation",
                            passes_score=False,
                            reason="Technical error during evaluation",
                            identified_issues=None
                        )

                    await _abort_if_cancelled(
                        f"🛑 Cancelled after receiving evaluation from {model_name} for layer {layer.name}."
                    )

            results[layer.name] = layer_results

            # ===== Minority deal-breakers & Gran Sabio logic (unchanged except logging text) =====
            deal_breakers = [eval for eval in layer_results.values() if eval.deal_breaker]

            if deal_breakers:
                total_models = len(qa_model_names)
                deal_breaker_count = len(deal_breakers)

                is_majority = deal_breaker_count > (total_models / 2)
                is_tie = (total_models % 2 == 0) and (deal_breaker_count == total_models / 2)
                is_minority = deal_breaker_count < (total_models / 2)

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
                        "total_models": total_models,
                        "deal_breaker_details": [
                            {"model": eval.model, "reason": eval.deal_breaker_reason or eval.reason or ""}
                            for eval in deal_breakers
                        ],
                        "majority_threshold": total_models / 2
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
                            gran_sabio_model=getattr(original_request, "gran_sabio_model", None)
                        )

                        minority_data = {
                            "has_minority_deal_breakers": True,
                            "deal_breaker_count": deal_breaker_count,
                            "total_evaluations": total_models,
                            "details": [
                                {
                                    "layer": layer.name,
                                    "model": eval.model,
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

                                for eval in deal_breakers:
                                    original_reason = eval.deal_breaker_reason or eval.reason or ""
                                    eval.deal_breaker = False
                                    eval.deal_breaker_reason = None
                                    eval.reason = (
                                        f"[Gran Sabio Override] Originally flagged as deal-breaker but "
                                        f"Gran Sabio determined it to be false positive. "
                                        f"Original reason: {original_reason}"
                                    )
                                    if getattr(gs_result, "final_score", None) is not None:
                                        prev_score = eval.score
                                        eval.score = gs_result.final_score
                                        eval.passes_score = (gs_result.final_score >= getattr(layer, "min_score", 0.0))
                                        logger.debug("Gran Sabio override updated %s score from %s to %s", eval.model, prev_score, eval.score)

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
    ) -> Dict[str, Any]:
        """
        Comprehensive content evaluation with progress tracking

        Args:
            marker_mode: "phrase" for text markers, "word_index" for word map indices
            marker_length: Number of words for phrase markers (4-12)
            word_map_formatted: Formatted word map string for word_index mode
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
        )

        if isinstance(qa_results, dict) and qa_results.get("summary", {}).get("force_iteration"):
            return qa_results

        summary = self._calculate_summary(qa_results, layers)
        critical_issues = self._identify_critical_issues(qa_results)
        layer_stats = self._calculate_layer_statistics(qa_results, layers)
        model_stats = self._calculate_model_statistics(qa_results, qa_model_names)
        
        end_time = datetime.now()
        evaluation_time = (end_time - start_time).total_seconds()
        
        if progress_callback:
            await progress_callback(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
        
        return {
            "qa_results": qa_results,
            "summary": summary,
            "critical_issues": critical_issues,
            "layer_statistics": layer_stats,
            "model_statistics": model_stats,
            "evaluation_metadata": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": evaluation_time,
                "total_evaluations": len(layers) * len(qa_models)
            }
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
    
    def _check_deal_breaker_consensus(self, layer_results: Dict[str, Any], total_models: List[str]) -> Dict[str, Any]:
        """
        Check if there's a majority consensus for deal-breakers in current evaluations
        """
        deal_breaker_count = 0
        deal_breaker_details = []

        for model, evaluation in layer_results.items():
            if evaluation.deal_breaker:
                deal_breaker_count += 1
                deal_breaker_details.append({
                    "model": model,
                    "reason": evaluation.deal_breaker_reason
                })

        total_evaluated = len(layer_results)
        total_models_count = len(total_models)

        majority_threshold = total_models_count / 2
        immediate_stop = deal_breaker_count > majority_threshold

        return {
            "immediate_stop": immediate_stop,
            "deal_breaker_count": deal_breaker_count,
            "total_evaluated": total_evaluated,
            "total_models": total_models_count,
            "deal_breaker_details": deal_breaker_details,
            "majority_threshold": majority_threshold
        }
    
    def _create_immediate_stop_result(self, partial_results: Dict[str, Any], consensus_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create result structure for immediate stop due to deal-breaker consensus"""
        
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
                "immediate_stop": True,
                "stop_reason": "majority_deal_breaker_consensus"
            },
            "critical_issues": [
                {
                    "type": "majority_deal_breaker",
                    "description": f"Majority consensus ({consensus_info['deal_breaker_count']}/{consensus_info['total_evaluated']}) detected deal-breakers",
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