"""
Consensus Engine Module for Gran Sabio LLM Engine
==================================================

Calculates consensus from multiple AI evaluations and determines
whether content meets quality standards across all layers.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING

from ai_service import AIService, get_ai_service
from models import QALayer, QAEvaluation, ConsensusResult
from model_aliasing import ModelAliasRegistry, get_evaluator_alias

if TYPE_CHECKING:
    from logging_utils import PhaseLogger


logger = logging.getLogger(__name__)

CONSENSUS_EXCLUDED_LAYER_NAMES = {"Evidence Grounding"}


class ConsensusEngine:
    """Consensus calculation and decision engine"""
    
    def __init__(self, ai_service: Optional[AIService] = None):
        """Initialize Consensus Engine with optional shared AI service."""
        self.ai_service = ai_service if ai_service is not None else get_ai_service()
    
    async def calculate_consensus(
        self,
        content: str,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
        layers: List[QALayer],
        original_request: Optional[Any] = None,
        phase_logger: Optional["PhaseLogger"] = None,
    ) -> ConsensusResult:
        """
        Calculate consensus from QA evaluations

        Args:
            content: Original content that was evaluated
            qa_results: QA results {layer_name: {model: QAEvaluation}}
            layers: List of QA layers with their configurations
            original_request: Original ContentRequest (provides iteration/mode context)
            phase_logger: Optional phase logger for detailed logging

        Returns:
            ConsensusResult with scoring and approval status
        """
        # Calculate layer averages
        layer_averages = {}
        for layer in layers:
            layer_name = layer.name
            if layer_name in qa_results:
                layer_results = qa_results[layer_name]
                scores = [eval.score for eval in layer_results.values() if eval.score is not None]
                if scores:
                    layer_averages[layer_name] = sum(scores) / len(scores)
                else:
                    layer_averages[layer_name] = 0.0
            else:
                layer_averages[layer_name] = 0.0
        
        # Calculate model averages
        per_model_averages = self._calculate_model_averages(qa_results)
        
        # Calculate overall average
        all_scores = [
            evaluation.score
            for _, _, evaluation in self._iter_consensus_scored_evaluations(qa_results)
        ]
        
        average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Collect deal-breakers and aggregate feedback for iteration guidance
        deal_breakers: List[str] = []
        alias_registry = getattr(original_request, "_model_alias_registry", None) if original_request else None
        for layer_name, layer_results in qa_results.items():
            for model, evaluation in layer_results.items():
                if bool(self._safe_get_evaluation_attr(evaluation, "deal_breaker", False)):
                    reason = self._safe_get_evaluation_attr(evaluation, "deal_breaker_reason") or \
                        self._safe_get_evaluation_attr(evaluation, "reason") or "Deal-breaker detected"
                    evaluator_name = get_evaluator_alias(
                        evaluation,
                        fallback=alias_registry.alias_for_identity(model, fallback=model) if alias_registry else model,
                    )
                    deal_breakers.append(f"{layer_name} ({evaluator_name}): {reason}")

        # Log consensus calculation info if phase_logger available
        if phase_logger:
            phase_logger.info(f"Calculating consensus across {len(layers)} layers")
            if phase_logger.extra_verbose:
                phase_logger.info("=== LAYER SCORES ===")
                for layer_name, avg_score in layer_averages.items():
                    phase_logger.info(f"  {layer_name}: {avg_score:.2f}/10")

        # Determine approval status
        approved = await self._determine_approval(
            content, qa_results, layers, layer_averages, average_score, deal_breakers, phase_logger
        )

        feedback_by_layer, actionable_feedback = self._collect_feedback_details(
            qa_results,
            layer_averages,
            alias_registry=alias_registry,
        )

        return ConsensusResult(
            average_score=average_score,
            layer_averages=layer_averages,
            per_model_averages=per_model_averages,
            total_evaluations=len(all_scores),
            approved=approved,
            deal_breakers=deal_breakers,
            feedback_by_layer=feedback_by_layer,
            actionable_feedback=actionable_feedback
        )
    
    def _calculate_model_averages(self, qa_results: Dict[str, Dict[str, QAEvaluation]]) -> Dict[str, float]:
        """Calculate average scores per model across all layers"""
        model_scores = {}
        
        # Collect scores by model
        for _, model, evaluation in self._iter_consensus_scored_evaluations(qa_results):
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(evaluation.score)
        
        # Calculate averages
        model_averages = {}
        for model, scores in model_scores.items():
            model_averages[model] = sum(scores) / len(scores) if scores else 0.0

        return model_averages

    def _iter_consensus_scored_evaluations(
        self,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
    ):
        """Yield scored evaluations that should influence consensus-wide aggregates."""
        for layer_name, layer_results in qa_results.items():
            if layer_name in CONSENSUS_EXCLUDED_LAYER_NAMES:
                continue

            for model, evaluation in layer_results.items():
                if evaluation.score is not None:
                    yield layer_name, model, evaluation
    
    async def _determine_approval(
        self,
        content: str,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
        layers: List[QALayer],
        layer_averages: Dict[str, float],
        average_score: float,
        deal_breakers: List[str],
        phase_logger: Optional["PhaseLogger"] = None,
    ) -> bool:
        """
        Determine if content should be approved based on all criteria
        
        Args:
            content: Original content
            qa_results: QA evaluation results
            layers: QA layer configurations
            layer_averages: Calculated layer averages
            average_score: Overall average score
            deal_breakers: List of deal-breaker issues
            
        Returns:
            Boolean indicating approval status
        """
        # Immediate rejection for deal-breakers
        if deal_breakers:
            rejection_reason = f"{len(deal_breakers)} deal-breaker(s) detected"
            if phase_logger:
                phase_logger.log_decision("REJECTED", reason=rejection_reason)
            else:
                logger.info(f"Content rejected due to {rejection_reason}")
            return False

        # Check each layer against its minimum score
        for layer in layers:
            layer_avg = layer_averages.get(layer.name, 0.0)
            if layer_avg < layer.min_score:
                rejection_reason = f"Layer {layer.name} score {layer_avg:.2f} < required {layer.min_score}"
                if phase_logger:
                    phase_logger.log_decision("REJECTED", reason=rejection_reason)
                else:
                    logger.info(f"Content rejected: {rejection_reason}")
                return False

        if phase_logger:
            phase_logger.log_decision("APPROVED", score=average_score)
        else:
            logger.info(f"Content approved with average score: {average_score:.2f}")
        return True

    def _collect_feedback_details(
        self,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
        layer_averages: Dict[str, float],
        alias_registry: Optional[ModelAliasRegistry] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Aggregate per-layer feedback and actionable notes for iteration guidance."""

        layer_feedback: List[Dict[str, Any]] = []
        actionable_feedback: List[str] = []
        seen_actionable: Set[str] = set()

        for layer_name, model_results in qa_results.items():
            layer_entry: Dict[str, Any] = {
                "layer": layer_name,
                "average_score": layer_averages.get(layer_name, 0.0),
                "deal_breakers": [],
                "model_feedback": []
            }

            for model_name, evaluation in model_results.items():
                evaluator_name = get_evaluator_alias(
                    evaluation,
                    fallback=alias_registry.alias_for_identity(model_name, fallback=model_name) if alias_registry else model_name,
                )
                score = self._safe_get_evaluation_attr(evaluation, "score", 0.0)
                feedback_text = (self._safe_get_evaluation_attr(evaluation, "feedback", "") or "").strip()
                deal_breaker = bool(self._safe_get_evaluation_attr(evaluation, "deal_breaker", False))
                deal_breaker_reason = (self._safe_get_evaluation_attr(evaluation, "deal_breaker_reason", "") or
                                       self._safe_get_evaluation_attr(evaluation, "reason", "") or "").strip()

                model_entry = {
                    "model": model_name,
                    "evaluator": evaluator_name,
                    "score": score,
                    "deal_breaker": deal_breaker,
                    "feedback": feedback_text
                }

                if deal_breaker_reason:
                    model_entry["deal_breaker_reason"] = deal_breaker_reason
                    layer_entry["deal_breakers"].append(f"{evaluator_name}: {deal_breaker_reason}")

                layer_entry["model_feedback"].append(model_entry)

                actionable_text = deal_breaker_reason or feedback_text
                if actionable_text:
                    formatted = f"{layer_name} ({evaluator_name}): {actionable_text}"
                    if formatted not in seen_actionable:
                        actionable_feedback.append(formatted)
                        seen_actionable.add(formatted)

            layer_feedback.append(layer_entry)

        return layer_feedback, actionable_feedback

    @staticmethod
    def _safe_get_evaluation_attr(evaluation: Any, attr: str, default: Any = None) -> Any:
        """Safely retrieve an attribute from QAEvaluation or dict without assuming structure."""

        if hasattr(evaluation, attr):
            return getattr(evaluation, attr)
        if isinstance(evaluation, dict):
            return evaluation.get(attr, default)
        return default
