"""
Consensus Engine Module for Gran Sabio LLM Engine
==================================================

Calculates consensus from multiple AI evaluations and determines
whether content meets quality standards across all layers.
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, TYPE_CHECKING
from datetime import datetime

from ai_service import AIService, get_ai_service
from models import QALayer, QAEvaluation, ConsensusResult
from config import config

if TYPE_CHECKING:
    from logging_utils import PhaseLogger


logger = logging.getLogger(__name__)


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
        all_scores = []
        for layer_results in qa_results.values():
            for evaluation in layer_results.values():
                if evaluation.score is not None:
                    all_scores.append(evaluation.score)
        
        average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Collect deal-breakers and aggregate feedback for iteration guidance
        deal_breakers: List[str] = []
        for layer_name, layer_results in qa_results.items():
            for model, evaluation in layer_results.items():
                if bool(self._safe_get_evaluation_attr(evaluation, "deal_breaker", False)):
                    reason = self._safe_get_evaluation_attr(evaluation, "deal_breaker_reason") or \
                        self._safe_get_evaluation_attr(evaluation, "reason") or "Deal-breaker detected"
                    deal_breakers.append(f"{layer_name} ({model}): {reason}")

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

        feedback_by_layer, actionable_feedback = self._collect_feedback_details(qa_results, layer_averages)

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
        for layer_results in qa_results.values():
            for model, evaluation in layer_results.items():
                if evaluation.score is not None:
                    if model not in model_scores:
                        model_scores[model] = []
                    model_scores[model].append(evaluation.score)
        
        # Calculate averages
        model_averages = {}
        for model, scores in model_scores.items():
            model_averages[model] = sum(scores) / len(scores) if scores else 0.0

        return model_averages
    
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
        layer_averages: Dict[str, float]
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
                score = self._safe_get_evaluation_attr(evaluation, "score", 0.0)
                feedback_text = (self._safe_get_evaluation_attr(evaluation, "feedback", "") or "").strip()
                deal_breaker = bool(self._safe_get_evaluation_attr(evaluation, "deal_breaker", False))
                deal_breaker_reason = (self._safe_get_evaluation_attr(evaluation, "deal_breaker_reason", "") or
                                       self._safe_get_evaluation_attr(evaluation, "reason", "") or "").strip()

                model_entry = {
                    "model": model_name,
                    "score": score,
                    "deal_breaker": deal_breaker,
                    "feedback": feedback_text
                }

                if deal_breaker_reason:
                    model_entry["deal_breaker_reason"] = deal_breaker_reason
                    layer_entry["deal_breakers"].append(f"{model_name}: {deal_breaker_reason}")

                layer_entry["model_feedback"].append(model_entry)

                actionable_text = deal_breaker_reason or feedback_text
                if actionable_text:
                    formatted = f"{layer_name} ({model_name}): {actionable_text}"
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
    
    async def advanced_consensus_analysis(
        self,
        content: str,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
        layers: List[QALayer],
        consensus_model: str = "gpt-4o"
    ) -> Dict[str, Any]:
        """
        Perform advanced consensus analysis using AI to interpret disagreements
        
        Args:
            content: Original content
            qa_results: QA evaluation results
            layers: QA layer configurations
            consensus_model: AI model to use for consensus analysis
            
        Returns:
            Advanced consensus analysis results
        """
        # Prepare analysis prompt
        analysis_prompt = self._build_consensus_analysis_prompt(content, qa_results, layers)
        
        try:
            # Get AI consensus analysis
            consensus_analysis = await self.ai_service.generate_content(
                prompt=analysis_prompt,
                model=consensus_model,
                temperature=0.3,
                max_tokens=2000,
                system_prompt=config.CONSENSUS_SYSTEM_PROMPT
            )
            
            # Parse and structure the analysis
            structured_analysis = self._parse_consensus_analysis(consensus_analysis)
            
            # Identify model disagreements
            disagreements = self._identify_model_disagreements(qa_results)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(qa_results)
            
            return {
                "consensus_analysis": consensus_analysis,
                "structured_analysis": structured_analysis,
                "model_disagreements": disagreements,
                "confidence_metrics": confidence_metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Advanced consensus analysis failed: {str(e)}")
            return {
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _build_consensus_analysis_prompt(
        self,
        content: str,
        qa_results: Dict[str, Dict[str, QAEvaluation]],
        layers: List[QALayer]
    ) -> str:
        """Build prompt for AI consensus analysis"""
        
        # Summarize evaluations by layer
        evaluation_summary = []
        for layer in sorted(layers, key=lambda x: x.order):
            layer_name = layer.name
            if layer_name in qa_results:
                layer_results = qa_results[layer_name]
                layer_summary = f"\n=== {layer_name} ===\n"
                layer_summary += f"Criteria: {layer.description}\n"
                layer_summary += f"Minimum required score: {layer.min_score}\n"
                
                for model, evaluation in layer_results.items():
                    score_display = f"{evaluation.score}/10" if evaluation.score is not None else "Score unavailable"
                    layer_summary += f"\n{model}: {score_display}\n"
                    layer_summary += f"Feedback: {evaluation.feedback}\n"
                    if evaluation.deal_breaker:
                        layer_summary += f"⚠️ DEAL-BREAKER: {evaluation.reason}\n"
                
                evaluation_summary.append(layer_summary)
        
        prompt = f"""
Analyze the following quality evaluations to determine a professional editorial consensus.

EVALUATED CONTENT:
{content}

EVALUATIONS BY LAYER:
{''.join(evaluation_summary)}

TASKS:
1. Identify patterns of consensus and disagreement among evaluators
2. Evaluate the validity of each critique
3. Determine whether discrepancies are minor or fundamental
4. Provide a final editorial recommendation
5. Suggest specific areas for improvement if necessary

RESPONSE FORMAT:
[CONSENSUS_SCORE]X.X[/CONSENSUS_SCORE]
[RECOMMENDATION]APPROVE/REJECT/CONDITIONAL[/RECOMMENDATION]
[ANALYSIS]Your detailed analysis[/ANALYSIS]
[IMPROVEMENTS]Suggested improvements if applicable[/IMPROVEMENTS]
"""
        
        return prompt
    
    def _parse_consensus_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse structured information from AI consensus analysis"""
        import re
        
        parsed = {}
        
        # Extract consensus score
        score_match = re.search(r'\[CONSENSUS_SCORE\]([\d\.]+)\[/CONSENSUS_SCORE\]', analysis, re.IGNORECASE)
        parsed['consensus_score'] = float(score_match.group(1)) if score_match else None
        
        # Extract recommendation
        rec_match = re.search(r'\[RECOMMENDATION\](APPROVE|REJECT|CONDITIONAL)\[/RECOMMENDATION\]', analysis, re.IGNORECASE)
        parsed['recommendation'] = rec_match.group(1).upper() if rec_match else "UNKNOWN"
        
        # Extract analysis
        analysis_match = re.search(r'\[ANALYSIS\](.*?)\[/ANALYSIS\]', analysis, re.IGNORECASE | re.DOTALL)
        parsed['analysis'] = analysis_match.group(1).strip() if analysis_match else ""
        
        # Extract improvements
        imp_match = re.search(r'\[IMPROVEMENTS\](.*?)\[/IMPROVEMENTS\]', analysis, re.IGNORECASE | re.DOTALL)
        parsed['improvements'] = imp_match.group(1).strip() if imp_match else ""
        
        return parsed
    
    def _identify_model_disagreements(self, qa_results: Dict[str, Dict[str, QAEvaluation]]) -> List[Dict[str, Any]]:
        """Identify significant disagreements between models"""
        disagreements = []
        
        for layer_name, layer_results in qa_results.items():
            if len(layer_results) < 2:
                continue
            
            scores = [eval.score for eval in layer_results.values() if eval.score is not None]
            
            if len(scores) < 2:
                continue
            
            # Calculate score variance
            score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
            score_range = max(scores) - min(scores)
            
            # Identify significant disagreements
            if score_range > 3.0:  # More than 3 points difference
                disagreements.append({
                    "layer": layer_name,
                    "score_range": score_range,
                    "score_variance": score_variance,
                    "scores": {model: eval.score for model, eval in layer_results.items()},
                    "severity": "high" if score_range > 5.0 else "moderate"
                })
        
        return disagreements
    
    def _calculate_confidence_metrics(self, qa_results: Dict[str, Dict[str, QAEvaluation]]) -> Dict[str, float]:
        """Calculate confidence metrics for the consensus"""
        all_scores = []
        layer_variances = []
        
        for layer_results in qa_results.values():
            scores = [eval.score for eval in layer_results.values() if eval.score is not None]
            if len(scores) > 1:
                layer_variance = statistics.variance(scores)
                layer_variances.append(layer_variance)
            all_scores.extend(scores)
        
        metrics = {}
        
        # Overall score consistency
        if len(all_scores) > 1:
            overall_variance = statistics.variance(all_scores)
            metrics['overall_variance'] = overall_variance
            metrics['consistency_score'] = max(0.0, 1.0 - (overall_variance / 25.0))  # Normalize to 0-1
        else:
            metrics['overall_variance'] = 0.0
            metrics['consistency_score'] = 0.0
        
        # Average layer consistency
        if layer_variances:
            metrics['average_layer_variance'] = sum(layer_variances) / len(layer_variances)
            metrics['layer_consistency'] = max(0.0, 1.0 - (metrics['average_layer_variance'] / 25.0))
        else:
            metrics['average_layer_variance'] = 0.0
            metrics['layer_consistency'] = 1.0
        
        # Overall confidence (weighted average of consistency metrics)
        metrics['overall_confidence'] = (metrics['consistency_score'] * 0.6 + metrics['layer_consistency'] * 0.4)
        
        return metrics
