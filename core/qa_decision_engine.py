"""
Layer-based QA approval logic and deal-breaker detection helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models import QALayer

from .feedback_formatter import create_user_friendly_reason


def _evaluate_layer_based_approval(
    qa_results,
    consensus_result,
    request,
    session_id,
    iteration,
    evaluated_layers,
    qa_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    New layer-based approval logic with deal-breaker consensus:
    1. Check for majority deal-breaker consensus - force iteration unless max iterations reached
    2. Check for minority deal-breakers (escalate to Gran Sabio)
    3. Check each layer individually for pass/fail
    4. Mandatory layers that fail after max iterations -> FINAL REJECTION
    5. If no mandatory failures, check global score override
    """

    # Step 1: Check if this came from a force_iteration (majority deal-breakers detected early)
    if qa_summary and qa_summary.get("summary", {}).get("force_iteration"):
        if iteration >= request.max_iterations:
            # Check if Gran Sabio fallback is enabled before final rejection
            if hasattr(request, "gran_sabio_fallback") and request.gran_sabio_fallback:
                # Allow fallback instead of final rejection
                user_friendly_reason = create_user_friendly_reason(
                    qa_summary,
                    qa_results,
                    iteration,
                    request.max_iterations,
                    request,
                    evaluated_layers,
                )
                return {
                    "approved": False,
                    "final_rejection": True,  # This will trigger fallback logic in main loop
                    "reason": user_friendly_reason,
                    "deal_breaker_type": "majority_consensus",
                    "allow_fallback": True,
                }
            else:
                # Final rejection - we've reached max iterations and no fallback
                user_friendly_reason = create_user_friendly_reason(
                    qa_summary,
                    qa_results,
                    iteration,
                    request.max_iterations,
                    request,
                    evaluated_layers,
                )
                return {
                    "approved": False,
                    "final_rejection": True,
                    "reason": user_friendly_reason,
                    "deal_breaker_type": "majority_consensus",
                }
        else:
            # Force iteration - we still have iterations left
            user_friendly_reason = create_user_friendly_reason(
                qa_summary,
                qa_results,
                iteration,
                request.max_iterations,
                request,
                evaluated_layers,
            )
            return {
                "approved": False,
                "final_rejection": False,
                "reason": user_friendly_reason,
                "deal_breaker_type": "majority_consensus_retry",
            }

    # Step 2: Check for 50-50 ties first (exact 50% deal-breakers should escalate to Gran Sabio)
    tie_deal_breakers = _check_50_50_tie_deal_breakers(qa_results, request.qa_models)
    if tie_deal_breakers["has_50_50_ties"]:
        user_friendly_reason = create_user_friendly_reason(
            qa_summary,
            qa_results,
            iteration,
            request.max_iterations,
            request,
            evaluated_layers,
        )
        return {
            "approved": False,
            "final_rejection": False,
            "reason": f"Split decision detected (50-50 tie) - escalating to senior review: {user_friendly_reason}",
            "deal_breaker_type": "50_50_tie",
            "tie_deal_breakers": tie_deal_breakers,
        }

    # Step 3: Check for minority deal-breakers (will escalate to Gran Sabio)
    minority_deal_breakers = _check_minority_deal_breakers(qa_results, request.qa_models)
    if minority_deal_breakers["has_minority_deal_breakers"]:
        # Don't reject immediately - let it go to Gran Sabio for review
        user_friendly_reason = create_user_friendly_reason(
            qa_summary,
            qa_results,
            iteration,
            request.max_iterations,
            request,
            evaluated_layers,
        )
        return {
            "approved": False,
            "final_rejection": False,
            "reason": f"Quality concerns detected - escalating to senior review: {user_friendly_reason}",
            "deal_breaker_type": "minority_consensus",
            "minority_deal_breakers": minority_deal_breakers,
        }

    # Step 2: Evaluate each layer individually
    layer_results = {}
    mandatory_failures = []
    non_mandatory_failures = []

    layer_config_by_name: Dict[str, QALayer] = {}
    if evaluated_layers:
        for layer in evaluated_layers:
            if layer and layer.name not in layer_config_by_name:
                layer_config_by_name[layer.name] = layer
    else:
        for layer in request.qa_layers:
            if layer and layer.name not in layer_config_by_name:
                layer_config_by_name[layer.name] = layer

    for layer_name, layer in layer_config_by_name.items():
        layer_avg = consensus_result.layer_averages.get(layer_name, 0)
        layer_passed = layer_avg >= layer.min_score

        layer_results[layer_name] = {
            "passed": layer_passed,
            "score": layer_avg,
            "is_mandatory": layer.is_mandatory,
            "min_score": layer.min_score,
        }

        if not layer_passed:
            if layer.is_mandatory:
                mandatory_failures.append(f"{layer_name} ({layer_avg:.2f} < {layer.min_score})")
            else:
                non_mandatory_failures.append(f"{layer_name} ({layer_avg:.2f} < {layer.min_score})")

    # Step 3: Check if we're at max iterations and have mandatory failures
    if iteration >= request.max_iterations and mandatory_failures:
        # Create user-friendly reason for mandatory failures
        friendly_reason = (
            f"Content rejected after {request.max_iterations} attempts. "
            f"Critical quality requirements not met: {', '.join(mandatory_failures)}"
        )

        # Check if Gran Sabio fallback is enabled
        if hasattr(request, "gran_sabio_fallback") and request.gran_sabio_fallback:
            return {
                "approved": False,
                "final_rejection": True,  # This will trigger fallback logic in main loop
                "reason": friendly_reason,
                "allow_fallback": True,
            }
        else:
            return {
                "approved": False,
                "final_rejection": True,
                "reason": friendly_reason,
            }

    # Step 4: If all layers passed, approve immediately
    if not mandatory_failures and not non_mandatory_failures:
        return {
            "approved": True,
            "final_rejection": False,
            "reason": f"All quality layers passed (Average score: {consensus_result.average_score:.2f})",
        }

    # Step 5: If we have failures but not at max iterations, continue iterating
    if iteration < request.max_iterations and (mandatory_failures or non_mandatory_failures):
        all_failures = mandatory_failures + non_mandatory_failures
        return {
            "approved": False,
            "final_rejection": False,
            "reason": f"Retrying iteration {iteration}/{request.max_iterations}: Quality requirements not met in {', '.join(all_failures)}",
        }

    # Step 6: At max iterations, check global override for non-mandatory failures only
    if consensus_result.average_score >= request.min_global_score and not mandatory_failures:
        return {
            "approved": True,
            "final_rejection": False,
            "reason": (
                f"Content approved by global score override (Average: {consensus_result.average_score:.2f} >= "
                f"{request.min_global_score}). Minor issues: {', '.join(non_mandatory_failures)}"
            ),
        }

    # Step 7: Final rejection - global score insufficient
    all_failures = mandatory_failures + non_mandatory_failures

    # Check if Gran Sabio fallback is enabled
    if hasattr(request, "gran_sabio_fallback") and request.gran_sabio_fallback:
        return {
            "approved": False,
            "final_rejection": True,  # This will trigger fallback logic in main loop
            "reason": (
                f"Content rejected: Overall quality score insufficient ({consensus_result.average_score:.2f} < "
                f"{request.min_global_score}). Issues in: {', '.join(all_failures)}"
            ),
            "allow_fallback": True,
        }
    else:
        return {
            "approved": False,
            "final_rejection": True,
            "reason": (
                f"Content rejected: Overall quality score insufficient ({consensus_result.average_score:.2f} < "
                f"{request.min_global_score}). Issues in: {', '.join(all_failures)}"
            ),
        }


def _check_minority_deal_breakers(qa_results, qa_models) -> Dict[str, Any]:
    """
    Check for minority deal-breakers across all layers.
    """

    all_deal_breakers = []
    total_evaluations = 0
    total_models = len(qa_models) if qa_models else 0

    for layer_name, model_results in qa_results.items():
        layer_total = len(model_results)
        if layer_total == 0:
            continue

        layer_db_details = []
        for model, evaluation in model_results.items():
            if getattr(evaluation, "deal_breaker", False):
                layer_db_details.append(
                    {"layer": layer_name, "model": model, "reason": evaluation.deal_breaker_reason}
                )

        if not layer_db_details:
            continue

        db_count = len(layer_db_details)
        is_tie = layer_total % 2 == 0 and db_count * 2 == layer_total
        is_minority = db_count < (layer_total / 2)

        if is_tie or is_minority:
            all_deal_breakers.extend(layer_db_details)
            total_evaluations += layer_total

    has_minority = len(all_deal_breakers) > 0

    return {
        "has_minority_deal_breakers": has_minority,
        "deal_breaker_count": len(all_deal_breakers),
        "total_evaluations": total_evaluations,
        "total_models": total_models,
        "details": all_deal_breakers,
        "summary": f"{len(all_deal_breakers)} deal-breakers from {total_evaluations} evaluations",
    }


def _check_50_50_tie_deal_breakers(qa_results, qa_models) -> Dict[str, Any]:
    """
    Check for 50-50 tie deal-breakers across all layers that should escalate to Gran Sabio.
    Only applies when there's an even number of models and exactly 50% are deal-breakers.
    """

    tie_layers = []
    total_models = len(qa_models)

    # Only proceed if we have an even number of models (50-50 ties only possible with even numbers)
    if total_models % 2 != 0:
        return {
            "has_50_50_ties": False,
            "tie_layers": [],
            "total_models": total_models,
            "summary": "No 50-50 ties possible with odd number of models",
        }

    for layer_name, model_results in qa_results.items():
        deal_breaker_count = 0
        approve_count = 0
        layer_details = []

        for model, evaluation in model_results.items():
            if evaluation.deal_breaker:
                deal_breaker_count += 1
                layer_details.append(
                    {"model": model, "decision": "deal_breaker", "reason": evaluation.deal_breaker_reason}
                )
            else:
                approve_count += 1
                layer_details.append({"model": model, "decision": "approve", "score": evaluation.score})

        # Check if this layer has exactly 50% deal-breakers (perfect tie)
        if deal_breaker_count == total_models / 2 and approve_count == total_models / 2:
            tie_layers.append(
                {
                    "layer": layer_name,
                    "deal_breaker_count": deal_breaker_count,
                    "approve_count": approve_count,
                    "total_models": total_models,
                    "details": layer_details,
                }
            )

    has_ties = len(tie_layers) > 0

    return {
        "has_50_50_ties": has_ties,
        "tie_layers": tie_layers,
        "total_models": total_models,
        "summary": f"{len(tie_layers)} layers with 50-50 ties detected from {total_models} models",
    }
