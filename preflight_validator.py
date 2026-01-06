"""
Preflight validation utilities for the Gran Sabio LLM Engine.

Runs a lightweight feasibility analysis before starting expensive
content generation iterations.
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Use optimized JSON helper to align with rest of codebase
import json_utils as json

if TYPE_CHECKING:
    from logging_utils import PhaseLogger

from ai_service import StreamChunk
from config import config
from models import ContentRequest, PreflightResult, PreflightIssue, WordCountAnalysis
from usage_tracking import UsageTracker
from word_count_utils import (
    is_word_count_enforcement_enabled,
    word_count_config_to_dict,
)


logger = logging.getLogger(__name__)


def _build_validation_payload(
    request: ContentRequest,
    context_documents: Optional[List[Dict[str, Any]]] = None,
    image_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect the relevant pieces of the request for the validator.

    Args:
        request: The content generation request.
        context_documents: Optional list of context document summaries.
        image_info: Optional dict with image metadata for vision-enabled requests:
            - count: Number of images
            - total_estimated_tokens: Sum of estimated tokens for all images
            - filenames: List of original filenames
            - total_size_bytes: Total size of all images
            - generator_supports_vision: Whether the generator model supports vision
            - detail_levels: List of detail levels used
    """
    qa_layers: List[Dict[str, Any]] = []
    for layer in request.qa_layers:
        if hasattr(layer, "model_dump"):
            layer_data = layer.model_dump(exclude_none=True)
        elif isinstance(layer, dict):
            layer_data = dict(layer)
        else:
            continue

        qa_layers.append({
            "name": layer_data.get("name"),
            "description": layer_data.get("description"),
            "criteria": layer_data.get("criteria"),
            "min_score": layer_data.get("min_score"),
            "is_mandatory": layer_data.get("is_mandatory") or layer_data.get("is_deal_breaker", False),
            "deal_breaker_criteria": layer_data.get("deal_breaker_criteria"),
            "order": layer_data.get("order"),
        })

    word_count_info: Dict[str, Any] = {}
    if request.min_words is not None:
        word_count_info["min_words"] = request.min_words
    if request.max_words is not None:
        word_count_info["max_words"] = request.max_words
    if request.word_count_enforcement:
        normalised_config = word_count_config_to_dict(request.word_count_enforcement)
        if normalised_config is not None:
            word_count_info["word_count_enforcement"] = normalised_config
        else:
            logger.warning(
                "Preflight payload: unable to normalise word_count_enforcement; using string representation"
            )
            word_count_info["word_count_enforcement"] = str(request.word_count_enforcement)

    phrase_frequency_info: Optional[Dict[str, Any]] = None
    if request.phrase_frequency:
        try:
            phrase_frequency_info = request.phrase_frequency.model_dump(exclude_none=True)
        except AttributeError:
            phrase_frequency_info = dict(request.phrase_frequency)

    lexical_diversity_info: Optional[Dict[str, Any]] = None
    if request.lexical_diversity:
        try:
            lexical_diversity_info = request.lexical_diversity.model_dump(exclude_none=True)
        except AttributeError:
            lexical_diversity_info = dict(request.lexical_diversity)

    payload: Dict[str, Any] = {
        "prompt": request.prompt,
        "content_type": request.content_type,
        "generator_model": request.generator_model,
        "qa_models": request.qa_models,
        "qa_layers": qa_layers,
        "min_global_score": request.min_global_score,
        "max_iterations": request.max_iterations,
        "gran_sabio_fallback": request.gran_sabio_fallback,
    }

    # Include source_text for QA validation if provided
    if hasattr(request, 'source_text') and request.source_text:
        payload["source_text"] = request.source_text


    if context_documents:
        payload["context_documents"] = context_documents
        payload["context_documents_total_bytes"] = sum(item.get("size_bytes", 0) for item in context_documents)
    if image_info:
        payload["images"] = image_info
    if word_count_info:
        payload["word_count"] = word_count_info
    if phrase_frequency_info:
        payload["phrase_frequency"] = phrase_frequency_info
    if lexical_diversity_info:
        payload["lexical_diversity"] = lexical_diversity_info

    return payload


def _build_validator_prompt(payload: Dict[str, Any]) -> str:
    """Create the instruction payload for the validator model."""
    schema_hint = {
        "decision": "proceed | reject",
        "summary": "ONLY when decision='proceed': use 'OK' or 'Pass'. When decision='reject': provide detailed summary",
        "user_feedback": "ONLY when decision='proceed': use 'Request approved' or similar (max 5 words). When decision='reject': provide actionable detailed feedback",
        "issues": [
            {
                "code": "short_identifier",
                "severity": "critical | warning | info",
                "message": "Human readable explanation",
                "blockers": True,
                "related_requirements": ["QA layer name or constraint"],
            }
        ],
        "word_count_analysis": {
            "conflicting_layers": ["layer_name1", "layer_name2"],
            "recommended_removals": ["layer_name1"],
            "analysis_reason": "ONLY when conflicts found: brief reason (max 10 words). Otherwise: 'None'"
        },
        "confidence": "Optional number between 0 and 1",
    }

    instructions = (
        "You will receive a JSON blob describing an upcoming editorial generation request. "
        "Decide whether the request can reasonably pass the QA contract before any content generation runs. "
        "Reject requests that contain contradictions, logical impossibilities, or violate mandatory QA rules regardless of creative effort. "

        "TOKEN OPTIMIZATION: "
        "When decision='proceed', use MINIMAL text in summary/user_feedback/analysis_reason (e.g., 'OK', 'Pass', 'Request approved'). "
        "Detailed explanations are ONLY needed when decision='reject'. "

        "WORD COUNT CONFLICT ANALYSIS: "
        "When the request specifies word limits (min_words, max_words), the system applies AUTOMATIC algorithmic word count enforcement. "
        "If QA layers also evaluate word count, length, or size, this creates REDUNDANT double validation that can cause false rejections. "
        "You MUST identify and recommend removing any QA layers that validate word count, length, brevity, size, word limits, "
        "character count, or any size-related criteria when algorithmic word count enforcement is active. "
        "IMPORTANT: After identifying redundant layers for removal, the request should PROCEED (not be rejected) as the conflict will be resolved by removing the redundant layers. "

        "VISION/IMAGE VALIDATION: "
        "If the request includes an 'images' block (images.count > 0): "
        "1. Check 'generator_supports_vision' field. If FALSE, REJECT with code 'vision_not_supported' - the generator model cannot process images. "
        "2. Check for contradictions: if QA layers require 'text-only' analysis or forbid visual references, but images are provided, flag as warning. "
        "3. If prompt explicitly mentions analyzing/describing images but no images are provided, issue a warning (not rejection). "
        "4. Vision-enabled requests are valid for all content types - images can provide context for biographies, articles, scripts, etc. "

        "If no blocking issue exists, respond with decision 'proceed'. "
        "Always return STRICT JSON matching the described schema. Do not include code fences or commentary."
    )

    prompt = (
        f"Instructions:\n{instructions}\n\n"
        f"Output schema (for reference):\n{json.dumps(schema_hint, ensure_ascii=True, indent=2)}\n\n"
        f"Request payload:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )

    return prompt


def _extract_json_blob(raw_output: str) -> Optional[str]:
    """Attempt to isolate a JSON object from the model response."""
    if not raw_output:
        return None

    candidate = raw_output.strip()

    if candidate.startswith("```"):
        # Remove optional markdown fences
        stripped = candidate
        if stripped.lower().startswith("```json"):
            stripped = stripped[7:]
        else:
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        candidate = stripped.strip()

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and start < end:
        possible = candidate[start:end + 1]
        try:
            json.loads(possible)
            return possible
        except json.JSONDecodeError:
            return None

    return None


def _normalise_issues(raw_issues: Any) -> List[PreflightIssue]:
    """Convert raw issues into PreflightIssue objects."""
    issues: List[PreflightIssue] = []
    if not isinstance(raw_issues, list):
        return issues

    for item in raw_issues:
        if not isinstance(item, dict):
            continue

        message = item.get("message") or item.get("details") or item.get("explanation")
        if not message:
            continue

        code = item.get("code") or item.get("type") or "unspecified"
        severity = str(item.get("severity") or "critical").lower()
        if severity not in {"critical", "warning", "info"}:
            severity = "critical" if item.get("blockers", True) else "warning"

        blockers = item.get("blockers")
        if blockers is None:
            blockers = severity == "critical"

        related = item.get("related_requirements") or item.get("requirements") or item.get("qa_layers")
        related_names: List[str] = []
        if isinstance(related, list):
            for value in related:
                if isinstance(value, (str, int, float)):
                    related_names.append(str(value))
        elif isinstance(related, (str, int, float)):
            related_names.append(str(related))

        issues.append(
            PreflightIssue(
                code=str(code),
                severity=severity,
                message=str(message),
                blockers=bool(blockers),
                related_requirements=related_names,
            )
        )

    return issues


def _parse_validator_response(raw_output: str) -> Optional[Dict[str, Any]]:
    """Parse the validator JSON response."""
    json_blob = _extract_json_blob(raw_output)
    if not json_blob:
        return None

    try:
        return json.loads(json_blob)
    except json.JSONDecodeError:
        return None


def _analyze_word_count_conflicts(request: ContentRequest) -> Dict[str, Any]:
    """
    Analyze QA layers for word count conflicts and determine optimization strategy.

    This function identifies scenarios where algorithmic word count enforcement
    would conflict with manual AI evaluation layers, causing unnecessary iterations
    and false rejections.

    Args:
        request: Content request to analyze

    Returns:
        Dictionary with optimization recommendations
    """
    # Check if we have word limits that would trigger algorithmic enforcement
    has_word_limits = request.min_words is not None or request.max_words is not None
    has_word_count_enforcement = is_word_count_enforcement_enabled(request.word_count_enforcement)

    # If no word limits or enforcement, no conflicts possible
    if not has_word_limits or not has_word_count_enforcement:
        return {
            "enable_algorithmic_word_count": False,
            "duplicate_layers_to_remove": [],
            "analysis_reason": "No word count enforcement configured"
        }

    # Algorithmic word count is ALWAYS enabled when word count enforcement is configured
    # The severity (deal_breaker vs important) only affects whether violations trigger deal_breaker

    # Identify QA layers that might conflict with algorithmic word counting
    # These are layers that evaluate word count, length, budget adherence, etc.
    word_count_related_keywords = [
        "word count", "word budget", "length", "brevity", "extensión",
        "longitud", "palabras", "budget adherence", "word limit",
        "count adherence", "length adherence", "size", "tamaño"
    ]

    duplicate_layers_to_remove = []
    conflicting_layers_found = []

    for layer in request.qa_layers:
        layer_name_lower = layer.name.lower()
        layer_desc_lower = layer.description.lower() if layer.description else ""
        layer_criteria_lower = layer.criteria.lower() if layer.criteria else ""

        # Check if this layer is related to word counting
        is_word_count_related = any(
            keyword in layer_name_lower or
            keyword in layer_desc_lower or
            keyword in layer_criteria_lower
            for keyword in word_count_related_keywords
        )

        if is_word_count_related:
            conflicting_layers_found.append(layer.name)
            # ALWAYS remove word count related layers - algorithmic enforcement handles this
            duplicate_layers_to_remove.append(layer.name)

    # Decision logic: Enable algorithmic enforcement if we found conflicting layers
    if conflicting_layers_found:
        analysis_reason = f"Found {len(conflicting_layers_found)} word count-related QA layers that may conflict with algorithmic enforcement: {', '.join(conflicting_layers_found)}"
        if duplicate_layers_to_remove:
            analysis_reason += f". Recommending removal of {len(duplicate_layers_to_remove)} layers with deal-breaker potential to prevent conflicts."
    else:
        analysis_reason = "No conflicting word count-related QA layers found"

    return {
        "enable_algorithmic_word_count": len(conflicting_layers_found) > 0,
        "duplicate_layers_to_remove": duplicate_layers_to_remove,
        "analysis_reason": analysis_reason,
        "conflicting_layers_found": conflicting_layers_found
    }


async def run_preflight_validation(
    ai_service,
    request: ContentRequest,
    context_documents: Optional[List[Dict[str, Any]]] = None,
    image_info: Optional[Dict[str, Any]] = None,
    stream_callback: Optional[callable] = None,
    usage_tracker: Optional[UsageTracker] = None,
    phase_logger: Optional["PhaseLogger"] = None,
) -> PreflightResult:
    """Run the preflight validator and return a structured decision.

    Args:
        ai_service: The AI service instance for generation.
        request: The content generation request to validate.
        context_documents: Optional list of context document summaries.
        image_info: Optional dict with image metadata for vision-enabled requests.
        stream_callback: Optional callback for streaming validation output.
        usage_tracker: Optional usage tracker for cost tracking.
        phase_logger: Optional phase logger for structured logging.

    Returns:
        PreflightResult with decision (proceed/reject) and validation feedback.
    """
    if not getattr(config, "PREFLIGHT_VALIDATION_MODEL", None):
        heuristic_analysis = _analyze_word_count_conflicts(request)
        return PreflightResult(
            decision="proceed",
            user_feedback="Preflight validator disabled by configuration.",
            summary="Preflight skipped (no model configured).",
            issues=[],
            word_count_analysis=None,
            enable_algorithmic_word_count=heuristic_analysis["enable_algorithmic_word_count"],
            duplicate_word_count_layers_to_remove=heuristic_analysis["duplicate_layers_to_remove"],
        )

    payload = _build_validation_payload(
        request,
        context_documents=context_documents,
        image_info=image_info,
    )
    validator_prompt = _build_validator_prompt(payload)

    try:
        if phase_logger:
            phase_logger.info(f"Running preflight validation with model: {config.PREFLIGHT_VALIDATION_MODEL}")
        else:
            logger.info(f"Running preflight validation with model: {config.PREFLIGHT_VALIDATION_MODEL}")
            logger.debug(f"Preflight prompt: {validator_prompt[:500]}...")

        # Log full prompt if extra_verbose is enabled via phase_logger
        if phase_logger:
            phase_logger.log_prompt(
                model=config.PREFLIGHT_VALIDATION_MODEL,
                system_prompt=config.PREFLIGHT_SYSTEM_PROMPT,
                user_prompt=validator_prompt,
                temperature=0.0,
                max_tokens=800
            )

        usage_callback = (
            usage_tracker.create_callback(
                phase="preflight",
                role="validator",
                operation="preflight_validation",
                metadata={
                    "context_documents": len(context_documents or []),
                },
            )
            if usage_tracker and usage_tracker.enabled
            else None
        )

        if stream_callback:
            # Use streaming generation
            accumulated_content = ""
            async for chunk in ai_service.generate_content_stream(
                prompt=validator_prompt,
                model=config.PREFLIGHT_VALIDATION_MODEL,
                temperature=0.0,
                max_tokens=800,
                system_prompt=config.PREFLIGHT_SYSTEM_PROMPT,
                extra_verbose=False,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
            ):
                # Handle StreamChunk (Claude with thinking) vs plain string
                if isinstance(chunk, StreamChunk):
                    chunk_text = chunk.text
                    is_thinking = chunk.is_thinking
                else:
                    chunk_text = chunk
                    is_thinking = False
                # Only accumulate non-thinking for final output
                if not is_thinking:
                    accumulated_content += chunk_text
                # Stream all (including thinking) for live monitoring
                if stream_callback and chunk_text:
                    stream_callback(chunk_text)

            raw_output = accumulated_content
        else:
            # Use non-streaming generation (original behavior)
            raw_output = await ai_service.generate_content(
                prompt=validator_prompt,
                model=config.PREFLIGHT_VALIDATION_MODEL,
                temperature=0.0,
                max_tokens=800,
                system_prompt=config.PREFLIGHT_SYSTEM_PROMPT,
                extra_verbose=False,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
            )

        # Log response if phase_logger is available
        if phase_logger:
            phase_logger.log_response(
                model=config.PREFLIGHT_VALIDATION_MODEL,
                response=raw_output
            )
        else:
            logger.info(f"Preflight validator raw output: {raw_output}")

    except Exception as exc:
        logger.error("Preflight validation request failed: %s", exc, exc_info=True)
        heuristic_analysis = _analyze_word_count_conflicts(request)
        return PreflightResult(
            decision="proceed",
            user_feedback="Preflight validator unavailable. Proceeding with generation.",
            summary="Preflight request failed",
            issues=[],
            word_count_analysis=None,
            enable_algorithmic_word_count=heuristic_analysis["enable_algorithmic_word_count"],
            duplicate_word_count_layers_to_remove=heuristic_analysis["duplicate_layers_to_remove"],
        )

    parsed = _parse_validator_response(raw_output)
    if not parsed:
        logger.warning("Preflight validator returned unparseable output: %s", raw_output)
        heuristic_analysis = _analyze_word_count_conflicts(request)
        return PreflightResult(
            decision="proceed",
            user_feedback="Preflight validator returned unexpected output. Proceeding with generation.",
            summary="Preflight response unparseable",
            issues=[],
            word_count_analysis=None,
            enable_algorithmic_word_count=heuristic_analysis["enable_algorithmic_word_count"],
            duplicate_word_count_layers_to_remove=heuristic_analysis["duplicate_layers_to_remove"],
        )

    decision = str(parsed.get("decision", "proceed")).strip().lower()
    if decision not in {"proceed", "reject"}:
        decision = "proceed"

    issues = _normalise_issues(parsed.get("issues"))

    user_feedback = parsed.get("user_feedback") or parsed.get("feedback")
    if not user_feedback:
        user_feedback = "Preflight validation completed."

    summary = parsed.get("summary") or parsed.get("explanation")

    confidence: Optional[float] = None
    confidence_raw = parsed.get("confidence")
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    elif isinstance(confidence_raw, str):
        try:
            confidence = float(confidence_raw)
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = None

    # Parse AI word count analysis
    word_count_analysis: Optional[WordCountAnalysis] = None
    wc_analysis_raw = parsed.get("word_count_analysis")
    if wc_analysis_raw and isinstance(wc_analysis_raw, dict):
        conflicting_layers = wc_analysis_raw.get("conflicting_layers", [])
        recommended_removals = wc_analysis_raw.get("recommended_removals", [])
        analysis_reason = wc_analysis_raw.get("analysis_reason", "No analysis provided")

        if not isinstance(conflicting_layers, list):
            conflicting_layers = []
        if not isinstance(recommended_removals, list):
            recommended_removals = []

        word_count_analysis = WordCountAnalysis(
            conflicting_layers=conflicting_layers,
            recommended_removals=recommended_removals,
            analysis_reason=str(analysis_reason)
        )

        # Remove recommended QA layers from request
        if recommended_removals:
            original_layers = len(request.qa_layers)
            request.qa_layers = [
                layer for layer in request.qa_layers
                if layer.name not in recommended_removals
            ]
            removed_count = original_layers - len(request.qa_layers)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} conflicting QA layers: {recommended_removals}")

    # Fallback to heuristic analysis if AI didn't provide word count analysis
    heuristic_analysis = _analyze_word_count_conflicts(request)

    result = PreflightResult(
        decision=decision,
        user_feedback=str(user_feedback),
        summary=str(summary) if summary is not None else None,
        issues=issues,
        confidence=confidence,
        word_count_analysis=word_count_analysis,
        enable_algorithmic_word_count=heuristic_analysis["enable_algorithmic_word_count"],
        duplicate_word_count_layers_to_remove=heuristic_analysis["duplicate_layers_to_remove"],
    )

    # Log decision using phase_logger if available
    if phase_logger:
        phase_logger.log_decision(
            decision=decision.upper(),
            reason=result.summary or result.user_feedback
        )
    elif decision != "proceed":
        logger.info("Preflight validator rejected request: %s", result.summary or result.user_feedback)

    # Log AI word count analysis if provided
    if word_count_analysis and word_count_analysis.recommended_removals:
        log_msg = f"Preflight AI word count analysis: {word_count_analysis.analysis_reason}"
        if phase_logger:
            phase_logger.info(log_msg)
        else:
            logger.info(log_msg)

    return result
