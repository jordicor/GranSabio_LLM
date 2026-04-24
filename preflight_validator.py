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
from model_aliasing import ModelAliasRegistry, PromptPart
from models import ContentRequest, PreflightResult, PreflightIssue, WordCountAnalysis
from usage_tracking import UsageTracker
from word_count_utils import word_count_config_to_dict


logger = logging.getLogger(__name__)


def _normalise_model_name(value: Any) -> Optional[str]:
    """Return a usable model name, treating empty strings as unconfigured."""
    if value is None:
        return None
    model_name = str(value).strip()
    return model_name or None


def resolve_preflight_model(request: ContentRequest) -> Optional[str]:
    """Resolve the required LLM model for preflight validation."""
    configured_model = _normalise_model_name(getattr(config, "PREFLIGHT_VALIDATION_MODEL", None))
    if configured_model:
        return configured_model

    arbiter_model = _normalise_model_name(getattr(request, "arbiter_model", None))
    if arbiter_model:
        return arbiter_model

    return _normalise_model_name(getattr(request, "gran_sabio_model", None))


def _build_preflight_failure_result(
    *,
    code: str,
    message: str,
    summary: str,
    model: Optional[str] = None,
) -> PreflightResult:
    """Build a fail-closed preflight result when LLM validation cannot complete."""
    related = ["preflight"]
    if model:
        related.append(model)

    return PreflightResult(
        decision="reject",
        user_feedback=message,
        summary=summary,
        issues=[
            PreflightIssue(
                code=code,
                severity="critical",
                message=message,
                blockers=True,
                related_requirements=related,
            )
        ],
        word_count_analysis=None,
        enable_algorithmic_word_count=False,
        duplicate_word_count_layers_to_remove=[],
    )


def _normalize_qa_models(qa_models: Any) -> List[str]:
    """Convert QAModelConfig objects to model name strings for JSON serialization.

    orjson cannot serialize Pydantic models directly, so we extract just the
    model identifier when QAModelConfig objects are provided.

    Args:
        qa_models: List of model names (str) or QAModelConfig objects

    Returns:
        List of model name strings safe for JSON serialization
    """
    if not qa_models:
        return []
    result = []
    for m in qa_models:
        if isinstance(m, str):
            result.append(m)
        elif hasattr(m, 'model'):
            # QAModelConfig - extract just the model name
            result.append(m.model)
        else:
            result.append(str(m))
    return result


def _build_validation_payload(
    request: ContentRequest,
    context_documents: Optional[List[Dict[str, Any]]] = None,
    image_info: Optional[Dict[str, Any]] = None,
    model_alias_registry: Optional[ModelAliasRegistry] = None,
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

    qa_model_names = _normalize_qa_models(request.qa_models)
    if model_alias_registry:
        qa_evaluators = [model_alias_registry.qa_alias(index) for index, _ in enumerate(qa_model_names)]
    else:
        qa_evaluators = [f"Evaluator {chr(ord('A') + index)}" for index, _ in enumerate(qa_model_names)]

    payload: Dict[str, Any] = {
        "prompt": request.prompt,
        "content_type": request.content_type,
        "generator_role": "Generator",
        "qa_evaluators": qa_evaluators,
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

    # Accent-vs-AI-style-request conflict detection (Cambio 1 v5, §6).
    # Only the mode enum value is serialized - never the user-supplied criteria.
    guard = getattr(request, "llm_accent_guard", None)
    if guard is not None and getattr(guard, "mode", "off") != "off":
        payload["llm_accent_guard_mode"] = guard.mode

    return payload


def _build_prompt_safety_parts(payload: Dict[str, Any]) -> List[PromptPart]:
    """Classify preflight prompt fields by source for model-identity blinding."""
    user_supplied_keys = {
        "prompt",
        "content_type",
        "source_text",
        "qa_layers",
        "context_documents",
        "context_documents_total_bytes",
        "images",
        "word_count",
        "phrase_frequency",
        "lexical_diversity",
    }
    system_payload = {
        key: value
        for key, value in payload.items()
        if key not in user_supplied_keys
    }
    user_payload = {
        key: value
        for key, value in payload.items()
        if key in user_supplied_keys
    }

    parts = [
        PromptPart(
            text=json.dumps(system_payload, ensure_ascii=True, indent=2),
            source="system_generated",
            label="preflight.validation_payload.system",
        )
    ]
    if user_payload:
        parts.append(
            PromptPart(
                text=json.dumps(user_payload, ensure_ascii=True, indent=2),
                source="user_supplied",
                label="preflight.validation_payload.user",
            )
        )
    return parts


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

        "LLM ACCENT GUARD CONFLICT DETECTION: "
        "If the payload includes 'llm_accent_guard_mode' with any value other than 'off' AND "
        "the user prompt explicitly requests AI-style, 'slop', or formulaic-AI content as the primary "
        "deliverable (NOT merely as illustrative examples embedded inside a larger legitimate deliverable), "
        "REJECT with decision='reject'. Provide user_feedback in the user's own language that (a) identifies "
        "the conflict between the accent guard and the explicit AI-style request, and (b) suggests disabling "
        "one of the two (either llm_accent_guard or the explicit AI-style request). "
        "Ambiguous cases (e.g., 'explain AI patterns with short examples') should PROCEED. "

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
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def _validate_evidence_grounding_config(request: ContentRequest) -> Optional[PreflightIssue]:
    """
    Validate evidence grounding configuration before generation.

    Checks that the configured model supports logprobs, which is required
    for the budget scoring phase of evidence grounding verification.

    Args:
        request: Content request to validate

    Returns:
        PreflightIssue if validation fails, None if valid or not enabled
    """
    # Check if evidence grounding is enabled
    if not request.evidence_grounding or not request.evidence_grounding.enabled:
        return None

    # Get the scoring model (requires logprobs validation)
    # If user specifies a model, it's used for both extraction and scoring
    eg_scoring_model = request.evidence_grounding.model or config.EVIDENCE_GROUNDING_SCORING_MODEL

    if not eg_scoring_model:
        return PreflightIssue(
            code="evidence_grounding_no_model",
            severity="critical",
            message="Evidence grounding is enabled but no scoring model is configured. "
                    "Set EVIDENCE_GROUNDING_SCORING_MODEL in config or specify model in request.",
            blockers=True,
            related_requirements=["evidence_grounding"],
        )

    # Check model info
    model_info = config.get_model_info(eg_scoring_model)
    if not model_info:
        return PreflightIssue(
            code="evidence_grounding_unknown_model",
            severity="critical",
            message=f"Evidence grounding model '{eg_scoring_model}' is not recognized. "
                    f"Please use a valid OpenAI model that supports logprobs.",
            blockers=True,
            related_requirements=["evidence_grounding"],
        )

    provider = model_info.get("provider", "")

    # Only OpenAI models support logprobs
    if provider != "openai":
        return PreflightIssue(
            code="evidence_grounding_no_logprobs",
            severity="critical",
            message=f"Evidence grounding model '{eg_scoring_model}' (provider: {provider}) does not support logprobs. "
                    f"Only OpenAI models support the logprob verification required for evidence grounding. "
                    f"Please use gpt-4o-mini, gpt-5-nano, or similar OpenAI models.",
            blockers=True,
            related_requirements=["evidence_grounding"],
        )

    # Check for reasoning models that don't support logprobs (o1, o3)
    model_id = model_info.get("model_id", eg_scoring_model).lower()
    reasoning_markers = ["o1-", "o1_", "o3-", "o3_", "/o1", "/o3"]
    is_reasoning_model = any(marker in model_id for marker in reasoning_markers)

    # Also check for standalone "o1" or "o3" but not "gpt-4o" patterns
    if not is_reasoning_model:
        # Check for models like "o1", "o1-mini", "o3", "o3-mini" but not "gpt-4o"
        parts = model_id.replace("-", "_").split("_")
        if parts and parts[0] in ("o1", "o3"):
            is_reasoning_model = True

    if is_reasoning_model:
        return PreflightIssue(
            code="evidence_grounding_reasoning_model",
            severity="critical",
            message=f"Evidence grounding model '{eg_scoring_model}' is a reasoning model that does not expose logprobs. "
                    f"Reasoning models (o1, o1-mini, o3, o3-mini) cannot be used for evidence grounding verification. "
                    f"Please use gpt-4o-mini, gpt-5-nano, or similar models instead.",
            blockers=True,
            related_requirements=["evidence_grounding"],
        )

    # Valid configuration
    logger.debug(f"Evidence grounding model '{eg_scoring_model}' validated successfully")
    return None


async def run_preflight_validation(
    ai_service,
    request: ContentRequest,
    context_documents: Optional[List[Dict[str, Any]]] = None,
    image_info: Optional[Dict[str, Any]] = None,
    stream_callback: Optional[callable] = None,
    usage_tracker: Optional[UsageTracker] = None,
    phase_logger: Optional["PhaseLogger"] = None,
    model_alias_registry: Optional[ModelAliasRegistry] = None,
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
    selected_model = resolve_preflight_model(request)
    if not selected_model:
        message = (
            "Preflight validation could not start because no preflight, Arbiter, "
            "or GranSabio model is configured."
        )
        logger.error(message)
        return _build_preflight_failure_result(
            code="preflight_model_unavailable",
            message=message,
            summary="Preflight model unavailable",
        )

    # Early validation: Check evidence grounding model compatibility (no API call needed)
    evidence_grounding_issue = _validate_evidence_grounding_config(request)
    if evidence_grounding_issue:
        logger.warning(f"Evidence grounding validation failed: {evidence_grounding_issue.message}")
        return PreflightResult(
            decision="reject",
            user_feedback=evidence_grounding_issue.message,
            summary="Evidence grounding configuration invalid",
            issues=[evidence_grounding_issue],
            word_count_analysis=None,
            enable_algorithmic_word_count=False,
            duplicate_word_count_layers_to_remove=[],
        )

    payload = _build_validation_payload(
        request,
        context_documents=context_documents,
        image_info=image_info,
        model_alias_registry=model_alias_registry,
    )
    validator_prompt = _build_validator_prompt(payload)
    prompt_safety_parts = _build_prompt_safety_parts(payload)

    try:
        if phase_logger:
            phase_logger.info(f"Running preflight validation with model: {selected_model}")
        else:
            logger.info(f"Running preflight validation with model: {selected_model}")
            logger.debug(f"Preflight prompt: {validator_prompt[:500]}...")

        # Log full prompt if extra_verbose is enabled via phase_logger
        if phase_logger:
            phase_logger.log_prompt(
                model=selected_model,
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
                    "model": selected_model,
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
                model=selected_model,
                temperature=0.0,
                max_tokens=800,
                system_prompt=config.PREFLIGHT_SYSTEM_PROMPT,
                extra_verbose=False,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                model_alias_registry=model_alias_registry,
                prompt_safety_parts=prompt_safety_parts,
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
                model=selected_model,
                temperature=0.0,
                max_tokens=800,
                system_prompt=config.PREFLIGHT_SYSTEM_PROMPT,
                extra_verbose=False,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                model_alias_registry=model_alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            )

        # Log response if phase_logger is available
        if phase_logger:
            phase_logger.log_response(
                model=selected_model,
                response=raw_output
            )
        else:
            logger.info(f"Preflight validator raw output: {raw_output}")

    except Exception as exc:
        logger.error("Preflight validation request failed: %s", exc, exc_info=True)
        return _build_preflight_failure_result(
            code="preflight_llm_call_failed",
            message=(
                f"Preflight validation could not be completed with model '{selected_model}': "
                f"{type(exc).__name__}: {exc}"
            ),
            summary="Preflight LLM call failed",
            model=selected_model,
        )

    parsed = _parse_validator_response(raw_output)
    if not parsed:
        logger.warning("Preflight validator returned unparseable output: %s", raw_output)
        return _build_preflight_failure_result(
            code="preflight_invalid_response",
            message=(
                f"Preflight validation failed because model '{selected_model}' returned "
                "an unparseable response."
            ),
            summary="Preflight response unparseable",
            model=selected_model,
        )

    decision_raw = parsed.get("decision")
    decision = str(decision_raw).strip().lower() if decision_raw is not None else ""
    if decision not in {"proceed", "reject"}:
        logger.warning("Preflight validator returned invalid decision: %s", decision_raw)
        return _build_preflight_failure_result(
            code="preflight_invalid_decision",
            message=(
                f"Preflight validation failed because model '{selected_model}' returned "
                f"an invalid decision: {decision_raw!r}."
            ),
            summary="Preflight decision invalid",
            model=selected_model,
        )

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

    llm_recommended_removals = (
        word_count_analysis.recommended_removals
        if word_count_analysis is not None
        else []
    )

    result = PreflightResult(
        decision=decision,
        user_feedback=str(user_feedback),
        summary=str(summary) if summary is not None else None,
        issues=issues,
        confidence=confidence,
        word_count_analysis=word_count_analysis,
        enable_algorithmic_word_count=bool(llm_recommended_removals),
        duplicate_word_count_layers_to_remove=llm_recommended_removals,
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
