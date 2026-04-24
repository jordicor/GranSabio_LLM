"""
Gran Sabio Module for Gran Sabio LLM Engine
============================================

The final escalation system that resolves conflicts and makes ultimate
decisions when the standard QA process cannot reach consensus.
"""

import asyncio
import logging
import re
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from logging_utils import PhaseLogger

from ai_service import AIService, AIRequestError, get_ai_service, StreamChunk
from deterministic_validation import DraftValidationResult, validate_generation_candidate
from models import ContentRequest, GranSabioResult, is_json_output_requested
from model_aliasing import PromptPart, get_evaluator_alias
from tools.ai_json_cleanroom import make_loose_json_validate_options
from tool_loop_models import (
    JsonContractError,
    LoopScope,
    OutputContract,
    PayloadScope,
    ToolLoopEnvelope,
    parse_json_with_markdown_fences,
)
from usage_tracking import UsageTracker
from config import config, get_default_models
from validation_context_factory import MeasurementRequest, build_measurement_request_for_layer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON schemas for the three live GranSabio methods (Phase 4, §3.4.3)
# ---------------------------------------------------------------------------


GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason", "score", "modifications_made", "final_content"],
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVED", "REJECTED"]},
        "reason": {"type": "string"},
        "score": {"type": "number"},
        "modifications_made": {"type": "boolean"},
        "final_content": {"type": ["string", "null"]},
    },
}


GRAN_SABIO_ESCALATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason", "score", "modifications_made", "final_content"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["APPROVE", "APPROVE_WITH_MODIFICATIONS", "REJECT"],
        },
        "reason": {"type": "string"},
        "score": {"type": "number"},
        "modifications_made": {"type": "boolean"},
        "final_content": {"type": ["string", "null"]},
    },
}


# ---------------------------------------------------------------------------
# Tool-loop activation helper (§3.4.3 + §3.4)
# ---------------------------------------------------------------------------


def _should_use_gransabio_tools(request: ContentRequest, model: str) -> bool:
    """Decide whether GranSabio should route through the shared tool loop.

    GranSabio always ships measurable constraints (the three live methods use
    JSON_STRUCTURED with strict schemas, plus optional deterministic
    measurement on the content under review), so the gate only inspects the
    request-level mode flag and provider support — analogous to
    ``_should_use_generation_tools`` but without the measurable-validator
    branch, per §3.4.3.
    """

    tools_mode = getattr(request, "gransabio_tools_mode", "auto")
    if tools_mode == "never":
        return False

    try:
        model_info = config.get_model_info(model)
    except Exception:
        return False

    provider = model_info.get("provider")
    model_id = str(model_info.get("model_id", "")).lower()
    provider_key = (provider or "").lower()
    supported_providers = {
        "openai", "claude", "anthropic", "gemini", "google", "xai", "openrouter",
    }
    if provider_key not in supported_providers:
        return False

    if provider_key == "openai" and AIService._is_openai_responses_api_model(model_id):
        return False

    return True


_GRAN_SABIO_MEASUREMENT_LAYER = SimpleNamespace(name="Gran Sabio measurement")


def _build_gran_sabio_measurement_request(request: Any) -> MeasurementRequest:
    """Return a measurement-only request for Gran Sabio validate_draft calls."""

    measurement_request = build_measurement_request_for_layer(
        request,
        _GRAN_SABIO_MEASUREMENT_LAYER,
    )
    if measurement_request is None:
        measurement_request = MeasurementRequest()

    json_expectations = getattr(request, "json_expectations", None)
    if json_expectations is not None and is_json_output_requested(measurement_request):
        # The shared factory intentionally avoids copying the whole request; this
        # JSON-contract field is safe and needed for measurement validation.
        setattr(measurement_request, "json_expectations", json_expectations)

    return measurement_request


# ---------------------------------------------------------------------------
# Parse helpers for JSON_STRUCTURED envelope payloads
# ---------------------------------------------------------------------------


class _GranSabioPayloadError(ValueError):
    """Raised when a JSON_STRUCTURED envelope payload violates invariants.

    The tool loop already enforces schema shape; this covers the cross-field
    invariant that ``modifications_made=true`` requires a non-empty
    ``final_content`` string (matrix row 1 fail-fast — §3.4.3).
    """


def _parse_minority_payload(payload: Dict[str, Any]) -> Tuple[str, float, str, bool, Optional[str]]:
    """Extract fields from a ``GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA`` payload."""

    decision = str(payload["decision"]).upper()
    score = float(payload["score"])
    reason = str(payload.get("reason") or "")
    modifications_made = bool(payload["modifications_made"])
    raw_final = payload.get("final_content")
    final_content = raw_final if isinstance(raw_final, str) else None
    if modifications_made and not (final_content and final_content.strip()):
        raise _GranSabioPayloadError(
            "modifications_made=true requires a non-empty final_content "
            "(minority override schema, matrix row 1)."
        )
    return decision, score, reason, modifications_made, final_content


def _parse_escalation_payload(payload: Dict[str, Any]) -> Tuple[str, float, str, bool, Optional[str]]:
    """Extract fields from a ``GRAN_SABIO_ESCALATION_SCHEMA`` payload."""

    decision = str(payload["decision"]).upper()
    score = float(payload["score"])
    reason = str(payload.get("reason") or "")
    modifications_made = bool(payload["modifications_made"])
    raw_final = payload.get("final_content")
    final_content = raw_final if isinstance(raw_final, str) else None
    if modifications_made and not (final_content and final_content.strip()):
        raise _GranSabioPayloadError(
            "modifications_made=true requires a non-empty final_content "
            "(escalation schema, matrix row 1)."
        )
    return decision, score, reason, modifications_made, final_content


# --- Streaming Retry Helpers ---

def _is_retryable_streaming_error(exc: Exception) -> bool:
    """Determine if a streaming error should trigger a retry."""
    if isinstance(exc, AIRequestError):
        return True

    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status in {408, 425, 429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_markers = [
        "timeout", "temporarily unavailable", "internal server error",
        "gateway", "rate limit", "overloaded", "unavailable",
        "service unavailable", "connection reset", "connection refused",
        "api_error"
    ]
    return any(marker in message for marker in transient_markers)


def _extract_error_reason(exc: Exception) -> str:
    """Extract a human-readable error reason from an exception."""
    if isinstance(exc, AIRequestError):
        cause = exc.cause
        if hasattr(cause, 'message'):
            return str(cause.message)
        return str(cause)

    exc_str = str(exc)
    if "'message':" in exc_str:
        try:
            match = re.search(r"'message':\s*'([^']+)'", exc_str)
            if match:
                return match.group(1)
        except Exception:
            pass

    return str(exc)[:200]


def _extract_provider(exc: Exception) -> Optional[str]:
    """Extract provider name from an exception if available."""
    if isinstance(exc, AIRequestError):
        return getattr(exc, "provider", None)

    exc_str = str(exc).lower()
    if "anthropic" in exc_str or "claude" in exc_str:
        return "anthropic"
    if "openai" in exc_str or "gpt" in exc_str:
        return "openai"
    if "gemini" in exc_str or "google" in exc_str:
        return "google"
    if "xai" in exc_str or "grok" in exc_str:
        return "xai"

    return None


class GranSabioInvocationError(RuntimeError):
    """Raised when Gran Sabio cannot complete an escalation or regeneration."""


class GranSabioProcessCancelled(Exception):
    """Raised when a Gran Sabio operation is cancelled by user request."""


CancelCallback = Optional[Callable[[], Awaitable[bool]]]


class GranSabioEngine:
    """Gran Sabio - Final arbitration and conflict resolution system"""

    def __init__(
        self,
        ai_service: Optional[AIService] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ):
        """Initialize Gran Sabio Engine with optional shared AI service.

        Args:
            ai_service: AI service instance. Uses ``get_ai_service()`` if omitted.
            tool_event_callback: Optional pre-bound callback for live tool-loop
                events pushed to ``/stream/project`` under phase ``"gran_sabio"``.
                Signature: ``async def cb(event_type, payload)``. When the engine
                is used as a shared singleton the caller cannot bind session
                context, so this may legitimately be ``None`` — the feature
                degrades gracefully.
        """
        self.ai_service = ai_service if ai_service is not None else get_ai_service()
        self._tool_event_callback = tool_event_callback

    def _get_configured_default_model(self) -> str:
        """Return the configured default model for Gran Sabio operations."""
        defaults = get_default_models()
        model = defaults.get("gran_sabio")
        if not model:
            raise RuntimeError(
                "Gran Sabio default model is not configured in model_specs.json under 'default_models.gran_sabio'."
            )
        return model

    def _get_default_thinking_tokens(self, model_name: str) -> Optional[int]:
        """Return the default thinking budget for a model when supported."""
        try:
            thinking_config = config._get_thinking_budget_config(model_name)
        except Exception:
            return None

        if thinking_config.get("supported"):
            return thinking_config.get("default_tokens")
        return None

    def _get_default_critical_analysis_model(
        self, requested_model: Optional[str]
    ) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Get default model for critical analysis and decision making (deal-breakers, conflicts)
        Returns (model, reasoning_effort, thinking_budget_tokens)
        """
        default_model = self._get_configured_default_model()
        selected_model = requested_model or default_model
        thinking_tokens = self._get_default_thinking_tokens(selected_model)
        return selected_model, None, thinking_tokens

    def _get_default_content_generation_model(
        self, requested_model: Optional[str]
    ) -> Tuple[str, Optional[str], Optional[int]]:
        """
        Get default model for content generation
        Returns (model, reasoning_effort, thinking_budget_tokens)
        """
        default_model = self._get_configured_default_model()
        selected_model = requested_model or default_model
        thinking_tokens = self._get_default_thinking_tokens(selected_model)
        return selected_model, None, thinking_tokens

    def _resolve_model_alias(self, model_name: str) -> str:
        """Resolve aliases defined in model specs."""
        aliases = config.model_specs.get("aliases", {})
        return aliases.get(model_name, model_name)

    def _ensure_adequate_model_capacity(
        self, content_length: int, requested_model: str, original_request: Any
    ) -> str:
        """
        Ensure the Gran Sabio model has adequate token capacity for the content

        Args:
            content_length: Length of content in characters
            requested_model: Originally requested Gran Sabio model
            original_request: Original content request for context

        Returns:
            Model name that can handle the content size
        """
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = content_length // 4
        # Add overhead for prompts/instructions/output
        total_tokens_needed = int(estimated_tokens * 1.5)

        try:
            model_info = config.get_model_info(requested_model)
        except RuntimeError as exc:
            logger.error(
                "Unable to resolve model '%s' for Gran Sabio capacity check: %s",
                requested_model,
                exc,
            )
            raise

        model_input_capacity = model_info.get("input_tokens", 0)

        logger.info(
            "Gran Sabio capacity check: content ~%s tokens, needed ~%s, model %s capacity: %s",
            estimated_tokens,
            total_tokens_needed,
            requested_model,
            model_input_capacity,
        )

        # If model has sufficient capacity, use it
        if model_input_capacity >= total_tokens_needed:
            return requested_model

        resolved_requested = self._resolve_model_alias(requested_model)
        provider_preference_map = {
            "claude": "anthropic",
            "openai": "openai",
            "gemini": "google",
            "xai": "xai",
        }
        preferred_provider = provider_preference_map.get(model_info.get("provider"))

        candidate_entries: List[Tuple[int, int, str]] = []
        specs = config.model_specs.get("model_specifications", {})
        for provider, models in specs.items():
            for model_name, model_data in models.items():
                resolved_name = model_name
                if resolved_name == resolved_requested:
                    continue
                capacity = model_data.get("input_tokens", 0)
                if capacity < total_tokens_needed:
                    continue
                priority = 0 if (preferred_provider and provider == preferred_provider) else 1
                candidate_entries.append((priority, capacity, model_name))

        candidate_entries.sort(key=lambda entry: (entry[0], entry[1]))

        for _, capacity, candidate in candidate_entries:
            try:
                candidate_info = config.get_model_info(candidate)
            except RuntimeError:
                continue
            if candidate_info.get("input_tokens", 0) >= total_tokens_needed:
                logger.warning(
                    "Gran Sabio model upgraded from %s to %s due to content size (%s tokens needed)",
                    requested_model,
                    candidate,
                    total_tokens_needed,
                )
                return candidate

        logger.error(
            "No model found with sufficient capacity for %s tokens. Using %s and proceeding; result may fail.",
            total_tokens_needed,
            requested_model,
        )
        return requested_model

    # ============================
    # Minority Deal-Breakers Review
    # ============================
    async def review_minority_deal_breakers(
        self,
        session_id: str,
        content: str,
        minority_deal_breakers: Dict[str, Any],
        original_request: ContentRequest,
        stream_callback: Optional[callable] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> GranSabioResult:
        """
        Review content when a MINORITY of QA evaluators flagged deal-breakers.
        Provide a decision ensuring the flags align with configured QA criteria.
        """
        separator = "=" * 80
        count = len(minority_deal_breakers.get("details", []))
        total = minority_deal_breakers.get("total_evaluations", 0)

        if phase_logger:
            phase_logger.info(f"Reviewing minority deal-breakers ({count}/{total})")
            if phase_logger.extra_verbose:
                phase_logger.log_content_preview(content)
        else:
            logger.info(separator)
            logger.info("GRAN SABIO ESCALATION - MINORITY DEAL-BREAKERS REVIEW")
            logger.info("Session: %s", session_id)
            logger.info("Deal-breakers: %s of %s", count, total)
            logger.info(separator)

        async def _abort_if_cancelled(stage: str) -> None:
            if cancel_callback and await cancel_callback():
                logger.info(
                    "Gran Sabio cancellation requested during %s for session %s",
                    stage,
                    session_id
                )
                raise GranSabioProcessCancelled(stage)

        await _abort_if_cancelled("minority_review_setup")

        # Build enhanced context with QA configuration
        qa_cfg = minority_deal_breakers.get("qa_configuration", {})
        enhanced_db_context = self._build_enhanced_deal_breaker_context(minority_deal_breakers)
        alias_registry = getattr(original_request, "_model_alias_registry", None)
        prompt_safety_parts = [
            PromptPart(
                text=enhanced_db_context,
                source="system_generated",
                label="gran_sabio.minority_deal_breakers",
            )
        ] if alias_registry else None

        prompt = f"""
You are Gran Sabio, the senior arbiter. Review content flagged as deal-breaker by a MINORITY of QA evaluators.

ORIGINAL REQUEST:
- Content type: {original_request.content_type}
- Prompt: {original_request.prompt}

QA LAYER CONFIGURATION UNDER REVIEW:
- Name: {qa_cfg.get('layer_name', 'N/A')}
- Description: {qa_cfg.get('description', 'N/A')}
- Evaluation criteria: {qa_cfg.get('criteria', 'N/A')}
- Deal-breaker criterion: {qa_cfg.get('deal_breaker_criteria', 'Not specified')}
- Minimum score required: {qa_cfg.get('min_score', 'N/A')}

DEAL-BREAKERS REPORTED (ENHANCED CONTEXT):
{enhanced_db_context}

CONTENT TO REVIEW:
{content}

INSTRUCTIONS:
1) Decide if the flagged issues truly match the configured 'deal_breaker_criteria'.
2) If the issue is merely low score without matching the configured deal-breaker criterion, it is NOT a real deal-breaker.
3) Consider whether QA models deviated from the configured criteria (e.g., editorial judgement during a fact-check-only layer).
4) Content handling:
   - If the deal-breaker is a FALSE POSITIVE, set decision=APPROVED and modifications_made=false. Do NOT fill final_content.
   - If the deal-breaker is REAL, set decision=REJECTED and modifications_made=false. Do NOT fill final_content.
   - Set modifications_made=true ONLY when there is a genuine but MINOR issue that can be fixed with a small edit, and in that case final_content MUST contain the full edited text (non-empty string).
5) Score semantics:
   - score is the effective score for the QA layer under review after your arbitration.
   - If decision is APPROVED, score MUST be greater than or equal to the layer's minimum score.
   - If a minority deal-breaker is a false positive and no edit is needed, use at least the minimum passing score for that layer.
   - If decision is REJECTED, score may be below the minimum.

OUTPUT CONTRACT (JSON object, no extra keys, no markdown):
{{
  "decision": "APPROVED" | "REJECTED",
  "reason": "Short justification explaining whether flags match the configured deal-breaker criterion.",
  "score": <number 0-10>,
  "modifications_made": true | false,
  "final_content": <string with the full edited content when modifications_made=true, otherwise null>
}}
""".strip()

        try:
            model, reasoning_effort, thinking_tokens = self._get_default_critical_analysis_model(
                original_request.gran_sabio_model
            )
            adequate_model = self._ensure_adequate_model_capacity(len(content), model, original_request)
            logger.info(
                "Gran Sabio using %s with reasoning_effort=%s for minority deal-breaker check",
                adequate_model, reasoning_effort
            )

            usage_callback = (
                usage_tracker.create_callback(
                    phase="gran_sabio",
                    role="review",
                    operation="minority_deal_breaker_review",
                    iteration=minority_deal_breakers.get("iteration"),
                    metadata={
                        "session_id": session_id,
                        "qa_layer": qa_cfg.get("layer_name"),
                    },
                )
                if usage_tracker
                else None
            )
            # Log prompt if phase_logger available
            if phase_logger:
                phase_logger.log_prompt(
                    model=adequate_model,
                    system_prompt=None,
                    user_prompt=prompt,
                    temperature=0.3,
                    max_tokens=original_request.max_tokens,
                    reasoning_effort=reasoning_effort,
                    thinking_budget_tokens=thinking_tokens
                )

            await _abort_if_cancelled("minority_review_before_generation")

            payload, response_content = await self._invoke_gran_sabio_llm(
                request=original_request,
                prompt=prompt,
                model=adequate_model,
                reasoning_effort=reasoning_effort,
                thinking_tokens=thinking_tokens,
                temperature=0.3,
                system_prompt=None,
                response_schema=GRAN_SABIO_MINORITY_OVERRIDE_SCHEMA,
                max_tool_rounds=config.GRAN_SABIO_DECISION_MAX_TOOL_ROUNDS,
                initial_measurement_text=content,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                alias_registry=alias_registry,
                prompt_safety_parts=prompt_safety_parts,
                stream_callback=stream_callback,
                stream_stage_label="deal_breakers_review",
                cancel_callback=cancel_callback,
                abort_stage="minority_review_stream",
                tool_event_callback=tool_event_callback,
            )
            await _abort_if_cancelled("minority_review_after_generation")

            # Log full Gran Sabio response
            if phase_logger:
                phase_logger.log_response(
                    model=adequate_model,
                    response=response_content
                )
            elif getattr(original_request, "extra_verbose", False):
                logger.info(f"\n{separator}")
                logger.info(f"[EXTRA_VERBOSE] GRAN SABIO RESPONSE - Minority Deal-Breakers Review")
                logger.info(f"Session: {session_id}")
                logger.info(f"{separator}")
                logger.info(response_content)
                logger.info(f"{separator}\n")

            decision, score, reason, modifications_made, llm_final_content = _parse_minority_payload(payload)
            approved = decision == "APPROVED"

            # Matrix §3.4.3 adapter for review_minority_deal_breakers:
            #   row 1 (approved, modifications_made=True): LLM-returned final_content (fail-fast enforced by parser).
            #   row 2 (approved, modifications_made=False): original `content`.
            #   row 5 (not approved, any): original `content` (v8: preserve content instead of "").
            if modifications_made:
                final_content = llm_final_content or ""
            elif approved:
                final_content = content
            else:
                final_content = content

            # Log decision
            if phase_logger:
                phase_logger.log_decision(
                    decision="APPROVED" if approved else "REJECTED",
                    score=score
                )
            else:
                logger.info(separator)
                logger.info("GRAN SABIO DECISION - MINORITY DEAL-BREAKERS")
                logger.info("Session: %s", session_id)
                logger.info("DECISION: %s", "APPROVED" if approved else "REJECTED")
                logger.info("Final Score: %.2f/10", score)
                logger.info("Modifications Made: %s", "YES" if modifications_made else "NO")
                logger.info("Reasoning (first 300 chars): %s", reason[:300] + ("..." if len(reason) > 300 else ""))
                logger.info(separator)

            return GranSabioResult(
                approved=approved,
                final_content=final_content,
                final_score=score,
                reason=reason,
                modifications_made=modifications_made,
                error=None,
            )

        except GranSabioProcessCancelled:
            raise
        except Exception as e:
            logger.error("Gran Sabio review failed for session %s: %s", session_id, str(e))
            failure_reason = f"Gran Sabio review failed: {str(e)}"
            # Matrix row 8: exception path for review_minority_deal_breakers keeps `content` as defensive default.
            return GranSabioResult(
                approved=False,
                final_content=content,
                final_score=0.0,
                reason=failure_reason,
                modifications_made=False,
                error=str(e),
            )

    # ========================
    # Content Regeneration Fallback
    # ========================
    async def regenerate_content(
        self,
        session_id: str,
        original_request: ContentRequest,
        previous_attempts: List[str] = None,
        images: Optional[List[Any]] = None,
        stream_callback: Optional[callable] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        fallback_reason: Optional[str] = None,
    ) -> GranSabioResult:
        """
        Final fallback: generate new content from scratch when iterations are exhausted.
        """
        separator = "=" * 80
        logger.info(separator)
        logger.info("GRAN SABIO ESCALATION - CONTENT REGENERATION")
        logger.info("Session: %s", session_id)
        logger.info(
            "Previous attempts available: %s", (len(previous_attempts) if previous_attempts else 0)
        )
        logger.info("Model: %s", original_request.gran_sabio_model or "default")
        logger.info(separator)

        try:
            async def _abort_if_cancelled(stage: str) -> None:
                if cancel_callback and await cancel_callback():
                    logger.info(
                        "Gran Sabio regeneration cancelled during %s for session %s",
                        stage,
                        session_id
                    )
                    raise GranSabioProcessCancelled(stage)

            await _abort_if_cancelled("regeneration_setup")

            previous_context = ""
            if previous_attempts:
                truncated_attempts = []
                for i, attempt in enumerate(previous_attempts[-3:], 1):
                    truncated = attempt[:800] + "..." if len(attempt) > 800 else attempt
                    truncated_attempts.append(f"Attempt {i}: {truncated}")

                previous_context = (
                    "\n\n"
                    "PREVIOUS ATTEMPTS CONTEXT:\n"
                    "The following drafts were rejected in previous iterations. "
                    "Generate a completely different draft that avoids these problems:\n\n"
                    + "\n".join(truncated_attempts)
                    + "\n\nIMPORTANT: Create a FULLY NEW draft avoiding previous errors."
                )

            failure_context = ""
            if fallback_reason:
                failure_context = (
                    "\n\n"
                    "FALLBACK TRIGGER CONTEXT:\n"
                    f"{fallback_reason}\n\n"
                    "Use this as corrective context. If the failure was output-token "
                    "truncation, produce a more compact complete response that still "
                    "satisfies the original request."
                )

            word_instructions = ""
            if original_request.min_words or original_request.max_words:
                if original_request.min_words and original_request.max_words:
                    word_instructions = (
                        f"\nCRITICAL WORD COUNT: EXACTLY between {original_request.min_words} and {original_request.max_words} words.\n"
                        "- Count words while writing.\n"
                        "- Do NOT exceed the max or go below the min.\n"
                        "- If you fail this range, the output is invalid."
                    )
                elif original_request.max_words:
                    word_instructions = f"\nCRITICAL WORD COUNT: MAX {original_request.max_words} words (MANDATORY)."
                elif original_request.min_words:
                    word_instructions = f"\nCRITICAL WORD COUNT: MIN {original_request.min_words} words (MANDATORY)."

            json_output_effective = is_json_output_requested(original_request)
            if json_output_effective:
                response_instruction = (
                    "Respond with the generated JSON payload only. The response "
                    "must be strict valid JSON, start with { or [, and contain no "
                    "markdown, code fences, preambles, tags, or explanatory text. "
                    "If any text is placed outside the first JSON object/array, "
                    "the application will discard it."
                )
            else:
                response_instruction = (
                    "Respond with the generated content only as plain text, no "
                    "preambles, no tags, no JSON wrapper."
                )

            generation_prompt = f"""
You are Gran Sabio. This is a single final chance to produce a draft that strictly meets all requirements.

ORIGINAL REQUEST:
- Content type: {original_request.content_type}
- Prompt: {original_request.prompt}
{word_instructions}

{failure_context}

{previous_context}

CRITICAL INSTRUCTIONS:
1) Follow the requested requirements exactly.
2) Produce professional-quality content.
3) Avoid prior issues.
4) If word limits are present, comply exactly (count words).

{response_instruction}
""".strip()
            prompt_safety_parts = [
                PromptPart(
                    text=(
                        "Gran Sabio content regeneration instructions. "
                        "Produce a new draft, avoid prior issues, and obey configured word limits."
                    ),
                    source="system_generated",
                    label="gran_sabio.regeneration.instructions",
                ),
                PromptPart(
                    text=(
                        f"Content type: {original_request.content_type}\n"
                        f"Prompt: {original_request.prompt}\n"
                        f"{failure_context}\n"
                        f"{previous_context}"
                    ),
                    source="user_supplied",
                    label="gran_sabio.regeneration.request_context",
                ),
            ]
            model_alias_registry = getattr(original_request, "_model_alias_registry", None)

            model, reasoning_effort, thinking_tokens = self._get_default_content_generation_model(
                original_request.gran_sabio_model
            )
            adequate_model = self._ensure_adequate_model_capacity(
                len(original_request.prompt) + len(previous_context), model, original_request
            )
            logger.info(
                "Gran Sabio using %s with thinking_budget_tokens=%s for content regeneration",
                adequate_model, thinking_tokens
            )

            generated_content = ""
            usage_callback = (
                usage_tracker.create_callback(
                    phase="gran_sabio",
                    role="regeneration",
                    operation="content_regeneration",
                    metadata={
                        "session_id": session_id,
                        "model": adequate_model,
                    },
                )
                if usage_tracker
                else None
            )

            # Streaming retry configuration
            max_stream_attempts = config.MAX_RETRIES if config.RETRY_STREAMING_AFTER_PARTIAL else 1
            stream_delay = config.RETRY_DELAY

            # §3.4.3 streaming escape: non-JSON regeneration keeps live
            # streaming when a callback is provided. JSON regeneration must
            # still route through the shared loop so the final payload is
            # checked and normalized through the same JSON_LOOSE /
            # JSON_STRUCTURED contract as normal generation.
            tool_loop_enabled = (
                _should_use_gransabio_tools(original_request, adequate_model)
                and (stream_callback is None or json_output_effective)
            )

            if tool_loop_enabled:
                await _abort_if_cancelled("regeneration_before_tool_loop")
                json_schema = getattr(original_request, "json_schema", None)
                json_options_for_loop = (
                    make_loose_json_validate_options()
                    if json_output_effective
                    and json_schema is None
                    else None
                )
                output_contract = OutputContract.FREE_TEXT
                response_format = None
                if json_output_effective:
                    if json_schema is None:
                        output_contract = OutputContract.JSON_LOOSE
                    else:
                        output_contract = OutputContract.JSON_STRUCTURED
                        response_format = json_schema

                def _validate_regeneration_draft(candidate: str) -> DraftValidationResult:
                    return validate_generation_candidate(
                        candidate,
                        original_request,
                        include_json_validation=json_output_effective
                        or bool(getattr(original_request, "target_field", None)),
                        json_options=json_options_for_loop,
                    )

                try:
                    generated_content, envelope = await self.ai_service.call_ai_with_validation_tools(
                        prompt=generation_prompt,
                        model=adequate_model,
                        validation_callback=_validate_regeneration_draft,
                        stop_on_approval=True,
                        output_contract=output_contract,
                        response_format=response_format,
                        json_expectations=getattr(original_request, "json_expectations", None),
                        payload_scope=PayloadScope.GENERATOR,
                        max_tool_rounds=config.GRAN_SABIO_REGENERATE_MAX_TOOL_ROUNDS,
                        loop_scope=LoopScope.GRAN_SABIO,
                        tool_event_callback=tool_event_callback if tool_event_callback is not None else self._tool_event_callback,
                        initial_measurement_text=None,
                        temperature=0.7,
                        max_tokens=original_request.max_tokens,
                        extra_verbose=getattr(original_request, "extra_verbose", False),
                        reasoning_effort=reasoning_effort,
                        thinking_budget_tokens=thinking_tokens,
                        content_type=original_request.content_type,
                        usage_callback=usage_callback,
                        images=images,
                        model_alias_registry=model_alias_registry,
                        prompt_safety_parts=prompt_safety_parts,
                    )
                except Exception:
                    raise

                if envelope.tools_skipped_reason or not generated_content:
                    # Tool loop unavailable (Responses API, no_tool_support,
                    # or context_too_large). Fall back to a single-shot call
                    # rather than returning an empty draft.
                    await _abort_if_cancelled("regeneration_before_single_shot_fallback")
                    generated_content = await self.ai_service.generate_content(
                        prompt=generation_prompt,
                        model=adequate_model,
                        temperature=0.7,
                        max_tokens=original_request.max_tokens,
                        extra_verbose=getattr(original_request, "extra_verbose", False),
                        reasoning_effort=reasoning_effort,
                        thinking_budget_tokens=thinking_tokens,
                        usage_callback=usage_callback,
                        images=images,
                        model_alias_registry=model_alias_registry,
                        prompt_safety_parts=prompt_safety_parts,
                    )
                if stream_callback is not None and generated_content:
                    await stream_callback(
                        generated_content,
                        adequate_model,
                        "content_regeneration",
                    )
            elif stream_callback:
                await _abort_if_cancelled("regeneration_before_stream")

                for stream_attempt in range(1, max_stream_attempts + 1):
                    # Reset state for each attempt
                    generated_content = ""

                    if stream_attempt > 1:
                        logger.info(
                            "Gran Sabio regeneration retry %d/%d for session %s",
                            stream_attempt, max_stream_attempts, session_id
                        )

                    try:
                        async for chunk in self.ai_service.generate_content_stream(
                            prompt=generation_prompt,
                            model=adequate_model,
                            temperature=0.7,
                            max_tokens=original_request.max_tokens,
                            extra_verbose=getattr(original_request, "extra_verbose", False),
                            reasoning_effort=reasoning_effort,
                            thinking_budget_tokens=thinking_tokens,
                            usage_callback=usage_callback,
                            images=images,
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
                            if chunk_text:
                                # Only accumulate non-thinking for final content
                                if not is_thinking:
                                    generated_content += chunk_text
                                # Stream all for live monitoring
                                await stream_callback(chunk_text, adequate_model, "content_regeneration")
                            await _abort_if_cancelled("regeneration_stream")

                        # Success - exit retry loop
                        break

                    except GranSabioProcessCancelled:
                        raise
                    except (AIRequestError, Exception) as stream_error:
                        is_retryable = _is_retryable_streaming_error(stream_error)
                        is_last_attempt = stream_attempt >= max_stream_attempts

                        if is_retryable and not is_last_attempt:
                            error_reason = _extract_error_reason(stream_error)
                            logger.warning(
                                "Gran Sabio regeneration API error on attempt %d/%d for session %s: %s. Retrying in %ss...",
                                stream_attempt, max_stream_attempts, session_id, error_reason, stream_delay
                            )
                            await asyncio.sleep(stream_delay)
                            continue
                        else:
                            error_reason = _extract_error_reason(stream_error)
                            logger.error(
                                "Gran Sabio regeneration failed after %d attempts for session %s: %s",
                                stream_attempt, session_id, error_reason
                            )
                            raise

            else:
                await _abort_if_cancelled("regeneration_before_generation")
                generated_content = await self.ai_service.generate_content(
                    prompt=generation_prompt,
                    model=adequate_model,
                    temperature=0.7,
                    max_tokens=original_request.max_tokens,
                    extra_verbose=getattr(original_request, "extra_verbose", False),
                    reasoning_effort=reasoning_effort,
                    thinking_budget_tokens=thinking_tokens,
                    usage_callback=usage_callback,
                    images=images,
                    model_alias_registry=model_alias_registry,
                    prompt_safety_parts=prompt_safety_parts,
                )
            await _abort_if_cancelled("regeneration_after_generation")

            # Log full Gran Sabio response if extra_verbose is enabled
            if getattr(original_request, "extra_verbose", False):
                logger.info(f"\n{separator}")
                logger.info(f"[EXTRA_VERBOSE] GRAN SABIO RESPONSE - Content Regeneration")
                logger.info(f"Session: {session_id}")
                logger.info(f"{separator}")
                logger.info(generated_content)
                logger.info(f"{separator}\n")

            word_count = len(generated_content.split())
            logger.info(separator)
            logger.info("GRAN SABIO REGENERATION COMPLETE")
            logger.info("Session: %s", session_id)
            logger.info("Words generated: %s", word_count)
            logger.info("Content length: %s characters", len(generated_content))
            logger.info("Note: Content will now go through QA evaluation")
            logger.info(separator)

            return GranSabioResult(
                approved=True,
                final_content=generated_content,
                final_score=None,
                reason="Gran Sabio regeneration attempt - content ready for QA evaluation",
                modifications_made=False,
                error=None,
            )

        except GranSabioProcessCancelled:
            raise
        except Exception as e:
            logger.error("Gran Sabio content regeneration failed for session %s: %s", session_id, str(e))
            # Matrix row 9: exception path preserves last previous attempt when available.
            fallback_content = previous_attempts[-1] if previous_attempts else ""
            return GranSabioResult(
                approved=False,
                final_content=fallback_content,
                final_score=0.0,
                reason=f"Gran Sabio regeneration failed: {str(e)}",
                modifications_made=False,
                error=str(e),
            )

    # ==========================
    # Iterations Comprehensive Review
    # ==========================
    async def review_iterations(
        self,
        session_id: str,
        iterations: List[Dict[str, Any]],
        original_request: ContentRequest,
        fallback_notes: Optional[List[str]] = None,
        stream_callback: Optional[callable] = None,
        cancel_callback: CancelCallback = None,
        usage_tracker: Optional[UsageTracker] = None,
        phase_logger: Optional["PhaseLogger"] = None,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> GranSabioResult:
        """
        Review all iterations when max iterations are reached without consensus.
        """
        separator = "=" * 80

        if phase_logger:
            phase_logger.info(f"Reviewing {len(iterations)} iterations for final decision...")
        else:
            logger.info(separator)
            logger.info("GRAN SABIO ESCALATION - COMPREHENSIVE ITERATIONS REVIEW")
            logger.info("Session: %s", session_id)
            logger.info("Total iterations to review: %s", len(iterations))
            if fallback_notes:
                logger.info("Fallback notes present: %s", len(fallback_notes))
            logger.info(separator)

        try:
            async def _abort_if_cancelled(stage: str) -> None:
                if cancel_callback and await cancel_callback():
                    logger.info(
                        "Gran Sabio iterations review cancelled during %s for session %s",
                        stage,
                        session_id
                    )
                    raise GranSabioProcessCancelled(stage)

            await _abort_if_cancelled("iterations_review_setup")

            iteration_analysis = self._analyze_iteration_patterns(iterations)
            review_prompt = self._build_gran_sabio_prompt(
                iterations, original_request, iteration_analysis, fallback_notes=fallback_notes
            )
            alias_registry = getattr(original_request, "_model_alias_registry", None)
            prompt_safety_parts = None
            if alias_registry:
                prompt_safety_parts = [
                    PromptPart(
                        text=self._format_trend_analysis(iteration_analysis)
                        + "\n"
                        + self._format_iteration_details(iterations, include_content=False)
                        + "\n"
                        + "\n".join(fallback_notes or []),
                        source="system_generated",
                        label="gran_sabio.iteration_review",
                    )
                ]

            best_iteration = self._pick_best_iteration(iterations)
            best_content = best_iteration.get("content", "") if best_iteration else ""

            model, reasoning_effort, thinking_tokens = self._get_default_critical_analysis_model(
                original_request.gran_sabio_model
            )
            adequate_model = self._ensure_adequate_model_capacity(len(best_content), model, original_request)
            logger.info(
                "Gran Sabio using %s with reasoning_effort=%s for iteration review",
                adequate_model, reasoning_effort
            )

            usage_callback = (
                usage_tracker.create_callback(
                    phase="gran_sabio",
                    role="review",
                    operation="iterations_review",
                    metadata={
                        "session_id": session_id,
                        "iterations": len(iterations),
                    },
                )
                if usage_tracker
                else None
            )
            # Log prompt if phase_logger available
            if phase_logger:
                phase_logger.log_prompt(
                    model=adequate_model,
                    system_prompt=config.GRAN_SABIO_SYSTEM_PROMPT,
                    user_prompt=review_prompt,
                    temperature=0.4,
                    max_tokens=original_request.max_tokens,
                    reasoning_effort=reasoning_effort,
                    thinking_budget_tokens=thinking_tokens
                )

            await _abort_if_cancelled("iterations_review_before_generation")

            payload, gran_sabio_response = await self._invoke_gran_sabio_llm(
                request=original_request,
                prompt=review_prompt,
                model=adequate_model,
                reasoning_effort=reasoning_effort,
                thinking_tokens=thinking_tokens,
                temperature=0.4,
                system_prompt=config.GRAN_SABIO_SYSTEM_PROMPT,
                response_schema=GRAN_SABIO_ESCALATION_SCHEMA,
                max_tool_rounds=config.GRAN_SABIO_ESCALATION_MAX_TOOL_ROUNDS,
                initial_measurement_text=best_content or None,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                alias_registry=alias_registry,
                prompt_safety_parts=prompt_safety_parts,
                stream_callback=stream_callback,
                stream_stage_label="iterations_review",
                cancel_callback=cancel_callback,
                abort_stage="iterations_review_stream",
                tool_event_callback=tool_event_callback,
            )

            await _abort_if_cancelled("iterations_review_after_generation")

            # Log full Gran Sabio response
            if phase_logger:
                phase_logger.log_response(
                    model=adequate_model,
                    response=gran_sabio_response
                )
            elif getattr(original_request, "extra_verbose", False):
                logger.info(f"\n{separator}")
                logger.info(f"[EXTRA_VERBOSE] GRAN SABIO RESPONSE - Iterations Review")
                logger.info(f"Session: {session_id}")
                logger.info(f"{separator}")
                logger.info(gran_sabio_response)
                logger.info(f"{separator}\n")

            decision, score, reason, modifications_made, llm_final_content = _parse_escalation_payload(payload)
            approved = decision in {"APPROVE", "APPROVE_WITH_MODIFICATIONS"}

            # Matrix §3.4.3 adapter for review_iterations:
            #   row 1 (APPROVE_WITH_MODIFICATIONS): LLM-returned final_content (fail-fast enforced by parser).
            #   row 3 (APPROVE): best_iteration["content"] when available (edge case: iterations=[] -> "").
            #   row 6 (REJECT): best_iteration["content"] (v8: preserve best content instead of "").
            if modifications_made:
                final_content = llm_final_content or ""
            else:
                final_content = best_iteration.get("content", "") if best_iteration else ""

            result = GranSabioResult(
                approved=approved,
                final_content=final_content,
                final_score=score,
                reason=reason,
                modifications_made=modifications_made,
                error=None,
            )

            # Log final decision
            if phase_logger:
                phase_logger.log_decision(
                    decision="APPROVED" if result.approved else "REJECTED",
                    score=result.final_score
                )
            else:
                logger.info(separator)
                logger.info("GRAN SABIO FINAL DECISION - ITERATIONS REVIEW")
                logger.info("Session: %s", session_id)
                logger.info("DECISION: %s", "APPROVED" if result.approved else "REJECTED")
                logger.info("Final Score: %.2f/10", result.final_score)
                logger.info("Modifications Made: %s", "YES" if result.modifications_made else "NO")
                logger.info("Reasoning (first 500 chars): %s", result.reason[:500] + ("..." if len(result.reason) > 500 else ""))
                logger.info(separator)

            return result

        except GranSabioProcessCancelled:
            raise
        except Exception as e:
            logger.error("Gran Sabio review failed for session %s: %s", session_id, str(e))
            # Matrix row 10: exception path preserves best iteration content when available.
            best_iteration_exc = self._pick_best_iteration(iterations)
            fallback_content = (
                best_iteration_exc.get("content", "") if best_iteration_exc else ""
            )
            return GranSabioResult(
                approved=False,
                final_content=fallback_content,
                final_score=0.0,
                reason=f"Gran Sabio review failed: {str(e)}",
                modifications_made=False,
                error=str(e),
            )

    def _analyze_iteration_patterns(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across iterations to identify issues"""
        if not iterations:
            return {"error": "No iterations to analyze"}

        layer_score_trends: Dict[str, List[float]] = {}
        model_consistency: Dict[str, List[float]] = {}
        deal_breaker_patterns: Dict[str, Dict[str, int]] = {}

        for i, iteration in enumerate(iterations, 1):
            qa_results = iteration.get("qa_results", {})
            for layer_name, layer_results in qa_results.items():
                if layer_name not in layer_score_trends:
                    layer_score_trends[layer_name] = []
                layer_scores: List[float] = []
                for model, evaluation in layer_results.items():
                    evaluator_name = get_evaluator_alias(evaluation, fallback=model)
                    score = evaluation.score
                    if score is None:
                        continue

                    layer_scores.append(score)

                    if evaluator_name not in model_consistency:
                        model_consistency[evaluator_name] = []
                    model_consistency[evaluator_name].append(score)

                    if evaluation.deal_breaker:
                        if layer_name not in deal_breaker_patterns:
                            deal_breaker_patterns[layer_name] = {}
                        if evaluator_name not in deal_breaker_patterns[layer_name]:
                            deal_breaker_patterns[layer_name][evaluator_name] = 0
                        deal_breaker_patterns[layer_name][evaluator_name] += 1

                if layer_scores:
                    avg_layer_score = sum(layer_scores) / len(layer_scores)
                    layer_score_trends[layer_name].append(avg_layer_score)

        analysis: Dict[str, Any] = {
            "total_iterations": len(iterations),
            "layer_trends": {},
            "model_analysis": {},
            "deal_breaker_analysis": deal_breaker_patterns,
            "improvement_patterns": {},
            "consistent_issues": [],
        }

        for layer_name, scores in layer_score_trends.items():
            if len(scores) > 1:
                first_half_avg = sum(scores[: len(scores) // 2]) / max(1, len(scores) // 2)
                second_half_avg = sum(scores[len(scores) // 2 :]) / max(1, len(scores) - len(scores) // 2)
                trend = (
                    "improving"
                    if second_half_avg > first_half_avg + 0.5
                    else "declining"
                    if second_half_avg < first_half_avg - 0.5
                    else "stable"
                )
                analysis["layer_trends"][layer_name] = {
                    "trend": trend,
                    "first_half_avg": first_half_avg,
                    "second_half_avg": second_half_avg,
                    "best_score": max(scores),
                    "worst_score": min(scores),
                    "consistency": 1.0 - (max(scores) - min(scores)) / 10.0,
                }

        for model, scores in model_consistency.items():
            if len(scores) > 2:
                variance = sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)
                analysis["model_analysis"][model] = {
                    "average_score": sum(scores) / len(scores),
                    "variance": variance,
                    "consistency": max(0.0, 1.0 - variance / 25.0),
                    "score_range": max(scores) - min(scores),
                }

        for layer_name, models in deal_breaker_patterns.items():
            total_iterations = len(iterations)
            for model, count in models.items():
                if count >= total_iterations * 0.7:
                    analysis["consistent_issues"].append(
                        f"{layer_name} consistently flagged by {model} ({count}/{total_iterations} iterations)"
                    )

        return analysis

    def _build_gran_sabio_prompt(
        self,
        iterations: List[Dict[str, Any]],
        original_request: ContentRequest,
        iteration_analysis: Dict[str, Any],
        fallback_notes: Optional[List[str]] = None,
    ) -> str:
        """Build comprehensive prompt for Gran Sabio review without duplicating heavy content."""

        best_iteration = None
        best_score = 0.0
        for it in iterations:
            consensus = it.get("consensus", {})
            if hasattr(consensus, "average_score"):
                avg_score = consensus.average_score
            elif isinstance(consensus, dict):
                avg_score = consensus.get("average_score", 0.0)
            else:
                avg_score = 0.0
            if avg_score > best_score:
                best_score = avg_score
                best_iteration = it

        # Pull QA configuration (identical for the session)
        qa_config = iterations[0].get("qa_layers_config", []) if iterations else []

        # Build sections
        qa_cfg_block = self._format_qa_layers_config(qa_config)
        trend_block = self._format_trend_analysis(iteration_analysis)
        iterations_block = self._format_iteration_details(iterations, include_content=False)
        best_meta = (
            f"- Words: {best_iteration.get('content_word_count', 'N/A')}\n"
            f"- Characters: {best_iteration.get('content_char_count', 'N/A')}"
            if best_iteration else "No metadata"
        )
        best_content_text = best_iteration["content"] if best_iteration and "content" in best_iteration else "Not available"

        prompt = f"""
You are Gran Sabio, the final arbiter. Review the full process after {len(iterations)} iterations without automatic approval.

QA EVALUATION CONFIGURATION (applies to the whole session):
{qa_cfg_block}

ORIGINAL REQUEST:
- Content type: {original_request.content_type}
- Prompt: {original_request.prompt}
- Global minimum score: {original_request.min_global_score}

BEST CONTENT (highest consensus: {best_score:.2f}/10):
{best_meta}
{best_content_text}

PER-ITERATION DETAILS (no content duplication, only metrics & highlights):
{iterations_block}

TRENDS & CONSISTENCY ANALYSIS:
{trend_block}

INSTRUCTIONS:
1) Check whether QA evaluations align with the configured criteria (above).
2) Distinguish real deal-breakers (matching 'deal_breaker_criteria') from false positives (criteria applied incorrectly).
3) Consider whether remaining issues are critical or minor.
4) Choose one:
   - APPROVE: The content is acceptable. Leave final_content=null and modifications_made=false.
   - APPROVE_WITH_MODIFICATIONS: Content needs MINOR fixes. Set modifications_made=true and place the FULL edited text in final_content (non-empty string).
   - REJECT: Content has unfixable issues. Leave final_content=null and modifications_made=false.
5) Prefer APPROVE over APPROVE_WITH_MODIFICATIONS. Only modify if strictly necessary.

OUTPUT CONTRACT (JSON object, no extra keys, no markdown):
{{
  "decision": "APPROVE" | "APPROVE_WITH_MODIFICATIONS" | "REJECT",
  "reason": "Your reasoning and justification.",
  "score": <number 0-10>,
  "modifications_made": true | false,
  "final_content": <string with the full edited content when modifications_made=true, otherwise null>
}}
""".strip()

        if fallback_notes:
            prompt += "\n\nFALLBACK NOTES:\n- " + "\n- ".join(fallback_notes)

        return prompt

    @staticmethod
    def _pick_best_iteration(
        iterations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return the iteration with the highest consensus average score.

        Extracted from the former ``_parse_gran_sabio_response._avg`` helper
        so happy-path and exception-path adapters (matrix rows 3/6/10) share
        the same selection logic.
        """

        if not iterations:
            return None

        def _avg(entry: Dict[str, Any]) -> float:
            consensus = entry.get("consensus", {})
            if hasattr(consensus, "average_score"):
                return consensus.average_score
            if isinstance(consensus, dict):
                return consensus.get("average_score", 0.0)
            return 0.0

        return max(iterations, key=_avg)

    # ---------------------------
    # Shared tool-loop invocation
    # ---------------------------
    async def _invoke_gran_sabio_llm(
        self,
        *,
        request: ContentRequest,
        prompt: str,
        model: str,
        reasoning_effort: Optional[str],
        thinking_tokens: Optional[int],
        temperature: float,
        system_prompt: Optional[str],
        response_schema: Dict[str, Any],
        max_tool_rounds: int,
        initial_measurement_text: Optional[str],
        usage_callback: Optional[Callable[[Dict[str, Any]], None]],
        phase_logger: Optional["PhaseLogger"],
        alias_registry: Optional[Any],
        prompt_safety_parts: Optional[List[Any]],
        stream_callback: Optional[Callable[..., Awaitable[None]]],
        stream_stage_label: str,
        cancel_callback: CancelCallback,
        abort_stage: str,
        tool_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Run a GranSabio JSON_STRUCTURED call with tool-loop or single-shot fallback.

        Returns ``(parsed_payload, raw_response_text)``. The tool loop is used
        when ``_should_use_gransabio_tools`` is True **and** no
        ``stream_callback`` is provided; otherwise (streaming request, or
        tools disabled/unsupported) we fall back to ``generate_content`` /
        ``generate_content_stream`` and parse the response here.
        """

        async def _abort(stage: str) -> None:
            if cancel_callback and await cancel_callback():
                logger.info(
                    "Gran Sabio cancellation requested during %s",
                    stage,
                )
                raise GranSabioProcessCancelled(stage)

        tool_loop_enabled = (
            stream_callback is None
            and _should_use_gransabio_tools(request, model)
        )

        if tool_loop_enabled:
            measurement_request = _build_gran_sabio_measurement_request(request)
            measurement_json_output = is_json_output_requested(measurement_request)
            measurement_json_options = (
                make_loose_json_validate_options()
                if measurement_json_output
                and getattr(measurement_request, "json_schema", None) is None
                else None
            )
            include_measurement_json_validation = (
                measurement_json_output
                or bool(getattr(measurement_request, "target_field", None))
            )

            def _measurement_validator(candidate: str) -> DraftValidationResult:
                return validate_generation_candidate(
                    candidate,
                    measurement_request,
                    include_json_validation=include_measurement_json_validation,
                    json_options=measurement_json_options,
                )

            content, envelope = await self.ai_service.call_ai_with_validation_tools(
                prompt=prompt,
                model=model,
                validation_callback=_measurement_validator,
                stop_on_approval=False,
                output_contract=OutputContract.JSON_STRUCTURED,
                response_format=response_schema,
                payload_scope=PayloadScope.MEASUREMENT_ONLY,
                max_tool_rounds=max_tool_rounds,
                loop_scope=LoopScope.GRAN_SABIO,
                tool_event_callback=tool_event_callback if tool_event_callback is not None else self._tool_event_callback,
                initial_measurement_text=initial_measurement_text,
                temperature=temperature,
                max_tokens=request.max_tokens,
                system_prompt=system_prompt,
                extra_verbose=getattr(request, "extra_verbose", False),
                reasoning_effort=reasoning_effort,
                thinking_budget_tokens=thinking_tokens,
                content_type=request.content_type,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                model_alias_registry=alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            )

            if envelope.tools_skipped_reason is None and envelope.payload is not None:
                return envelope.payload, content

            # Tools skipped (responses_api / no_tool_support / context_too_large):
            # fall back to a single-shot JSON call rather than returning an
            # empty envelope payload.
            logger.info(
                "Gran Sabio tool loop skipped (%s); falling back to single-shot JSON call",
                envelope.tools_skipped_reason,
            )

        # Single-shot path (either streaming requested, tool loop disabled,
        # or provider-level fallback). We still expect a JSON response
        # matching the schema; the prompt contract instructs the model.
        response_text = ""
        if stream_callback is not None:
            await _abort(f"{abort_stage}_before_stream")
            async for chunk in self.ai_service.generate_content_stream(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=request.max_tokens,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                thinking_budget_tokens=thinking_tokens,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                model_alias_registry=alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            ):
                if isinstance(chunk, StreamChunk):
                    chunk_text = chunk.text
                    is_thinking = chunk.is_thinking
                else:
                    chunk_text = chunk
                    is_thinking = False
                if chunk_text:
                    if not is_thinking:
                        response_text += chunk_text
                    await stream_callback(chunk_text, model, stream_stage_label)
                await _abort(abort_stage)
        else:
            await _abort(f"{abort_stage}_before_single_shot")
            response_text = await self.ai_service.generate_content(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=request.max_tokens,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
                thinking_budget_tokens=thinking_tokens,
                json_output=True,
                json_schema=response_schema,
                usage_callback=usage_callback,
                phase_logger=phase_logger,
                model_alias_registry=alias_registry,
                prompt_safety_parts=prompt_safety_parts,
            )

        parsed = self._parse_single_shot_json(response_text, response_schema)
        return parsed, response_text

    @staticmethod
    def _parse_single_shot_json(
        response_text: str,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse a JSON object out of a single-shot Gran Sabio response.

        Handles markdown code fences and common LLM JSON wrappers via the
        shared cleanroom parser. Raises ``ValueError`` when the body is empty,
        malformed, or not a JSON object.
        """

        if not (response_text or "").strip():
            raise ValueError("Gran Sabio returned an empty response")

        try:
            return parse_json_with_markdown_fences(
                response_text,
                schema=response_schema,
                context="Gran Sabio response",
            )
        except JsonContractError as exc:
            raise ValueError(
                f"Gran Sabio response is not valid JSON: {exc}"
            ) from exc

    # ---------------------------
    # Formatting helpers
    # ---------------------------
    def _build_deal_breaker_context(self, minority_deal_breakers: Dict[str, Any]) -> str:
        """Legacy: simple formatter kept for compatibility."""
        details = minority_deal_breakers.get("details", [])
        if not details:
            return "No deal-breakers detected."
        lines = [f"- Layer '{db['layer']}' by model '{db['model']}': {db['reason']}" for db in details]
        return f"\nTotal: {len(details)} deal-breakers out of {minority_deal_breakers.get('total_evaluations', 0)} evaluations\nDetails:\n" + "\n".join(lines)

    def _get_word_config(self, request: ContentRequest) -> str:
        """Return word count config summary in English."""
        if request.min_words and request.max_words:
            return f"{request.min_words}-{request.max_words} words"
        elif request.max_words:
            return f"Max {request.max_words} words"
        elif request.min_words:
            return f"Min {request.min_words} words"
        else:
            return "No specific word limits"

    # New helpers for better formatting without duplicating heavy content
    def _format_qa_layers_config(self, qa_config: List[Dict]) -> str:
        if not qa_config:
            return "No QA layers configuration available."
        chunks = []
        for layer in qa_config:
            chunks.append(
                (
                    f"Layer: {layer.get('name', 'N/A')}\n"
                    f"  - Description: {layer.get('description', 'N/A')}\n"
                    f"  - Criteria: {layer.get('criteria', 'N/A')}\n"
                    f"  - Min score: {layer.get('min_score', 'N/A')}\n"
                    f"  - Deal-breaker criterion: {layer.get('deal_breaker_criteria') or 'Not configured'}\n"
                    f"  - Order: {layer.get('order', 'N/A')}\n"
                    f"  - Is deal-breaker layer: {layer.get('is_deal_breaker', 'N/A')}\n"
                    f"  - Is mandatory: {layer.get('is_mandatory', 'N/A')}"
                )
            )
        return "\n".join(chunks)

    def _format_iteration_details(self, iterations: List[Dict], include_content: bool = False) -> str:
        details = []
        for it in iterations:
            consensus = it.get("consensus", {})
            if hasattr(consensus, "average_score"):
                avg_score = consensus.average_score
                db_count = len(getattr(consensus, "deal_breakers", []) or [])
            elif isinstance(consensus, dict):
                avg_score = consensus.get("average_score", 0.0)
                db_count = consensus.get("deal_breakers_count", 0)
            else:
                avg_score = 0.0
                db_count = 0

            line = (
                f"Iteration {it.get('iteration')}: "
                f"avg={avg_score:.2f}/10, deal_breakers={db_count}, "
                f"words={it.get('content_word_count', 'N/A')}, "
                f"chars={it.get('content_char_count', 'N/A')}"
            )
            if include_content and it.get("content_summary"):
                line += f"\n  Preview: {it['content_summary']}"
            details.append(line)
        return "\n".join(details) if details else "No iterations detail available."

    def _build_enhanced_deal_breaker_context(self, minority_deal_breakers: Dict) -> str:
        details = minority_deal_breakers.get("details", [])
        if not details:
            return "No deal-breakers reported."
        lines = []
        for db in details:
            evaluator_name = db.get("evaluator") or db.get("model") or "Evaluator"
            lines.append(
                (
                    "Deal-breaker reported:\n"
                    f"  - QA layer: {db.get('layer')}\n"
                    f"  - Evaluator: {evaluator_name}\n"
                    f"  - Score given: {db.get('score_given', 'N/A')}\n"
                    f"  - Evaluator reason: \"{db.get('reason', '').strip()}\"\n"
                    f"  - Configured deal-breaker criterion: \"{db.get('layer_deal_breaker_criteria') or 'Not configured'}\"\n"
                    f"  - Layer min score: {db.get('layer_min_score', 'N/A')}"
                )
            )
        header = f"Summary: {len(details)} deal-breakers out of {minority_deal_breakers.get('total_evaluations', 0)} evaluations"
        footer = (
            "\nNOTE: A deal-breaker is *valid* only if it matches the configured 'deal_breaker_criteria'. "
            "A low score alone without a matching criterion should NOT be considered a real deal-breaker."
        )
        return header + "\n" + "\n\n".join(lines) + footer

    def _format_trend_analysis(self, analysis: Dict) -> str:
        if analysis.get("error"):
            return f"Trend analysis error: {analysis['error']}"
        parts: List[str] = []

        layer_trends = analysis.get("layer_trends", {})
        if layer_trends:
            parts.append("LAYER TRENDS:")
            for layer_name, trend in layer_trends.items():
                parts.append(
                    f"{layer_name}:\n"
                    f"  - Trend: {trend.get('trend', 'unknown')}\n"
                    f"  - Best score: {trend.get('best_score', 0):.2f}\n"
                    f"  - Worst score: {trend.get('worst_score', 0):.2f}\n"
                    f"  - Consistency: {trend.get('consistency', 0):.2f}"
                )

        consistent_issues = analysis.get("consistent_issues", [])
        if consistent_issues:
            parts.append("\nCONSISTENT ISSUES:")
            for issue in consistent_issues:
                parts.append(f"  - {issue}")

        model_analysis = analysis.get("model_analysis", {})
        if model_analysis:
            parts.append("\nEVALUATOR CONSISTENCY:")
            for model, stats in model_analysis.items():
                parts.append(
                    f"{model}:\n"
                    f"  - Avg score: {stats.get('average_score', 0):.2f}\n"
                    f"  - Consistency: {stats.get('consistency', 0):.2f}\n"
                    f"  - Score range: {stats.get('score_range', 0):.2f}"
                )

        return "\n".join(parts) if parts else "No trends available."
