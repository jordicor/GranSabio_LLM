"""
Generation and streaming API routes for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import math
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Body, HTTPException
from fastapi.responses import JSONResponse

from logging_utils import create_phase_logger, Phase
from attachments_router import get_attachment_manager
from deal_breaker_tracker import get_tracker
from models import (
    ContentRequest,
    GenerationInitResponse,
    GenerationStatus,
    ImageData,
    PreflightIssue,
    PreflightResult,
    ProjectInitRequest,
    ProjectInitResponse,
)
from preflight_validator import run_preflight_validation
from model_aliasing import ModelAliasRegistry
from services.attachment_manager import (
    AttachmentManager,
    AttachmentError,
    AttachmentNotFoundError,
    AttachmentValidationError,
    ResolvedAttachment,
)
from usage_tracking import UsageTracker
from word_count_utils import (
    is_word_count_enforcement_enabled,
    validate_word_count_config,
)

from .app_state import (
    FORCE_EXTRA_VERBOSE_ENV_VAR,
    FORCE_VERBOSE_ENV_VAR,
    TRUTHY_ENV_VALUES,
    _debug_record_event,
    _debug_session_start,
    _debug_update_status,
    _ensure_services,
    _is_project_cancelled,
    _is_project_reserved,
    _reserve_project_id,
    publish_project_phase_chunk,
    _start_project,
    _stop_project,
    _store_final_result,
    active_sessions,
    app,
    config,
    logger,
    mutate_session,
    pop_session,
    register_session,
    get_project_status,
    update_session_status,
)
from .generation_processor import (
    _attach_json_guard_metadata,
    _build_final_result,
    _get_final_content,
    add_verbose_log,
    process_content_generation,
    resolve_images_for_generation,
    ai_service,
)


def compute_base_effective_layers(request: "ContentRequest") -> List[str]:
    """Effective QA layers for this request.

    Returns label tokens for:
      - User-defined QA layers (by `.name`)
      - Built-in synthetic layers that QA will actually run (word_count, phrase_frequency,
        lexical_diversity, cumulative_repetition, evidence_grounding) - based on their
        INJECTION preconditions, not just validator activation.
    """
    labels: List[str] = []
    labels.extend(layer.name for layer in (request.qa_layers or []))

    word_count_enforcement = getattr(request, "word_count_enforcement", None)
    min_words = getattr(request, "min_words", None)
    max_words = getattr(request, "max_words", None)
    if (
        word_count_enforcement
        and getattr(word_count_enforcement, "enabled", False)
        and (min_words or max_words)
    ):
        labels.append("__auto_word_count_enforcement__")

    phrase_frequency = getattr(request, "phrase_frequency", None)
    phrase_frequency_active = bool(phrase_frequency and getattr(phrase_frequency, "enabled", False))
    if phrase_frequency_active:
        labels.append("__auto_phrase_frequency__")

    lexical_diversity = getattr(request, "lexical_diversity", None)
    if lexical_diversity and getattr(lexical_diversity, "enabled", False):
        labels.append("__auto_lexical_diversity__")

    if getattr(request, "cumulative_text", None) and phrase_frequency_active:
        labels.append("__auto_cumulative_repetition__")

    evidence_grounding = getattr(request, "evidence_grounding", None)
    if evidence_grounding and getattr(evidence_grounding, "enabled", False):
        labels.append("__auto_evidence_grounding__")

    return labels


MIN_ITERATION_TIMEOUT_SECONDS = 900  # 15 minutes baseline per iteration
QA_LAYER_PADDING_SECONDS = 120       # 2 minutes per QA layer
GRAN_SABIO_PADDING_SECONDS = 600     # 10 minutes buffer if Gran Sabio fallback enabled
SESSION_TIMEOUT_CAP_SECONDS = 8 * 3600  # Never recommend more than 8 hours
DEFAULT_WORD_LIMIT_TOKEN_FLOOR = 8000
ESTIMATED_TOKENS_PER_WORD = 2.2
LONG_FORM_TOKEN_BUFFER = 1024


def _get_request_fields_set(request: Any) -> set:
    """Return the set of fields explicitly provided in the incoming request."""
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(request, "__fields_set__", set())
    return set(fields_set or [])


def _estimate_session_timeout(
    request: ContentRequest,
    reasoning_timeout_hint: Optional[int],
) -> int:
    """Estimate a realistic client-side timeout considering iterations and QA."""

    per_iteration = reasoning_timeout_hint or 0
    per_iteration = max(per_iteration, MIN_ITERATION_TIMEOUT_SECONDS)

    iteration_budget = per_iteration * max(1, request.max_iterations)
    qa_layers = len(request.qa_layers) if request.qa_layers else 0
    qa_budget = max(1, qa_layers) * QA_LAYER_PADDING_SECONDS

    total = iteration_budget + qa_budget
    if getattr(request, "gran_sabio_fallback", False):
        total += GRAN_SABIO_PADDING_SECONDS

    # Apply sane bounds to avoid runaway recommendations
    return int(min(max(total, MIN_ITERATION_TIMEOUT_SECONDS), SESSION_TIMEOUT_CAP_SECONDS))


def _estimate_tokens_for_word_target(request: ContentRequest) -> Optional[int]:
    """Estimate a generation token budget from the requested word target."""

    target_words: Optional[int] = None
    if request.max_words:
        target_words = request.max_words
    elif request.min_words:
        target_words = request.min_words

    if not target_words:
        return None

    estimated_tokens = int(math.ceil(target_words * ESTIMATED_TOKENS_PER_WORD))
    estimated_tokens += LONG_FORM_TOKEN_BUFFER
    return max(DEFAULT_WORD_LIMIT_TOKEN_FLOOR, estimated_tokens)


def _build_advisory(*, code: str, message: str, severity: str = "warning") -> PreflightIssue:
    """Create a non-blocking accepted-path advisory."""

    return PreflightIssue(
        code=code,
        severity=severity,
        message=message,
        blockers=False,
    )


def _supports_long_text_generation_tools(model_name: str) -> bool:
    """Return whether the generator supports Long Text section-draft tool loops."""

    try:
        model_info = config.get_model_info(model_name)
    except Exception:
        return False

    provider = str(model_info.get("provider", "")).lower()
    model_id = str(model_info.get("model_id", "")).lower()
    if provider not in {"openai", "claude", "anthropic", "gemini", "google", "xai", "openrouter"}:
        return False
    if provider == "openai" and any(marker in model_id for marker in ["o3-pro", "gpt-5-pro"]):
        return False
    return True


def _clip_long_text_bound(
    value: int,
    *,
    request: ContentRequest,
    hard_cap_words: int,
    lower: bool,
) -> int:
    """Clip Long Text target bands against request bounds and the hard cap."""

    if lower:
        if request.min_words is not None:
            value = max(value, request.min_words)
        value = min(value, hard_cap_words)
        return max(1, value)

    if request.max_words is not None:
        value = min(value, request.max_words)
    value = min(value, hard_cap_words)
    if request.min_words is not None:
        value = max(value, request.min_words)
    return max(1, value)


def _resolve_long_text_mode(
    request: ContentRequest,
    request_fields_set: set,
) -> tuple[Dict[str, Any], List[PreflightIssue]]:
    """Resolve request-level Long Text activation and derived word bands."""

    requested_mode = getattr(request, "long_text_mode", "auto")
    hard_cap_words = int(config.LONG_TEXT_HARD_CAP_WORDS)
    user_set_max_iterations = "max_iterations" in request_fields_set
    explicit_min_words = "min_words" in request_fields_set and request.min_words is not None
    explicit_max_words = "max_words" in request_fields_set and request.max_words is not None
    advisories: List[PreflightIssue] = []

    resolved: Dict[str, Any] = {
        "enabled": False,
        "requested_mode": requested_mode,
        "activation_reason": "Long Text Mode disabled.",
        "derived_target_words": None,
        "target_min_words": None,
        "target_max_words": None,
        "emergency_min_words": None,
        "emergency_max_words": None,
        "hard_cap_words": hard_cap_words,
        "user_set_max_iterations": user_set_max_iterations,
    }

    if requested_mode == "off":
        resolved["activation_reason"] = "Long Text Mode explicitly disabled by the caller."
        return resolved, advisories

    derived_target_words: Optional[int] = None
    if request.min_words is not None and request.max_words is not None:
        derived_target_words = int(round((request.min_words + request.max_words) / 2.0))
    elif requested_mode == "on" and request.min_words is not None:
        derived_target_words = request.min_words
    elif requested_mode == "on" and request.max_words is not None:
        derived_target_words = request.max_words

    resolved["derived_target_words"] = derived_target_words

    if requested_mode == "auto":
        if not (explicit_min_words and explicit_max_words):
            reason = "Long Text Mode stayed off because auto mode requires both min_words and max_words."
            resolved["activation_reason"] = reason
            if request.min_words is not None or request.max_words is not None:
                advisories.append(
                    _build_advisory(
                        code="long_text_auto_declined_one_sided_bounds",
                        message=reason,
                    )
                )
            return resolved, advisories

        if derived_target_words is None:
            resolved["activation_reason"] = "Long Text Mode stayed off because no derived target could be resolved."
            return resolved, advisories

        if derived_target_words < config.LONG_TEXT_AUTO_MIN_WORDS:
            reason = (
                f"Long Text Mode stayed off because the derived target ({derived_target_words} words) "
                f"is below the auto-activation threshold ({config.LONG_TEXT_AUTO_MIN_WORDS})."
            )
            resolved["activation_reason"] = reason
            advisories.append(
                _build_advisory(
                    code="long_text_auto_declined_below_threshold",
                    message=reason,
                    severity="info",
                )
            )
            return resolved, advisories

    if requested_mode == "on" and derived_target_words is None:
        raise HTTPException(
            status_code=400,
            detail="long_text_mode='on' requires min_words, max_words, or both so a target can be derived.",
        )

    if derived_target_words is not None:
        if derived_target_words < config.LONG_TEXT_AUTO_MIN_WORDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode requires a derived target between {config.LONG_TEXT_AUTO_MIN_WORDS} "
                    f"and {hard_cap_words} words; got {derived_target_words}."
                ),
            )
        if derived_target_words > hard_cap_words:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode V1 is capped at {hard_cap_words} words. Split the request across "
                    "multiple runs instead of requesting a larger single document."
                ),
            )

    resolved["enabled"] = True
    resolved["activation_reason"] = (
        f"Long Text Mode enabled ({requested_mode}) with a derived target of {derived_target_words} words."
    )

    band_delta = max(1, int(round((derived_target_words or 0) * (config.LONG_TEXT_TARGET_BAND_PERCENT / 100.0))))
    emergency_delta = max(
        1,
        int(round((derived_target_words or 0) * (config.LONG_TEXT_EMERGENCY_BAND_PERCENT / 100.0))),
    )

    resolved["target_min_words"] = _clip_long_text_bound(
        (derived_target_words or 0) - band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["target_max_words"] = _clip_long_text_bound(
        (derived_target_words or 0) + band_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )
    resolved["emergency_min_words"] = _clip_long_text_bound(
        (derived_target_words or 0) - emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=True,
    )
    resolved["emergency_max_words"] = _clip_long_text_bound(
        (derived_target_words or 0) + emergency_delta,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=False,
    )

    return resolved, advisories

@app.post("/project/new", response_model=ProjectInitResponse)
async def allocate_project_id(request: Optional[ProjectInitRequest] = Body(default=None)):
    """
    Reserve or allocate a project identifier without triggering generation workflows.
    Clients may supply their own identifier or let the API generate one.
    """
    payload = request or ProjectInitRequest()
    supplied_id = payload.project_id

    if supplied_id:
        if _is_project_reserved(supplied_id):
            logger.info("GRANSABIO_MAIN: Reusing existing project_id reservation: %s", supplied_id)
        else:
            logger.info("GRANSABIO_MAIN: Reserving client-supplied project_id: %s", supplied_id)
            _reserve_project_id(supplied_id)
        return ProjectInitResponse(project_id=supplied_id)

    project_id: Optional[str] = None
    for _ in range(10):
        candidate = uuid.uuid4().hex
        if not _is_project_reserved(candidate):
            project_id = candidate
            break
    if not project_id:
        raise HTTPException(status_code=500, detail="Unable to allocate a unique project identifier. Try again.")

    logger.info("GRANSABIO_MAIN: Generated NEW project_id via /project/new: %s", project_id)
    _reserve_project_id(project_id)
    return ProjectInitResponse(project_id=project_id)


@app.post("/project/start/{project_id}")
async def start_project(project_id: str):
    """
    Activate a project to accept new generation requests.

    This should be called before starting generation for a project that may
    have been previously cancelled. If the project was not cancelled, this
    is a no-op.

    Args:
        project_id: The project identifier to activate.

    Returns:
        Status indicating whether the project was reactivated.
    """
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")

    was_cancelled = _start_project(project_id)
    status = "reactivated" if was_cancelled else "already_active"

    logger.info(
        "GRANSABIO_MAIN: Project %s %s via /project/start",
        project_id,
        "reactivated (was cancelled)" if was_cancelled else "confirmed active"
    )

    return {
        "project_id": project_id,
        "status": status,
        "was_cancelled": was_cancelled,
    }


@app.post("/project/stop/{project_id}")
async def stop_project(project_id: str):
    """
    Cancel a project and all its active generation sessions.

    This marks the project as cancelled, which will:
    1. Cancel all currently active sessions belonging to this project
    2. Reject any new /generate requests with this project_id

    The project can be reactivated later via /project/start/{project_id}.

    Args:
        project_id: The project identifier to cancel.

    Returns:
        Status including number of sessions cancelled.
    """
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")

    sessions_cancelled = _stop_project(project_id)

    logger.info(
        "GRANSABIO_MAIN: Project %s cancelled via /project/stop - %d active sessions stopped",
        project_id,
        sessions_cancelled
    )

    return {
        "project_id": project_id,
        "status": "cancelled",
        "sessions_cancelled": sessions_cancelled,
    }


@app.post("/generate", response_model=GenerationInitResponse)
async def generate_content(request: ContentRequest):
    """
    Generate content using the Gran Sabio LLM Engine

    This endpoint initiates content generation with multi-layer QA evaluation.
    Returns a session ID for tracking progress via the /status endpoint.
    """
    force_extra_verbose = os.getenv(FORCE_EXTRA_VERBOSE_ENV_VAR, "").lower() in TRUTHY_ENV_VALUES
    force_verbose = os.getenv(FORCE_VERBOSE_ENV_VAR, "").lower() in TRUTHY_ENV_VALUES

    if force_extra_verbose:
        if not request.extra_verbose or not request.verbose:
            logger.info("Force extra verbose mode enabled via runtime flag; overriding request verbosity settings.")
        request.extra_verbose = True
        request.verbose = True
    elif force_verbose:
        if not request.verbose:
            logger.info("Force verbose mode enabled via runtime flag; overriding request verbosity setting.")
        request.verbose = True

    try:
        request_payload = request.model_dump(mode="json", exclude_none=False)  # type: ignore[attr-defined]
    except AttributeError:
        request_payload = request.dict()
    #logger.info("Incoming /generate request payload: %s", json.dumps(request_payload, ensure_ascii=False))

    request_fields_set = _get_request_fields_set(request)
    resolved_long_text_mode, long_text_advisories = _resolve_long_text_mode(request, request_fields_set)
    request._resolved_long_text_mode = resolved_long_text_mode
    explicit_arbiter_model = "arbiter_model" in request_fields_set

    # Accent guard + Long Text fail-fast (Cambio 1 v5, §5.1)
    if resolved_long_text_mode.get("enabled") and request.llm_accent_guard.mode != "off":
        raise HTTPException(
            status_code=400,
            detail=(
                "llm_accent_guard is not supported when Long Text Mode is enabled. "
                "Disable accent guard or set long_text_mode to 'auto'/'off'."
            ),
        )

    # Accent post-mode on empty pipeline fail-fast (Cambio 1 v5, §5.1)
    accent_mode = request.llm_accent_guard.mode
    if accent_mode in {"post", "inline_post"}:
        base_layers = compute_base_effective_layers(request)
        if not base_layers and not request.llm_accent_guard.force_accent_with_empty_layers:
            raise HTTPException(
                status_code=400,
                detail=(
                    "accent post mode would re-enable QA on an empty pipeline; "
                    "set force_accent_with_empty_layers=true or add a qa_layer."
                ),
            )
        if (
            not base_layers
            and request.llm_accent_guard.force_accent_with_empty_layers
            and not request.qa_models
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "force_accent_with_empty_layers requires at least one qa_model "
                    "to evaluate the synthetic accent layer."
                ),
            )

    # Accent inline requires tool-calling provider (Cambio 1 v5, §5.1)
    if accent_mode in {"inline", "inline_post"}:
        try:
            gen_info = config.get_model_info(request.generator_model)
        except Exception:
            gen_info = None
        if gen_info is not None:
            provider_key = (gen_info.get("provider") or "").lower()
            model_id_lc = str(gen_info.get("model_id") or "").lower()
            supported_providers = {"openai", "claude", "anthropic", "gemini", "google", "xai", "openrouter"}
            if provider_key not in supported_providers or any(m in model_id_lc for m in ("o3-pro", "gpt-5-pro")):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "accent_guard inline mode requires a provider with tool-calling support; "
                        "this generator_model does not qualify."
                    ),
                )

    def _ensure_model_known(model_name: str, field_label: str) -> None:
        """Validate that a model exists in model_specs.json, raising HTTP 400 otherwise."""
        try:
            config.get_model_info(model_name)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field_label} '{model_name}': {exc}",
            ) from exc

    # Require explicit generator model
    if not ({"generator_model", "model"} & request_fields_set):
        raise HTTPException(
            status_code=400,
            detail="generator_model is required and must match a model declared in model_specs.json.",
        )
    _ensure_model_known(request.generator_model, "generator_model")

    # Require QA models when effective QA pipeline is non-empty or accent post/inline_post mode
    qa_required = bool(compute_base_effective_layers(request)) or request.llm_accent_guard.mode in {"post", "inline_post"}
    if qa_required:
        if "qa_models" not in request_fields_set:
            raise HTTPException(
                status_code=400,
                detail="qa_models is required when QA layers are provided.",
            )
        if not request.qa_models:
            raise HTTPException(
                status_code=400,
                detail="qa_models cannot be empty when QA layers are provided.",
            )
        qa_model_names: List[str] = []
        for model_entry in request.qa_models:
            if isinstance(model_entry, str):
                qa_model_names.append(model_entry)
            else:
                name = getattr(model_entry, "model", None)
                if not name:
                    raise HTTPException(
                        status_code=400,
                        detail="Each QA model entry must include a model name.",
                    )
                qa_model_names.append(name)
        for model_name in qa_model_names:
            _ensure_model_known(model_name, "qa_models")

    # Require Gran Sabio model when its flows can run (QA enabled or fallback requested)
    gran_sabio_needed = qa_required or bool(getattr(request, "gran_sabio_fallback", False))
    if gran_sabio_needed:
        if "gran_sabio_model" not in request_fields_set:
            raise HTTPException(
                status_code=400,
                detail="gran_sabio_model is required when QA is enabled or Gran Sabio fallback is active.",
            )
        _ensure_model_known(request.gran_sabio_model, "gran_sabio_model")

    # Handle project identifiers (optional handshake for grouping sessions)
    project_id: Optional[str] = None
    if getattr(request, "project_id", None):
        candidate = str(request.project_id).strip()
        if candidate:
            if len(candidate) > 128:
                raise HTTPException(status_code=400, detail="project_id must be 128 characters or fewer")
            project_id = candidate
            logger.info(f"GRANSABIO_MAIN: Received EXISTING project_id in request: {project_id}")

            # Check if this project has been cancelled - reject new requests
            if _is_project_cancelled(project_id):
                logger.warning(
                    "GRANSABIO_MAIN: Rejecting request for CANCELLED project_id: %s "
                    "(client should call /project/start/%s first)",
                    project_id, project_id
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Project '{project_id}' has been cancelled. "
                           f"Call POST /project/start/{project_id} to reactivate it."
                )

    # Generate session_id early so we can use it as default project_id
    # This unifies session and project streaming: when no project_id is provided,
    # the session_id becomes the project_id, allowing clients to use
    # /stream/project/{session_id} for real-time updates
    session_id = str(uuid.uuid4())

    if not project_id:
        # Unification: session_id = project_id when not explicitly provided
        project_id = session_id
        logger.info(f"GRANSABIO_MAIN: Using session_id as project_id: {project_id}")
    else:
        logger.info(f"GRANSABIO_MAIN: Using explicit project_id: {project_id}")

    if project_id:
        _reserve_project_id(project_id)
    request.project_id = project_id
    request_payload["project_id"] = project_id

    # Validate word count enforcement configuration early
    if is_word_count_enforcement_enabled(request.word_count_enforcement):
        is_valid, error_msg = validate_word_count_config(request.word_count_enforcement)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid word count enforcement configuration: {error_msg}")
        
        # Check that word limits are provided when enforcement is enabled
        if not request.min_words and not request.max_words:
            raise HTTPException(status_code=400, detail="Word count enforcement is enabled but no min_words or max_words specified")
    
    # When word targets are present and the caller did not set an explicit token budget,
    # auto-raise the default generation budget so long-form requests are not silently capped.
    if (
        (request.min_words or request.max_words)
        and request.max_tokens_percentage is None
        and "max_tokens" not in request_fields_set
    ):
        estimated_budget = _estimate_tokens_for_word_target(request)
        if estimated_budget is not None and estimated_budget > (request.max_tokens or 0):
            logger.info(
                "Auto-adjusting generation max_tokens for word-targeted request: %s -> %s "
                "(min_words=%s, max_words=%s)",
                request.max_tokens,
                estimated_budget,
                request.min_words,
                request.max_words,
            )
            request.max_tokens = estimated_budget

    json_output_requested = request.json_output or request.content_type == "json"

    usage_tracker = UsageTracker(detail_level=getattr(request, "show_query_costs", 0))

    if json_output_requested:
        fields_set = getattr(request, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(request, "__fields_set__", set())
        if "max_iterations" not in fields_set:
            adjusted_iterations = 10
            if request.max_iterations != adjusted_iterations:
                logger.info(
                    "Adjusting max_iterations for JSON content type: requested=%s, applied=%s",
                    request.max_iterations,
                    adjusted_iterations,
                )
            request.max_iterations = adjusted_iterations

    long_text_enabled = bool(resolved_long_text_mode.get("enabled"))
    long_text_tools_supported = _supports_long_text_generation_tools(request.generator_model)
    if long_text_enabled:
        if not config.has_model_spec(config.LONG_TEXT_CONTROLLER_EVAL_MODEL):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text controller eval model '{config.LONG_TEXT_CONTROLLER_EVAL_MODEL}' "
                    "is not declared in model_specs.json."
                ),
            )
        if not resolved_long_text_mode.get("user_set_max_iterations"):
            request.max_iterations = 2
        elif request.max_iterations > config.LONG_TEXT_MAX_OUTER_ITERATIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text Mode allows at most {config.LONG_TEXT_MAX_OUTER_ITERATIONS} outer iterations "
                    f"when max_iterations is set explicitly; got {request.max_iterations}."
                ),
            )

        if json_output_requested:
            raise HTTPException(
                status_code=400,
                detail="Long Text Mode only supports plain-text generation and cannot be combined with JSON output.",
            )
        if getattr(request, "target_field", None):
            raise HTTPException(
                status_code=400,
                detail="Long Text Mode does not support target_field or text_field_path requests.",
            )
        if getattr(request, "images", None):
            raise HTTPException(
                status_code=400,
                detail="Long Text Mode V1 does not support image inputs.",
            )
        if explicit_arbiter_model:
            raise HTTPException(
                status_code=400,
                detail="Long Text Mode manages structural repairs internally and does not accept an explicit arbiter_model.",
            )
        if request.generation_tools_mode == "always" and not long_text_tools_supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"generation_tools_mode='always' is not supported for generator_model "
                    f"'{request.generator_model}' in Long Text section drafting."
                ),
            )
        if request.generation_tools_mode == "auto" and not long_text_tools_supported:
            long_text_advisories.append(
                _build_advisory(
                    code="long_text_tools_auto_downgraded",
                    message=(
                        f"Generator '{request.generator_model}' does not support tool-assisted Long Text "
                        "section drafting. The request was accepted with standard streaming drafts instead."
                    ),
                    severity="info",
                )
            )

    attachment_manager: Optional[AttachmentManager] = None
    resolved_attachments: List[ResolvedAttachment] = []
    preflight_context: List[Dict[str, Any]] = []

    if request.context_documents:
        if not request.username:
            raise HTTPException(status_code=400, detail="username is required when providing context_documents")
        attachment_manager = get_attachment_manager()
        max_allowed = config.ATTACHMENTS.max_files_per_request
        if len(request.context_documents) > max_allowed:
            raise HTTPException(status_code=400, detail=f"Maximum of {max_allowed} context documents are allowed per request")
        seen_upload_ids = set()
        for ref in request.context_documents:
            if ref.username != request.username:
                raise HTTPException(status_code=403, detail="Context document does not belong to the requesting user")
            if ref.upload_id in seen_upload_ids:
                raise HTTPException(status_code=400, detail=f"Duplicate context document: {ref.upload_id}")
            seen_upload_ids.add(ref.upload_id)
            try:
                resolved = attachment_manager.resolve_attachment(
                    username=request.username,
                    upload_id=ref.upload_id,
                )
            except AttachmentValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except AttachmentNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AttachmentError as exc:
                logger.exception("Unexpected attachment resolution error", exc_info=exc)
                raise HTTPException(status_code=500, detail="Unable to access attachment content") from exc

            resolved_attachments.append(resolved)
            preflight_context.append(attachment_manager.build_preflight_summary(resolved))

        logger.info(
            "Resolved %d context documents for user %s",
            len(resolved_attachments),
            request.username,
        )

    # Validate and resolve images for vision-enabled generation
    resolved_images: List[ImageData] = []
    if request.images:
        if not request.username:
            raise HTTPException(status_code=400, detail="username is required when providing images")

        if not attachment_manager:
            attachment_manager = get_attachment_manager()

        max_images = config.IMAGE.max_images_per_request
        if len(request.images) > max_images:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum of {max_images} images allowed per request"
            )

        # Validate image references before starting generation
        seen_image_ids = set()
        for img_ref in request.images:
            if img_ref.username != request.username:
                raise HTTPException(
                    status_code=403,
                    detail="Image does not belong to the requesting user"
                )
            if img_ref.upload_id in seen_image_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate image reference: {img_ref.upload_id}"
                )
            seen_image_ids.add(img_ref.upload_id)

            # Validate image exists and is actually an image type
            try:
                resolved = attachment_manager.resolve_attachment(
                    username=request.username,
                    upload_id=img_ref.upload_id,
                )
                if not attachment_manager._is_image(resolved.record):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Attachment {img_ref.upload_id} is not an image "
                               f"(type: {resolved.record.mime_type})"
                    )
            except AttachmentValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except AttachmentNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AttachmentError as exc:
                logger.exception("Unexpected image resolution error", exc_info=exc)
                raise HTTPException(status_code=500, detail="Unable to access image content") from exc

        # Resolve all images to ImageData (fail-fast on any error)
        try:
            resolved_images = await resolve_images_for_generation(request, attachment_manager)
            logger.info(
                "Resolved %d images for vision-enabled generation, user %s",
                len(resolved_images),
                request.username,
            )
        except (AttachmentError, AttachmentNotFoundError, AttachmentValidationError) as exc:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {exc}") from exc

    # Build image_info for preflight validation (vision-enabled requests)
    preflight_image_info: Optional[Dict[str, Any]] = None
    if resolved_images:
        # Check if the generator model supports vision
        try:
            generator_model_info = config.get_model_info(request.generator_model)
            generator_capabilities = generator_model_info.get("capabilities", [])
            generator_supports_vision = "vision" in [
                c.lower() for c in generator_capabilities if isinstance(c, str)
            ]
        except Exception:
            # If we can't determine capabilities, assume vision is not supported
            generator_supports_vision = False

        # Validate vision support - reject early if model cannot process images
        # This validation runs regardless of qa_layers (preflight might be skipped)
        if not generator_supports_vision:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{request.generator_model}' does not support vision/images. "
                    f"Request includes {len(resolved_images)} image(s). "
                    "Please use a vision-capable model (e.g., gpt-4o, claude-sonnet-4, "
                    "gemini-2.5-flash) or remove the images from the request."
                )
            )

        preflight_image_info = {
            "count": len(resolved_images),
            "total_estimated_tokens": sum(
                img.estimated_tokens or 0 for img in resolved_images
            ),
            "filenames": [img.original_filename for img in resolved_images],
            "total_size_bytes": sum(img.size_bytes for img in resolved_images),
            "generator_supports_vision": generator_supports_vision,
            "detail_levels": list(set(
                img.detail for img in resolved_images if img.detail
            )) or ["auto"],
        }

    reasoning_timeout_hint: Optional[int] = None
    try:
        validation_preview = config.validate_token_limits(
            request.generator_model,
            request.max_tokens,
            getattr(request, "reasoning_effort", None),
            getattr(request, "thinking_budget_tokens", None),
            getattr(request, "max_tokens_percentage", None),
        )
        reasoning_timeout_hint = validation_preview.get("reasoning_timeout_seconds")
    except Exception as exc:
        logger.debug("Could not preview reasoning timeout for %s: %s", request.generator_model, exc)
        reasoning_timeout_hint = None

    recommended_timeout_seconds = _estimate_session_timeout(request, reasoning_timeout_hint)
    model_alias_registry = ModelAliasRegistry.from_request(
        request,
        preflight_model=getattr(config, "PREFLIGHT_VALIDATION_MODEL", None),
    )
    request._model_alias_registry = model_alias_registry

    # session_id was already generated earlier (before project_id handling)
    # Register temp session for preflight streaming
    temp_session = {
        "preflight_content": "",
        "current_phase": "preflight_validation"
    }
    await register_session(session_id, temp_session)

    # Create phase_logger for preflight validation
    preflight_phase_logger = create_phase_logger(
        session_id=session_id,
        verbose=request.verbose,
        extra_verbose=request.extra_verbose
    )

    # Define preflight streaming callback
    def preflight_stream_callback(chunk: str):
        temp_session["preflight_content"] += chunk
        if project_id and chunk:
            # Fire-and-forget to avoid blocking the sync callback
            asyncio.create_task(
                publish_project_phase_chunk(
                    project_id,
                    "preflight",
                    chunk,
                    session_id=session_id,
                    request_name=request.request_name,
                )
            )

    grounding_config = getattr(request, "evidence_grounding", None)
    grounding_enabled = bool(grounding_config and grounding_config.enabled)

    # Check if semantic QA is disabled and grounding is not active - bypass preflight if so
    if not request.qa_layers and not grounding_enabled and request.llm_accent_guard.mode == "off":
        # No QA layers means no potential contradictions - always proceed
        from preflight_validator import _analyze_word_count_conflicts
        from models import PreflightResult
        word_count_analysis = _analyze_word_count_conflicts(request)
        preflight_result = PreflightResult(
            decision="proceed",
            user_feedback="QA evaluation disabled - proceeding without preflight validation",
            summary="Bypassed preflight validation due to empty qa_layers",
            confidence=1.0,
            enable_algorithmic_word_count=word_count_analysis["enable_algorithmic_word_count"],
            duplicate_word_count_layers_to_remove=word_count_analysis["duplicate_layers_to_remove"],
        )
        temp_session["preflight_content"] += "⚡ Preflight validation bypassed - QA evaluation disabled\n"
        if project_id:
            asyncio.create_task(
                publish_project_phase_chunk(
                    project_id,
                    "preflight",
                    "⚡ Preflight validation bypassed - QA evaluation disabled\n",
                    session_id=session_id,
                    request_name=request.request_name,
                )
            )
    else:
        # Run a preflight validation to catch impossible or contradictory requirements early
        with preflight_phase_logger.phase(Phase.PREFLIGHT):
            preflight_result = await run_preflight_validation(
                ai_service,
                request,
                context_documents=preflight_context,
                image_info=preflight_image_info,
                stream_callback=preflight_stream_callback,
                usage_tracker=usage_tracker,
                phase_logger=preflight_phase_logger,
                model_alias_registry=model_alias_registry,
            )

    if long_text_enabled and not request.qa_layers and not grounding_enabled:
        long_text_advisories.append(
            _build_advisory(
                code="long_text_qa_bypass",
                message=(
                    "Long Text Mode was accepted with qa_layers=[]. Outer QA, consensus, and Gran Sabio "
                    "safeguards are bypassed for this request."
                ),
            )
        )

    if preflight_result.decision != "proceed":
        # Clean up session on rejection
        await pop_session(session_id)
        return GenerationInitResponse(
            status="preflight_rejected",
            session_id=None,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
            preflight_feedback=preflight_result,
        )

    # Extract QA layer names for status tracking (respect processing order)
    if request.qa_layers:
        qa_layer_names = [layer.name for layer in sorted(request.qa_layers, key=lambda x: getattr(x, "order", 0))]
    else:
        qa_layer_names = []

    # Initialize session with preflight content
    await register_session(session_id, {
        "status": GenerationStatus.INITIALIZING,
        "request": request,
        "request_name": getattr(request, "request_name", None),
        "created_at": datetime.now(),
        "last_activity_at": datetime.now(),
        "iterations": [],
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "verbose_log": [],
        "context_documents": preflight_context,
        "resolved_context": resolved_attachments,
        "cancelled": False,
        # New fields for separated streams
        "generation_content": "",
        "generation_content_length": 0,
        "generation_content_word_count": 0,
        "qa_content": "",
        "preflight_content": temp_session.get("preflight_content", ""),
        "current_phase": "initializing",  # initializing, generating, qa_evaluation, consensus, completed, failed
        # Store preflight analysis for later use
        "preflight_result": preflight_result,
        "recommended_timeout_seconds": recommended_timeout_seconds,
        # Gran Sabio escalation tracking
        "gran_sabio_escalations": [],  # List[str] escalation_ids
        "gran_sabio_escalation_count": 0,  # Total count for this session
        "usage_tracker": usage_tracker,
        "model_alias_registry": model_alias_registry,
        "model_alias_map_internal": model_alias_registry.internal_snapshot(),
        "model_alias_map_prompt": model_alias_registry.prompt_snapshot(),
        "show_query_costs": getattr(request, "show_query_costs", 0),
        "project_id": project_id,
        # Project status tracking fields
        "qa_models_config": request.qa_models,  # Store original QA models config
        "qa_layer_names": qa_layer_names,  # Store QA layer names
        "min_global_score": request.min_global_score,  # Store min score for consensus
        "gran_sabio_model": request.gran_sabio_model,  # Store Gran Sabio model
        # QA progress tracking (updated by generation_processor)
        "current_qa_model": None,
        "current_qa_layer": None,
        "qa_evaluations_completed": 0,
        "qa_evaluations_total": 0,
        # Consensus tracking
        "last_consensus_score": None,
        "approved": False,
        # Cached metrics for project status
        "last_generated_content_length": 0,
        "last_generated_content_word_count": 0,
        "resolved_long_text_mode": resolved_long_text_mode,
        "long_text_state": ({
            "resolved_mode": resolved_long_text_mode,
            "source_brief": None,
            "frozen_plan": None,
            "sections_by_id": {},
            "accepted_section_ids": [],
            "failed_section_ids": [],
            "pending_repair_targets": [],
            "candidate_history": [],
            "outer_feedback_digest": None,
            "last_controller_summary": None,
            "plan_invalidation_count": 0,
            "no_viable_candidate_count": 0,
            "generator_call_count": 0,
            "semantic_eval_call_count": 0,
            "consecutive_post_repair_assembly_failures": 0,
        } if long_text_enabled else None),
    })

    logger.info(f"GRANSABIO_MAIN: About to record session {session_id[:8]}... with project_id: {project_id if project_id else 'NULL'}")
    try:
        await _debug_session_start(
            session_id,
            request_payload=request,
            preflight_payload=preflight_result,
            project_id=project_id,
            attachments=resolved_attachments,
            preflight_context=[entry.get("text", "") if isinstance(entry, dict) else entry for entry in preflight_context],
        )
    except Exception as e:
        logger.error(f"GRANSABIO_MAIN: _debug_session_start failed for {session_id[:8]}..., continuing anyway: {e}")

    # Note: temp_session is automatically overwritten by register_session above
    # No cleanup needed since we reuse the same session_id

    # Start generation process in background
    asyncio.create_task(process_content_generation(
        session_id,
        request,
        resolved_attachments,
        attachment_manager,
        resolved_images,
    ))

    return GenerationInitResponse(
        status="initialized",
        session_id=session_id,
        project_id=project_id,
        request_name=getattr(request, "request_name", None),
        recommended_timeout_seconds=recommended_timeout_seconds,
        advisories=(long_text_advisories or None),
    )


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get the current status of a content generation session"""
    # Get escalations from tracker
    tracker = get_tracker()
    escalations = tracker.get_session_escalations(session_id)

    summary = await mutate_session(session_id, lambda session: {
        "session_id": session_id,
        "project_id": session.get("project_id"),
        "request_name": session.get("request_name"),
        "status": session["status"].value,
        "current_iteration": session["current_iteration"],
        "max_iterations": session["max_iterations"],
        "verbose_log": session["verbose_log"][-100:],  # Last 100 entries
        "created_at": session["created_at"].isoformat(),
        "gran_sabio_escalations": {
            "total_count": len(escalations),
            "escalations": [
                {
                    "escalation_id": esc.escalation_id,
                    "iteration": esc.iteration,
                    "layer": esc.layer_name,
                    "trigger_type": esc.trigger_type,
                    "model": esc.triggering_model,
                    "decision": esc.decision,
                    "was_real": esc.was_real_deal_breaker,
                    "duration_seconds": esc.duration_seconds
                }
                for esc in escalations
            ]
        },
        "error": session.get("error"),
        "failure_reason": (
            (session.get("final_result") or {}).get("failure_reason")
            or session.get("error")
        ),
        "approved": session.get("approved"),
    })
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@app.get("/result/{session_id}")
async def get_result(session_id: str):
    '''Get the final result of a completed or failed generation session'''

    final_payload = await mutate_session(
        session_id,
        lambda session: _build_final_result(session)
    )
    if final_payload is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status, payload = final_payload
    if status == "unfinished":
        return JSONResponse(
            status_code=202,
            content={"detail": "Generation still in progress"},
            headers={"Retry-After": "5"}
        )
    return payload


@app.post("/stop/{session_id}")
async def stop_session(session_id: str):
    '''Stop/cancel an active content generation session'''

    def _cancel(session: Dict[str, Any]):
        final_states = {
            GenerationStatus.COMPLETED,
            GenerationStatus.REJECTED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }
        if session.get("status") in final_states:
            return {
                "changed": False,
                "response": {
                    "session_id": session_id,
                    "message": f"Session already finished with status: {session['status'].value}",
                    "stopped": False,
                },
            }

        session["cancelled"] = True
        raw_content = session.get("last_generated_content", "No content generated before cancellation")
        original_request = session.get("request")
        final_result = {
            "content": _get_final_content(session, raw_content, original_request) if original_request else raw_content,
            "final_iteration": session.get("current_iteration", 0),
            "final_score": 0.0,
            "approved": False,
            "failure_reason": "Session cancelled by user",
            "generated_at": datetime.now().isoformat(),
        }
        if original_request:
            _attach_json_guard_metadata(session, final_result, original_request)
        _store_final_result(
            session,
            final_result,
            session_id,
            final_status=GenerationStatus.CANCELLED.value,
        )
        update_session_status(session, session_id, GenerationStatus.CANCELLED)
        asyncio.create_task(
            _debug_record_event(
                session_id,
                "session_cancelled",
                {
                    "final_result": final_result,
                },
            )
        )
        asyncio.create_task(
            _debug_update_status(
                session_id,
                status=GenerationStatus.CANCELLED.value,
                final_payload=final_result,
            )
        )
        return {
            "changed": True,
            "response": {
                "session_id": session_id,
                "message": "Session stopped successfully",
                "stopped": True,
                "status": GenerationStatus.CANCELLED.value,
            },
        }

    result = await mutate_session(session_id, _cancel)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if result["changed"]:
        await add_verbose_log(session_id, "Session cancelled by user request")
    return result["response"]


@app.get("/status/project/{project_id}")
async def get_project_status_endpoint(project_id: str):
    """
    Get the current status of all sessions in a project.

    Returns a JSON object with:
    - project_id: The project identifier
    - status: Overall project status (idle, running, completed, failed, cancelled)
    - sessions: List of session status objects with generation, QA, consensus, and Gran Sabio info
    - summary: Counts of total, active, and completed sessions

    This endpoint is useful for building a "control panel" UI that displays
    the current state of all generation activities within a project.
    """
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")

    status_data = await get_project_status(project_id)
    return status_data
