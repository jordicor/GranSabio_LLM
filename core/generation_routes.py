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

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from ai_service import AIService
from attachments_router import get_attachment_manager
from auto_qa_planner import (
    AutoQAPlanningError,
    apply_auto_qa_plan,
    run_auto_qa_planning,
    validate_auto_qa_effective_contract,
)
from deal_breaker_tracker import get_tracker
from logging_utils import Phase, create_phase_logger
from model_aliasing import ModelAliasRegistry
from models import (
    ContentRequest,
    GenerationInitResponse,
    GenerationStatus,
    ImageData,
    PreflightIssue,
    ProjectInitRequest,
    ProjectInitResponse,
    is_json_output_requested,
)
from phrase_frequency_config import is_phrase_frequency_active
from preflight_validator import resolve_preflight_model, run_preflight_validation
from services.attachment_manager import (
    AttachmentError,
    AttachmentManager,
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
    _is_project_reserved,
    _reserve_project_id,
    _store_final_result,
    apply_session_cancelled_state,
    app,
    config,
    get_project_status,
    hard_stop_project_runtime,
    is_project_cancelled_or_stopping,
    is_terminal_session,
    logger,
    mutate_session,
    pop_session,
    publish_project_phase_chunk,
    publish_project_session_end,
    register_session,
    pause_project_runtime,
    start_project_runtime,
    update_session_status,
)
from .cancellation import CancelMode, CancellationToken, cancellation_registry
from .generation_processor import (
    _attach_json_guard_metadata,
    _build_final_result,
    _get_final_content,
    add_verbose_log,
    ai_service,
    process_content_generation,
    resolve_images_for_generation,
)

_session_hard_stop_completion_tasks: set[asyncio.Task] = set()


def _track_session_hard_stop_task(task: asyncio.Task) -> None:
    _session_hard_stop_completion_tasks.add(task)
    task.add_done_callback(_session_hard_stop_completion_tasks.discard)


def _log_session_hard_stop_task_result(task: asyncio.Task) -> None:
    task_name = task.get_name() if hasattr(task, "get_name") else "session-hard-stop"
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("Detached session hard-stop task was cancelled: %s", task_name)
    except Exception:
        logger.exception("Detached session hard-stop task failed: %s", task_name)


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
    phrase_frequency_active = is_phrase_frequency_active(
        phrase_frequency,
        context="effective layer computation",
    )
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
DEFAULT_MODEL_MAX_TOKENS_FALLBACK = 8192
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


def _model_default_max_tokens(model_name: str) -> int:
    """Return model max output tokens from specs, falling back to 8192."""

    try:
        model_info = config.get_model_info(model_name)
    except Exception:
        return DEFAULT_MODEL_MAX_TOKENS_FALLBACK

    value = model_info.get("output_tokens")
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MODEL_MAX_TOKENS_FALLBACK
    return tokens if tokens > 0 else DEFAULT_MODEL_MAX_TOKENS_FALLBACK


def _apply_external_generation_min_tokens(request: ContentRequest) -> Optional[Dict[str, Any]]:
    """Apply the public-generation min token floor to a request, if configured."""

    adjustment = config.apply_external_generation_min_tokens(
        request.generator_model,
        request.max_tokens,
        getattr(request, "reasoning_effort", None),
        getattr(request, "thinking_budget_tokens", None),
    )
    if not adjustment.get("was_adjusted"):
        return None

    request.max_tokens = int(adjustment["adjusted_tokens"])
    request._external_generation_min_tokens_adjustment = adjustment
    logger.info(
        "Raised external generation max_tokens for %s: %s -> %s "
        "(floor=%s, source=%s, safe_limit=%s)",
        request.generator_model,
        adjustment.get("original_tokens"),
        adjustment.get("adjusted_tokens"),
        adjustment.get("min_tokens"),
        adjustment.get("source"),
        adjustment.get("safe_limit"),
    )
    return adjustment


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
    if provider == "openai" and AIService._is_openai_responses_api_model(model_id):
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

    try:
        start_result = await start_project_runtime(project_id)
    except asyncio.CancelledError as exc:
        raise HTTPException(status_code=409, detail="Project hard stop is still in progress") from exc
    was_cancelled = bool(start_result.get("was_cancelled"))
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
        "project_epoch": start_result.get("project_epoch"),
    }


@app.post("/project/stop/{project_id}")
async def stop_project(project_id: str, request: Request):
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
    if "mode" in request.query_params:
        raise HTTPException(status_code=400, detail="Project stop is always hard; use /project/pause for soft cancellation")

    result = await hard_stop_project_runtime(project_id)

    logger.info(
        "GRANSABIO_MAIN: Project %s cancelled via /project/stop - %d active sessions stopped",
        project_id,
        result.get("sessions_cancelled", 0),
    )

    return result


@app.post("/project/pause/{project_id}")
async def pause_project(project_id: str, request: Request):
    """Cooperatively pause a project and block new work until /project/start."""
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")
    if "mode" in request.query_params:
        raise HTTPException(status_code=400, detail="Project pause is always soft; use /project/stop for hard cancellation")
    return await pause_project_runtime(project_id)


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
            if provider_key not in supported_providers or (
                provider_key == "openai" and AIService._is_openai_responses_api_model(model_id_lc)
            ):
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

    # If the caller did not provide an output budget, default to the model's
    # declared max output capacity. This avoids silently capping quality-focused
    # JSON/multimodal generations at the legacy 4000-token fallback.
    if (
        request.max_tokens_percentage is None
        and (request.max_tokens is None or "max_tokens" not in request_fields_set)
    ):
        default_max_tokens = _model_default_max_tokens(request.generator_model)
        logger.info(
            "Defaulting max_tokens from model specs for %s: %s -> %s",
            request.generator_model,
            request.max_tokens,
            default_max_tokens,
        )
        request.max_tokens = default_max_tokens

    auto_qa_config = getattr(request, "auto_qa", None)
    auto_qa_requested = bool(auto_qa_config and auto_qa_config.enabled)
    if (
        auto_qa_requested
        and request.qa_layers
        and getattr(auto_qa_config, "manual_layer_policy", "reject") == "reject"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Auto-QA is enabled and the request also includes manual qa_layers. "
                "Use auto_qa.manual_layer_policy='replace'/'merge' or disable Auto-QA."
            ),
        )

    # Require QA models when the effective QA pipeline is non-empty, accent post mode can run, or Auto-QA can create layers.
    qa_required = (
        bool(compute_base_effective_layers(request))
        or request.llm_accent_guard.mode in {"post", "inline_post"}
        or auto_qa_requested
    )
    if qa_required:
        if "qa_models" not in request_fields_set:
            raise HTTPException(
                status_code=400,
                detail="qa_models is required when QA is enabled or Auto-QA is enabled.",
            )
        if not request.qa_models:
            raise HTTPException(
                status_code=400,
                detail="qa_models cannot be empty when QA is enabled or Auto-QA is enabled.",
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

            # Check if this project has been stopped/paused - reject new requests
            if await is_project_cancelled_or_stopping(project_id):
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
    try:
        project_epoch = await cancellation_registry.begin_project_admission(project_id)
    except asyncio.CancelledError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Project '{project_id}' is stopped or paused. Call POST /project/start/{project_id} to reactivate it.",
        ) from exc

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

    if request.max_tokens_percentage is None:
        token_floor_adjustment = _apply_external_generation_min_tokens(request)
        if token_floor_adjustment:
            long_text_advisories.append(
                _build_advisory(
                    code="external_generation_min_tokens_applied",
                    message=(
                        f"max_tokens was raised from {token_floor_adjustment.get('original_tokens')} "
                        f"to {token_floor_adjustment.get('adjusted_tokens')} for "
                        f"{request.generator_model} because the configured external generation "
                        "minimum is higher than the requested budget."
                    ),
                    severity="info",
                )
            )

    json_output_requested = is_json_output_requested(request)

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

    selected_preflight_model = resolve_preflight_model(request)
    model_alias_registry = ModelAliasRegistry.from_request(
        request,
        preflight_model=selected_preflight_model,
    )
    request._model_alias_registry = model_alias_registry

    # session_id was already generated earlier (before project_id handling)
    # Register temp session for preflight streaming
    now = datetime.now()
    temp_session = {
        "status": GenerationStatus.INITIALIZING,
        "session_id": session_id,
        "project_id": project_id,
        "project_epoch": project_epoch,
        "request": request,
        "request_name": getattr(request, "request_name", None),
        "created_at": now,
        "last_activity_at": now,
        "iterations": [],
        "current_iteration": 0,
        "max_iterations": request.max_iterations,
        "verbose_log": [],
        "cancelled": False,
        "cancel_mode": None,
        "hard_cancelled": False,
        "provider_calls_closed": 0,
        "tasks_cancelled": 0,
        "late_writes_blocked": 0,
        "auto_qa_content": "",
        "preflight_content": "",
        "generation_content": "",
        "qa_content": "",
        "current_phase": "auto_qa_planning" if auto_qa_requested else "preflight_validation",
        "usage_tracker": usage_tracker,
    }
    try:
        await cancellation_registry.register_session(session_id, project_id, project_epoch)
    except asyncio.CancelledError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Project '{project_id}' is stopped or paused. Call POST /project/start/{project_id} to reactivate it.",
        ) from exc
    await register_session(session_id, temp_session)
    pre_start_cancellation_token = CancellationToken(
        session_id=session_id,
        project_id=project_id,
        phase="pre_start",
        operation="generate_admission",
        registry=cancellation_registry,
    )

    async def _cancel_before_background_start(
        reason: str,
        *,
        cancel_mode: str = CancelMode.HARD.value,
        hard: bool = True,
    ) -> GenerationInitResponse:
        def _cancel_temp(session: Dict[str, Any]) -> None:
            if is_terminal_session(session) and session.get("final_result") is not None:
                return
            apply_session_cancelled_state(
                session,
                session_id,
                cancel_mode=cancel_mode,
                reason=reason,
                hard=hard,
            )

        await mutate_session(session_id, _cancel_temp)
        return GenerationInitResponse(
            status="cancelled",
            session_id=session_id,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
        )

    async def _cancel_pre_start_from_token(reason: str) -> GenerationInitResponse:
        if await cancellation_registry.is_soft_cancelled(session_id):
            return await _cancel_before_background_start(
                reason,
                cancel_mode=CancelMode.SOFT.value,
                hard=False,
            )
        return await _cancel_before_background_start(reason)

    async def _pre_start_cancel_response_if_requested(reason: str) -> Optional[GenerationInitResponse]:
        if await cancellation_registry.is_cancelled(session_id):
            return await _cancel_pre_start_from_token(reason)
        return None

    try:
        await cancellation_registry.register_current_task(session_id, "generate_pre_start")
    except asyncio.CancelledError:
        return await _cancel_pre_start_from_token("Generation cancelled before preflight start")

    # Create phase_logger for Auto-QA planning and preflight validation
    preflight_phase_logger = create_phase_logger(
        session_id=session_id,
        verbose=request.verbose,
        extra_verbose=request.extra_verbose
    )

    auto_qa_plan_payload: Optional[Dict[str, Any]] = None

    def auto_qa_stream_callback(chunk: str):
        if temp_session.get("hard_cancelled"):
            return
        temp_session["auto_qa_content"] += chunk
        if project_id and chunk:
            asyncio.create_task(
                publish_project_phase_chunk(
                    project_id,
                    "auto_qa",
                    chunk,
                    session_id=session_id,
                    project_epoch=project_epoch,
                    request_name=request.request_name,
                )
            )

    if auto_qa_requested:
        try:
            await _debug_record_event(
                session_id,
                "auto_qa_started",
                {
                    "rigor": request.auto_qa.rigor,
                    "manual_layer_policy": request.auto_qa.manual_layer_policy,
                },
            )
            with preflight_phase_logger.phase(Phase.AUTO_QA):
                auto_qa_plan = await run_auto_qa_planning(
                    ai_service,
                    request,
                    context_documents=preflight_context,
                    image_info=preflight_image_info,
                    model_alias_registry=model_alias_registry,
                    stream_callback=auto_qa_stream_callback,
                    usage_callback=usage_tracker.create_callback(
                        phase="auto_qa",
                        role="gran_sabio",
                        operation="auto_qa_planning",
                    ),
                    phase_logger=preflight_phase_logger,
                    cancellation_token=pre_start_cancellation_token,
                )
            await _debug_record_event(
                session_id,
                "auto_qa_completed",
                {
                    "layer_count": len(auto_qa_plan.qa_layers),
                    "layer_names": list(auto_qa_plan.generated_layer_names),
                    "warnings": list(auto_qa_plan.warnings),
                },
            )
            apply_auto_qa_plan(
                request,
                auto_qa_plan,
                request_fields_set=request_fields_set,
            )
            auto_qa_plan_payload = auto_qa_plan.public_dict()
            await _debug_record_event(
                session_id,
                "auto_qa_plan_applied",
                auto_qa_plan_payload,
            )
            temp_session["current_phase"] = "preflight_validation"
        except asyncio.CancelledError:
            return await _cancel_pre_start_from_token("Generation cancelled during Auto-QA planning")
        except AutoQAPlanningError as exc:
            feedback = exc.to_feedback()
            await _debug_record_event(session_id, "auto_qa_failed", feedback)
            cancel_response = await _pre_start_cancel_response_if_requested(
                "Generation cancelled before Auto-QA rejection"
            )
            if cancel_response is not None:
                return cancel_response
            await publish_project_session_end(
                project_id,
                session_id,
                "auto_qa_rejected",
                request_name=getattr(request, "request_name", None),
                project_epoch=project_epoch,
            )
            await pop_session(session_id)
            return GenerationInitResponse(
                status="auto_qa_rejected",
                session_id=None,
                project_id=project_id,
                request_name=getattr(request, "request_name", None),
                auto_qa_feedback=feedback,
                auto_qa_plan=auto_qa_plan_payload,
            )

    # Define preflight streaming callback
    def preflight_stream_callback(chunk: str):
        if temp_session.get("hard_cancelled"):
            return
        temp_session["preflight_content"] += chunk
        if project_id and chunk:
            # Fire-and-forget to avoid blocking the sync callback
            asyncio.create_task(
                publish_project_phase_chunk(
                    project_id,
                    "preflight",
                    chunk,
                    session_id=session_id,
                    project_epoch=project_epoch,
                    request_name=request.request_name,
                )
            )

    grounding_config = getattr(request, "evidence_grounding", None)
    grounding_enabled = bool(grounding_config and grounding_config.enabled)

    # Run preflight validation for every request. Requests without semantic QA still
    # require the LLM preflight gate so model/configuration failures are fail-closed.
    try:
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
                cancellation_token=pre_start_cancellation_token,
            )
    except asyncio.CancelledError:
        return await _cancel_pre_start_from_token("Generation cancelled during preflight validation")

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
        if auto_qa_requested and auto_qa_plan_payload is not None:
            await _debug_record_event(
                session_id,
                "auto_qa_plan_rejected_by_preflight",
                {
                    "preflight_decision": preflight_result.decision,
                    "auto_qa_plan": auto_qa_plan_payload,
                },
            )
        cancel_response = await _pre_start_cancel_response_if_requested(
            "Generation cancelled before preflight rejection"
        )
        if cancel_response is not None:
            return cancel_response
        await publish_project_session_end(
            project_id,
            session_id,
            "preflight_rejected",
            request_name=getattr(request, "request_name", None),
            project_epoch=project_epoch,
        )
        # Clean up session on rejection
        await pop_session(session_id)
        return GenerationInitResponse(
            status="preflight_rejected",
            session_id=None,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
            preflight_feedback=preflight_result,
            auto_qa_plan=auto_qa_plan_payload,
        )

    if auto_qa_requested:
        try:
            validate_auto_qa_effective_contract(
                request,
                preflight_result=preflight_result,
            )
            removed_auto_qa_layers = getattr(request, "_auto_qa_removed_by_preflight", []) or []
            if removed_auto_qa_layers and auto_qa_plan_payload is not None:
                auto_qa_plan_payload["removed_by_preflight"] = list(removed_auto_qa_layers)
        except AutoQAPlanningError as exc:
            feedback = exc.to_feedback()
            removed_auto_qa_layers = getattr(request, "_auto_qa_removed_by_preflight", []) or []
            if removed_auto_qa_layers and auto_qa_plan_payload is not None:
                auto_qa_plan_payload["removed_by_preflight"] = list(removed_auto_qa_layers)
            await _debug_record_event(
                session_id,
                "auto_qa_plan_rejected_by_preflight",
                {
                    "feedback": feedback,
                    "removed_layers": list(removed_auto_qa_layers),
                    "auto_qa_plan": auto_qa_plan_payload,
                },
            )
            cancel_response = await _pre_start_cancel_response_if_requested(
                "Generation cancelled before Auto-QA contract rejection"
            )
            if cancel_response is not None:
                return cancel_response
            await publish_project_session_end(
                project_id,
                session_id,
                "auto_qa_rejected",
                request_name=getattr(request, "request_name", None),
                project_epoch=project_epoch,
            )
            await pop_session(session_id)
            return GenerationInitResponse(
                status="auto_qa_rejected",
                session_id=None,
                project_id=project_id,
                request_name=getattr(request, "request_name", None),
                preflight_feedback=preflight_result,
                auto_qa_feedback=feedback,
                auto_qa_plan=auto_qa_plan_payload,
            )

    recommended_timeout_seconds = _estimate_session_timeout(request, reasoning_timeout_hint)

    # Extract QA layer names for status tracking (respect processing order)
    if request.qa_layers:
        qa_layer_names = [layer.name for layer in sorted(request.qa_layers, key=lambda x: getattr(x, "order", 0))]
    else:
        qa_layer_names = []

    # Initialize session with preflight content
    try:
        await cancellation_registry.validate_project_admission(project_id, project_epoch)
        if await cancellation_registry.is_hard_cancelled(session_id):
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        return await _cancel_pre_start_from_token("Generation cancelled before start")
    if await cancellation_registry.is_soft_cancelled(session_id):
        return await _cancel_before_background_start(
            "Generation paused before start",
            cancel_mode=CancelMode.SOFT.value,
            hard=False,
        )

    await register_session(session_id, {
        "status": GenerationStatus.INITIALIZING,
        "session_id": session_id,
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
        "cancel_mode": None,
        "hard_cancelled": False,
        "provider_calls_closed": 0,
        "tasks_cancelled": 0,
        "late_writes_blocked": 0,
        # New fields for separated streams
        "generation_content": "",
        "generation_content_length": 0,
        "generation_content_word_count": 0,
        "qa_content": "",
        "auto_qa_content": temp_session.get("auto_qa_content", ""),
        "auto_qa_plan": auto_qa_plan_payload,
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
        "project_epoch": project_epoch,
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

    # Start generation process in background under cancellation registry control.
    task = await cancellation_registry.create_task(
        session_id,
        "process_content_generation",
        lambda: process_content_generation(
            session_id,
            request,
            resolved_attachments,
            attachment_manager,
            resolved_images,
        ),
    )
    if task is None:
        return GenerationInitResponse(
            status="cancelled",
            session_id=session_id,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
            recommended_timeout_seconds=recommended_timeout_seconds,
            advisories=(long_text_advisories or None),
            auto_qa_plan=auto_qa_plan_payload,
        )

    return GenerationInitResponse(
        status="initialized",
        session_id=session_id,
        project_id=project_id,
        request_name=getattr(request, "request_name", None),
        recommended_timeout_seconds=recommended_timeout_seconds,
        advisories=(long_text_advisories or None),
        auto_qa_plan=auto_qa_plan_payload,
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


async def _hard_stop_session_runtime(session_id: str) -> Dict[str, Any]:
    seal = await cancellation_registry.seal_session_for_hard_cancel(session_id)
    cancel_result = await cancellation_registry.request_hard_cancel(session_id)

    def _cancel(session: Dict[str, Any]):
        final_states = {
            GenerationStatus.COMPLETED,
            GenerationStatus.REJECTED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }
        already_terminal = session.get("status") in final_states and not session.get("cancel_mode") == CancelMode.SOFT.value
        if already_terminal:
            status = session.get("status")
            status_value = status.value if hasattr(status, "value") else str(status)
            return {
                "changed": False,
                "final_result": session.get("final_result"),
                "response": {
                    "session_id": session_id,
                    "message": f"Session already finished with status: {status_value}",
                    "stopped": False,
                    "status": status_value,
                    "tasks_cancelled": cancel_result.get("tasks_cancelled", 0),
                    "provider_calls_closed": cancel_result.get("provider_calls_closed", 0),
                },
            }

        final_result = apply_session_cancelled_state(
            session,
            session_id,
            cancel_mode=CancelMode.HARD.value,
            reason="Session hard-stopped by user",
            hard=True,
        )
        session["tasks_cancelled"] = cancel_result.get("tasks_cancelled", 0)
        session["provider_calls_closed"] = cancel_result.get("provider_calls_closed", 0)
        original_request = session.get("request")
        if original_request:
            _attach_json_guard_metadata(session, final_result, original_request)
        return {
            "changed": True,
            "final_result": final_result,
            "response": {
                "session_id": session_id,
                "message": "Session hard-stopped successfully",
                "stopped": True,
                "status": GenerationStatus.CANCELLED.value,
                "cancel_mode": CancelMode.HARD.value,
                "tasks_cancelled": cancel_result.get("tasks_cancelled", 0),
                "provider_calls_closed": cancel_result.get("provider_calls_closed", 0),
            },
        }

    result = await mutate_session(session_id, _cancel)
    if result is None:
        if seal.get("tasks") or seal.get("provider_calls"):
            return {
                "session_id": session_id,
                "message": "Session hard-stop requested",
                "stopped": True,
                "status": GenerationStatus.CANCELLED.value,
                "cancel_mode": CancelMode.HARD.value,
                "tasks_cancelled": cancel_result.get("tasks_cancelled", 0),
                "provider_calls_closed": cancel_result.get("provider_calls_closed", 0),
            }
        await cancellation_registry.unregister_session(session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    if result["changed"]:
        await add_verbose_log(session_id, "Session hard-stopped by user request")
        await _debug_record_event(
            session_id,
            "session_hard_stopped",
            {"final_result": result["final_result"]},
        )
        await _debug_update_status(
            session_id,
            status=GenerationStatus.CANCELLED.value,
            final_payload=result["final_result"],
        )
    return result["response"]


@app.post("/stop/{session_id}")
async def stop_session(session_id: str, request: Request):
    """Hard-stop an active content generation session."""
    if "mode" in request.query_params:
        raise HTTPException(status_code=400, detail="Stop is always hard; use /pause for soft cancellation")

    completion_task = asyncio.create_task(
        _hard_stop_session_runtime(session_id),
        name=f"session-hard-stop:{session_id}",
    )
    _track_session_hard_stop_task(completion_task)
    try:
        return await asyncio.shield(completion_task)
    except asyncio.CancelledError:
        completion_task.add_done_callback(_log_session_hard_stop_task_result)
        raise


@app.post("/pause/{session_id}")
async def pause_session(session_id: str):
    """Cooperatively pause an active content generation session."""
    await cancellation_registry.request_soft_cancel(session_id)

    def _pause(session: Dict[str, Any]):
        if is_terminal_session(session):
            status = session.get("status")
            status_value = status.value if hasattr(status, "value") else str(status)
            return {
                "session_id": session_id,
                "paused": False,
                "status": status_value,
                "cancel_mode": session.get("cancel_mode"),
                "final_result": session.get("final_result"),
            }
        if session.get("cancelled") and session.get("cancel_mode") == CancelMode.HARD.value:
            return {
                "session_id": session_id,
                "paused": False,
                "status": GenerationStatus.CANCELLED.value,
                "cancel_mode": CancelMode.HARD.value,
            }
        final_result = apply_session_cancelled_state(
            session,
            session_id,
            cancel_mode=CancelMode.SOFT.value,
            reason="Session paused by user",
            hard=False,
        )
        return {
            "session_id": session_id,
            "paused": True,
            "status": GenerationStatus.CANCELLED.value,
            "cancel_mode": CancelMode.SOFT.value,
            "final_result": final_result,
        }

    result = await mutate_session(session_id, _pause)
    if result is None:
        await cancellation_registry.unregister_session(session_id)
        raise HTTPException(status_code=404, detail="Session not found")
    if result.get("paused"):
        await add_verbose_log(session_id, "Session paused by user request")
    return result


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
