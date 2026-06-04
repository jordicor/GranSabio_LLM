"""
Generation and streaming API routes for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Body, HTTPException, Request
from fastapi.responses import JSONResponse

from attachments_router import get_attachment_manager
from auto_qa_planner import (
    apply_auto_qa_plan,
    run_auto_qa_planning,
    validate_auto_qa_effective_contract,
)
from deal_breaker_tracker import get_tracker
from logging_utils import create_phase_logger
from model_aliasing import ModelAliasRegistry
from model_capability_registry import (
    model_qualifies_for_inline_accent_guard,
    resolve_model_capability_context,
)
from models import (
    ContentRequest,
    GenerationInitResponse,
    GenerationStatus,
    QAModelConfig,
    PreflightIssue,
    ProjectInitRequest,
    ProjectInitResponse,
    is_json_output_requested,
)
from phrase_frequency_config import is_phrase_frequency_active
from preflight_validator import resolve_preflight_model, run_preflight_validation
from llm_routing import (
    LLMRouteResolution,
    LLMRoutingError,
    attach_request_llm_routing,
    resolve_call,
    resolve_call_models,
)
from usage_tracking import UsageTracker
from services.runtime_console import bind_console_context
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
)
from .cancellation import CancelMode, cancellation_registry
from .generation.input_resolution import resolve_generation_inputs
from .generation.prestart_flow import PrestartFlowDeps, run_prestart_flow
from .generation.session_bootstrap import SessionBootstrapDeps, finalize_generation_session
from .generation import request_admission as admission_helpers
from .generation_processor import (
    _attach_json_guard_metadata,
    _build_final_result,
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


MIN_ITERATION_TIMEOUT_SECONDS = admission_helpers.MIN_ITERATION_TIMEOUT_SECONDS
QA_LAYER_PADDING_SECONDS = admission_helpers.QA_LAYER_PADDING_SECONDS
GRAN_SABIO_PADDING_SECONDS = admission_helpers.GRAN_SABIO_PADDING_SECONDS
SESSION_TIMEOUT_CAP_SECONDS = admission_helpers.SESSION_TIMEOUT_CAP_SECONDS
DEFAULT_WORD_LIMIT_TOKEN_FLOOR = admission_helpers.DEFAULT_WORD_LIMIT_TOKEN_FLOOR
DEFAULT_MODEL_MAX_TOKENS_FALLBACK = admission_helpers.DEFAULT_MODEL_MAX_TOKENS_FALLBACK
ESTIMATED_TOKENS_PER_WORD = admission_helpers.ESTIMATED_TOKENS_PER_WORD
LONG_FORM_TOKEN_BUFFER = admission_helpers.LONG_FORM_TOKEN_BUFFER


def _get_request_fields_set(request: Any) -> set:
    """Return the set of fields explicitly provided in the incoming request."""
    return admission_helpers.get_request_fields_set(request)


def _estimate_session_timeout(
    request: ContentRequest,
    reasoning_timeout_hint: Optional[int],
) -> int:
    """Estimate a realistic client-side timeout considering iterations and QA."""
    return admission_helpers.estimate_session_timeout(request, reasoning_timeout_hint)


def _estimate_tokens_for_word_target(request: ContentRequest) -> Optional[int]:
    """Estimate a generation token budget from the requested word target."""
    return admission_helpers.estimate_tokens_for_word_target(request)


def _model_default_max_tokens(model_name: str) -> int:
    """Return model max output tokens from specs, falling back to 8192."""
    return admission_helpers.model_default_max_tokens(model_name, config)


def _apply_external_generation_min_tokens(request: ContentRequest) -> Optional[Dict[str, Any]]:
    """Apply the public-generation min token floor to a request, if configured."""
    return admission_helpers.apply_external_generation_min_tokens(
        request,
        config_obj=config,
        logger=logger,
    )


def _build_advisory(*, code: str, message: str, severity: str = "warning") -> PreflightIssue:
    """Create a non-blocking accepted-path advisory."""
    return admission_helpers.build_advisory(code=code, message=message, severity=severity)


def _apply_resolution_params(
    request: ContentRequest,
    request_fields_set: set,
    resolution: LLMRouteResolution,
) -> None:
    """Apply routed generation params unless the client set the legacy field explicitly."""

    for field_name in ("temperature", "max_tokens", "reasoning_effort", "thinking_budget_tokens"):
        if field_name in request_fields_set:
            continue
        if field_name in resolution.params:
            setattr(request, field_name, resolution.params[field_name])


def _qa_model_config_from_route(entry: Dict[str, Any]) -> QAModelConfig:
    params = dict(entry.get("params", {}) or {})
    payload: Dict[str, Any] = {"model": entry["model"]}
    for key in ("max_tokens", "reasoning_effort", "thinking_budget_tokens", "temperature"):
        if key in entry:
            payload[key] = entry[key]
        elif key in params:
            payload[key] = params[key]
    return QAModelConfig(**payload)


def _routing_warnings_as_advisories(request: ContentRequest) -> List[PreflightIssue]:
    warnings = getattr(request, "_llm_routing_warnings", []) or []
    return [
        _build_advisory(
            code="llm_routing_param_ignored",
            message=str(message),
            severity="info",
        )
        for message in warnings
    ]


def _apply_request_model_routing(request: ContentRequest, request_fields_set: set) -> None:
    """Resolve default/request LLM routing into legacy runtime fields."""

    attach_request_llm_routing(request, request_fields_set)

    generator_route = resolve_call("generation.main", request=request)
    request.generator_model = generator_route.model
    _apply_resolution_params(request, request_fields_set, generator_route)

    if "qa_models" not in request_fields_set:
        qa_route = resolve_call_models("qa.evaluate_layer", request=request)
        request.qa_models = [_qa_model_config_from_route(entry) for entry in qa_route.models]

    if "gran_sabio_model" not in request_fields_set:
        request.gran_sabio_model = resolve_call("gransabio.review", request=request).model

    if "arbiter_model" not in request_fields_set:
        request.arbiter_model = resolve_call("arbiter.resolve", request=request).model

    preflight_route = resolve_call("preflight.validate", request=request)
    request._preflight_route = preflight_route
    request._preflight_model = preflight_route.model
    request._long_text_controller_eval_model = resolve_call("long_text.semantic_eval", request=request).model


def _supports_long_text_generation_tools(model_name: str) -> bool:
    """Return whether the generator supports Long Text section-draft tool loops."""
    return admission_helpers.supports_long_text_generation_tools(model_name, config)


def _clip_long_text_bound(
    value: int,
    *,
    request: ContentRequest,
    hard_cap_words: int,
    lower: bool,
) -> int:
    """Clip Long Text target bands against request bounds and the hard cap."""
    return admission_helpers.clip_long_text_bound(
        value,
        request=request,
        hard_cap_words=hard_cap_words,
        lower=lower,
    )


def _resolve_long_text_mode(
    request: ContentRequest,
    request_fields_set: set,
) -> tuple[Dict[str, Any], List[PreflightIssue]]:
    """Resolve request-level Long Text activation and derived word bands."""
    return admission_helpers.resolve_long_text_mode(
        request,
        request_fields_set,
        config_obj=config,
    )

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

    request_fields_set = _get_request_fields_set(request)
    request._legacy_gran_sabio_model_explicit = "gran_sabio_model" in request_fields_set
    request._legacy_arbiter_model_explicit = "arbiter_model" in request_fields_set
    try:
        _apply_request_model_routing(request, request_fields_set)
    except LLMRoutingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_long_text_mode, long_text_advisories = _resolve_long_text_mode(request, request_fields_set)
    long_text_advisories.extend(_routing_warnings_as_advisories(request))
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
        specs = getattr(config, "model_specs", {}) or {}
        context = resolve_model_capability_context(request.generator_model, specs)
        if context.status == "resolved":
            if not model_qualifies_for_inline_accent_guard(
                context.provider,
                context.model_id,
                specs,
                model_data=context.model_data,
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
        if not request.qa_models:
            raise HTTPException(
                status_code=400,
                detail="qa_models cannot be empty when QA is enabled or Auto-QA is enabled. Configure qa.evaluate_layer in llm_routing.",
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
        _ensure_model_known(request.gran_sabio_model, "gran_sabio_model")

    if request.arbiter_model:
        _ensure_model_known(request.arbiter_model, "arbiter_model")

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

    bind_console_context(session_id=session_id, project_id=project_id, phase="admission")

    if project_id:
        _reserve_project_id(project_id)
    request.project_id = project_id
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

    usage_tracker = UsageTracker(
        detail_level=max(
            getattr(request, "show_query_costs", 0),
            getattr(request, "show_query_stats", 0),
        )
    )

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
    long_text_controller_eval_model = getattr(request, "_long_text_controller_eval_model", None)
    request._long_text_controller_eval_model = long_text_controller_eval_model
    if long_text_enabled:
        if not config.has_model_spec(long_text_controller_eval_model):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Long Text controller eval model '{long_text_controller_eval_model}' "
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

    input_resolution = await resolve_generation_inputs(
        request,
        attachment_manager_factory=get_attachment_manager,
        config_obj=config,
        resolve_images_for_generation_fn=resolve_images_for_generation,
        logger=logger,
    )
    attachment_manager = input_resolution.attachment_manager
    resolved_attachments = input_resolution.resolved_attachments
    resolved_images = input_resolution.resolved_images
    preflight_context = input_resolution.preflight_context
    preflight_image_info = input_resolution.preflight_image_info

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

    prestart_result = await run_prestart_flow(
        request=request,
        session_id=session_id,
        project_id=project_id,
        project_epoch=project_epoch,
        auto_qa_requested=auto_qa_requested,
        request_fields_set=request_fields_set,
        usage_tracker=usage_tracker,
        preflight_context=preflight_context,
        preflight_image_info=preflight_image_info,
        model_alias_registry=model_alias_registry,
        deps=PrestartFlowDeps(
            ai_service=ai_service,
            cancellation_registry=cancellation_registry,
            register_session=register_session,
            mutate_session=mutate_session,
            pop_session=pop_session,
            is_terminal_session=is_terminal_session,
            apply_session_cancelled_state=apply_session_cancelled_state,
            publish_project_phase_chunk=publish_project_phase_chunk,
            publish_project_session_end=publish_project_session_end,
            debug_record_event=_debug_record_event,
            create_phase_logger=create_phase_logger,
            run_auto_qa_planning=run_auto_qa_planning,
            run_preflight_validation=run_preflight_validation,
            apply_auto_qa_plan=apply_auto_qa_plan,
            validate_auto_qa_effective_contract=validate_auto_qa_effective_contract,
        ),
    )
    if prestart_result.response is not None:
        return prestart_result.response

    temp_session = prestart_result.temp_session
    preflight_result = prestart_result.preflight_result
    auto_qa_plan_payload = prestart_result.auto_qa_plan_payload

    grounding_config = getattr(request, "evidence_grounding", None)
    grounding_enabled = bool(grounding_config and grounding_config.enabled)
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

    recommended_timeout_seconds = _estimate_session_timeout(request, reasoning_timeout_hint)

    return await finalize_generation_session(
        request=request,
        session_id=session_id,
        project_id=project_id,
        project_epoch=project_epoch,
        request_payload_for_debug=request,
        preflight_context=preflight_context,
        resolved_attachments=resolved_attachments,
        attachment_manager=attachment_manager,
        resolved_images=resolved_images,
        temp_session=temp_session,
        preflight_result=preflight_result,
        recommended_timeout_seconds=recommended_timeout_seconds,
        usage_tracker=usage_tracker,
        model_alias_registry=model_alias_registry,
        resolved_long_text_mode=resolved_long_text_mode,
        long_text_enabled=long_text_enabled,
        long_text_advisories=long_text_advisories,
        auto_qa_plan_payload=auto_qa_plan_payload,
        deps=SessionBootstrapDeps(
            cancellation_registry=cancellation_registry,
            register_session=register_session,
            mutate_session=mutate_session,
            is_terminal_session=is_terminal_session,
            apply_session_cancelled_state=apply_session_cancelled_state,
            debug_session_start=_debug_session_start,
            process_content_generation=process_content_generation,
            logger=logger,
        ),
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
        "error_type": (
            (session.get("final_result") or {}).get("error_type")
            or session.get("error_type")
        ),
        "error_code": (
            (session.get("final_result") or {}).get("error_code")
            or session.get("error_code")
        ),
        "provider_error": (
            (session.get("final_result") or {}).get("provider_error")
            or session.get("provider_error")
        ),
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
