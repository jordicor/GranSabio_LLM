"""
Generation and streaming API routes for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import Body, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

import json_utils as json
from attachments_router import get_attachment_manager
from auto_qa_planner import (
    apply_auto_qa_plan,
    run_auto_qa_planning,
    validate_auto_qa_effective_contract,
)
from deal_breaker_tracker import get_tracker
from llm_routing import (
    LLMRouteResolution,
    LLMRoutingError,
    attach_request_llm_routing,
    resolve_call,
    resolve_call_models,
)
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
    PreflightIssue,
    ProjectInitRequest,
    ProjectInitResponse,
    QAModelConfig,
    is_json_output_requested,
)
from phrase_frequency_config import is_phrase_frequency_active
from preflight_validator import (
    _should_bypass_llm_preflight,
    resolve_preflight_model,
    run_preflight_validation,
)
from services.runtime_console import bind_console_context
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
    app,
    apply_session_cancelled_state,
    config,
    get_project_status,
    hard_stop_project_runtime,
    is_project_cancelled_or_stopping,
    is_terminal_session,
    logger,
    mutate_session,
    pause_project_runtime,
    pop_session,
    publish_project_session_end,
    register_session,
    session_exists,
    start_project_runtime,
    update_session_status,
)
from .cancellation import CancelMode, cancellation_registry
from .generation import request_admission as admission_helpers
from .generation.idempotency import (
    GenerationIdempotencyEntry,
    GenerationIdempotencyRegistry,
    IdempotencyCapacityError,
    IdempotencyConflictError,
    IdempotencyWaitTimeoutError,
)
from .generation.input_resolution import resolve_generation_inputs
from .generation.prestart_flow import PrestartFlowDeps, run_prestart_flow
from .generation.session_bootstrap import SessionBootstrapDeps, finalize_generation_session
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


async def _project_runtime_known(project_id: str) -> bool:
    """Return whether a project has known runtime state or a live reservation.

    Stop/pause endpoints use this to reject unknown identifiers with 404
    instead of materializing a cancelled ProjectRuntime that would linger in
    the registry (audit GEN-06). A project counts as known while the registry
    tracks it or while its identifier is still reserved (reserved via
    /project/new or referenced by any active session).
    """
    if await cancellation_registry.has_project_runtime(project_id):
        return True
    return _is_project_reserved(project_id)


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
async def stop_project(
    project_id: str,
    request: Request,
):
    """
    Cancel a project and all its active generation sessions.

    This marks the project as cancelled, which will:
    1. Cancel all currently active sessions belonging to this project
    2. Reject any new /generate requests with this project_id

    Only projects known to the current runtime can be stopped. Debug telemetry
    is intentionally not consulted as project or restart-recovery authority.

    The project can be reactivated later via /project/start/{project_id}.

    Args:
        project_id: The project identifier to cancel.
        request: Incoming request metadata.

    Returns:
        Status including number of sessions cancelled.
    """
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")
    if "mode" in request.query_params:
        raise HTTPException(status_code=400, detail="Project stop is always hard; use /project/pause for soft cancellation")
    if not await _project_runtime_known(project_id):
        raise HTTPException(status_code=404, detail="Project not found")

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
    if not await _project_runtime_known(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return await pause_project_runtime(project_id)


generation_idempotency_registry = GenerationIdempotencyRegistry(
    ttl_seconds=config.GENERATION_IDEMPOTENCY_TTL_SECONDS,
    max_entries=config.GENERATION_IDEMPOTENCY_MAX_ENTRIES,
)


class GenerationAdmissionGate:
    """Process-local bounded gate for concurrently active generation sessions.

    Enforces config.MAX_ACTIVE_SESSIONS at /generate admission (audit API-03);
    the knob was previously dead configuration. Slots are acquired atomically
    (no await between the check and the increment) before any session state is
    registered, held for the whole generation, and released exactly once: when
    the background generation task finishes, or when admission exits without
    handing the session off to one.
    """

    def __init__(self) -> None:
        self._active = 0

    @property
    def active_count(self) -> int:
        """Number of currently held admission slots."""
        return self._active

    def try_acquire(self) -> bool:
        """Take one admission slot without blocking; False when at capacity."""
        limit = max(1, int(getattr(config, "MAX_ACTIVE_SESSIONS", 100) or 100))
        if self._active >= limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        """Return one admission slot; logs when released more than acquired."""
        if self._active <= 0:
            logger.error("Generation admission gate released without a held slot")
            return
        self._active -= 1


_generation_admission_gate = GenerationAdmissionGate()

INTERNAL_FAILURE_REASON = "Internal generation error"


def _client_safe_failure_reason(
    session: Dict[str, Any],
    final_result: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return the client-safe failure reason for status/result payloads (audit GEN-07).

    Persisted final results already carry sanitized reasons. The raw
    session["error"] fallback is only served for terminal states whose reason
    is client-relevant by construction (cancellations, QA rejections, provider
    or context-admission errors); anything else degrades to a fixed message
    while the raw detail stays in server-side state and logs.
    """
    if final_result and final_result.get("failure_reason"):
        return final_result["failure_reason"]
    raw_error = session.get("error")
    if raw_error is None:
        return None
    status = session.get("status")
    status_value = status.value if isinstance(status, GenerationStatus) else str(status)
    if (
        status_value == GenerationStatus.FAILED.value
        and session.get("error_type") in (None, "internal_error")
    ):
        if raw_error != INTERNAL_FAILURE_REASON:
            logger.warning(
                "Withholding raw failure detail of session %s from the client payload",
                session.get("session_id"),
            )
        return INTERNAL_FAILURE_REASON
    return raw_error


async def _abort_admission_session(
    session_id: str,
    exc: BaseException,
    *,
    project_id: Optional[str],
    request_name: Optional[str],
) -> None:
    """Force a terminal state when generation admission dies mid-flight (audit GEN-02).

    Any uncontrolled exit between the first session registration and the
    background task handoff would otherwise leave the session stuck in
    INITIALIZING forever, pinning the full request payload in memory and its
    CancellationRegistry entry. The client-facing failure reason stays generic
    (audit GEN-07); raw exception detail is only logged server-side.
    """
    if isinstance(exc, asyncio.CancelledError):
        def _mark_aborted(session: Dict[str, Any]) -> None:
            if is_terminal_session(session):
                return
            apply_session_cancelled_state(
                session,
                session_id,
                cancel_mode=CancelMode.HARD.value,
                reason="Generation admission aborted",
                hard=True,
            )

        await mutate_session(session_id, _mark_aborted)
    else:
        logger.exception("Generation admission failed for session %s", session_id)
        error_code = type(exc).__name__

        def _mark_failed(session: Dict[str, Any]) -> None:
            if is_terminal_session(session):
                return
            final_result = {
                "content": "No content generated",
                "final_iteration": session.get("current_iteration", 0),
                "final_score": 0.0,
                "approved": False,
                "status": GenerationStatus.FAILED.value,
                "failure_reason": INTERNAL_FAILURE_REASON,
                "error_type": "internal_error",
                "error_code": error_code,
                "generated_at": datetime.now().isoformat(),
            }
            if project_id:
                final_result["project_id"] = project_id
            if request_name:
                final_result["request_name"] = request_name
            session["final_result"] = final_result
            session["error"] = INTERNAL_FAILURE_REASON
            session["error_type"] = "internal_error"
            session["error_code"] = error_code
            update_session_status(session, session_id, GenerationStatus.FAILED, "failed")

        await mutate_session(session_id, _mark_failed)
    await cancellation_registry.unregister_session(session_id)


async def _generate_content_once(request: ContentRequest):
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

    # Accent guard + Long Text fail-fast
    if resolved_long_text_mode.get("enabled") and request.llm_accent_guard.mode != "off":
        raise HTTPException(
            status_code=400,
            detail=(
                "llm_accent_guard is not supported when Long Text Mode is enabled. "
                "Disable accent guard or set long_text_mode to 'auto'/'off'."
            ),
        )

    # Accent post-mode on empty pipeline fail-fast
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

    # Accent inline requires tool-calling provider
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
    # resolved safe output budget. This avoids silently capping quality-focused
    # JSON/multimodal generations at a service-level fallback.
    if (
        request.max_tokens_percentage is None
        and (request.max_tokens is None or "max_tokens" not in request_fields_set)
    ):
        default_max_tokens = _model_default_max_tokens(request.generator_model)
        logger.info(
            "Defaulting max_tokens from output-budget policy for %s: %s -> %s",
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
        token_resolution = (
            config.resolve_output_max_tokens(
                request.generator_model,
                requested_max_tokens=estimated_budget,
                call_id="generation.word_target",
            )
            if estimated_budget is not None
            else {}
        )
        resolved_estimated_budget = token_resolution.get("max_tokens") if token_resolution else None
        if resolved_estimated_budget is not None and resolved_estimated_budget > (request.max_tokens or 0):
            logger.info(
                "Auto-adjusting generation max_tokens for word-targeted request: %s -> %s "
                "(estimated=%s, min_words=%s, max_words=%s, safe_limit=%s)",
                request.max_tokens,
                resolved_estimated_budget,
                estimated_budget,
                request.min_words,
                request.max_words,
                token_resolution.get("safe_limit"),
            )
            request.max_tokens = int(resolved_estimated_budget)

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

    # Stash the client-named QA models flag (pre-routing ground truth) so the
    # barrier-2 mirror rule can distinguish client-named from routing-chosen
    # evaluators deep inside the QA engine.
    request._client_named_qa_models = "qa_models" in request_fields_set

    # --- Barrier 1: early technical context admission (pre-session, pre-LLM) ---
    preflight_will_run = not _should_bypass_llm_preflight(
        request,
        context_documents=preflight_context,
        image_info=preflight_image_info,
    )
    auto_qa_admission_model: Optional[str] = None
    auto_qa_admission_max_tokens = 6000
    if auto_qa_requested:
        try:
            auto_qa_route = resolve_call("auto_qa.plan", request=request)
            auto_qa_admission_model = auto_qa_route.model
            auto_qa_admission_max_tokens = (
                config.resolve_output_max_tokens(
                    auto_qa_route.model,
                    routed_max_tokens=(getattr(auto_qa_route, "params", {}) or {}).get("max_tokens"),
                    call_id="auto_qa.plan",
                ).get("max_tokens")
                or 6000
            )
        except Exception as exc:
            logger.debug("Could not resolve auto_qa.plan routing for admission: %s", exc)
    preflight_admission_max_tokens = 800
    _preflight_route = getattr(request, "_preflight_route", None)
    if preflight_will_run and selected_preflight_model and _preflight_route is not None:
        try:
            preflight_admission_max_tokens = (
                config.resolve_output_max_tokens(
                    selected_preflight_model,
                    routed_max_tokens=(getattr(_preflight_route, "params", {}) or {}).get("max_tokens"),
                    call_id="preflight.validate",
                ).get("max_tokens")
                or 800
            )
        except Exception as exc:
            logger.debug("Could not resolve preflight max_tokens for admission: %s", exc)

    technical_admission = await admission_helpers.run_technical_context_admission(
        request,
        request_fields_set=request_fields_set,
        preflight_model=selected_preflight_model,
        preflight_will_run=preflight_will_run,
        auto_qa_requested=auto_qa_requested,
        auto_qa_model=auto_qa_admission_model,
        qa_mandatory=qa_required,
        preflight_image_info=preflight_image_info,
        alias_registry=model_alias_registry,
        native_counter=getattr(ai_service, "_native_token_count", None),
        preflight_max_tokens=preflight_admission_max_tokens,
        auto_qa_max_tokens=auto_qa_admission_max_tokens,
        context_documents_total_bytes=sum(
            int(item.get("size_bytes", 0) or 0) for item in (preflight_context or [])
        ),
    )
    if technical_admission.diagnostics:
        logger.info(
            "Context admission plan (session %s): %s",
            session_id,
            technical_admission.diagnostics,
        )
    if technical_admission.advisories:
        long_text_advisories.extend(technical_admission.advisories)
    if technical_admission.rejected:
        await publish_project_session_end(
            project_id,
            session_id,
            "preflight_rejected",
            request_name=getattr(request, "request_name", None),
            project_epoch=project_epoch,
        )
        return GenerationInitResponse(
            status="preflight_rejected",
            session_id=None,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
            preflight_feedback=technical_admission.preflight_result,
            advisories=long_text_advisories or None,
        )

    if not _generation_admission_gate.try_acquire():
        logger.warning(
            "Rejecting /generate admission: active generation sessions at configured capacity (%s)",
            getattr(config, "MAX_ACTIVE_SESSIONS", 100),
        )
        raise HTTPException(
            status_code=503,
            detail="Generation capacity reached: too many active sessions. Retry later.",
            headers={"Retry-After": "5"},
        )

    background_task_started = False

    def _on_background_task_started(task: asyncio.Task) -> None:
        nonlocal background_task_started
        background_task_started = True
        task.add_done_callback(lambda _finished: _generation_admission_gate.release())

    try:
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
                background_task_hook=_on_background_task_started,
            ),
        )
    except HTTPException:
        raise
    except BaseException as exc:
        await _abort_admission_session(
            session_id,
            exc,
            project_id=project_id,
            request_name=getattr(request, "request_name", None),
        )
        raise
    finally:
        if not background_task_started:
            _generation_admission_gate.release()


def _normalize_saas_request_id(value: object) -> str | None:
    """Validate the mechanically structured SaaS idempotency header."""
    if not isinstance(value, str):
        return None
    if not value or len(value) > 128:
        raise HTTPException(status_code=400, detail="Invalid SaaS request ID")
    allowed_punctuation = "._:-"
    if not all(character.isascii() and (character.isalnum() or character in allowed_punctuation) for character in value):
        raise HTTPException(status_code=400, detail="Invalid SaaS request ID")
    return value


def _generation_request_fingerprint(request: ContentRequest) -> str:
    payload = request.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _replay_idempotent_entry(entry: GenerationIdempotencyEntry) -> GenerationInitResponse:
    if entry.state == "completed" and entry.response_payload is not None:
        return GenerationInitResponse.model_validate(entry.response_payload)
    if entry.state == "error":
        raise HTTPException(
            status_code=entry.error_status_code or 500,
            detail=entry.error_detail,
            headers=entry.error_headers,
        )
    raise RuntimeError("Idempotency entry is not ready for replay")


@app.post("/generate", response_model=GenerationInitResponse)
async def generate_content(
    request: ContentRequest,
    saas_request_id: str | None = Header(
        default=None,
        alias="X-GranSabio-SaaS-Request-ID",
    ),
):
    """Start a generation, deduplicating SaaS submissions by request ID."""
    normalized_request_id = _normalize_saas_request_id(saas_request_id)
    if normalized_request_id is None:
        return await _generate_content_once(request)

    fingerprint = _generation_request_fingerprint(request)
    while True:
        try:
            claim = await generation_idempotency_registry.claim(
                normalized_request_id,
                fingerprint,
            )
        except IdempotencyConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="SaaS request ID was already used with a different generation payload",
            ) from exc
        except IdempotencyCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail="Generation idempotency registry is temporarily full",
            ) from exc

        if claim.action == "replay":
            return _replay_idempotent_entry(claim.entry)
        if claim.action == "wait":
            try:
                await generation_idempotency_registry.wait(claim.entry)
            except IdempotencyWaitTimeoutError:
                return JSONResponse(
                    status_code=202,
                    content={
                        "detail": (
                            "The original request with this SaaS request ID is still "
                            "in progress; retry later with the same request ID."
                        ),
                        "request_id": normalized_request_id,
                    },
                    headers={"Retry-After": "5"},
                )
            continue

        try:
            response = await _generate_content_once(request)
            validated_response = GenerationInitResponse.model_validate(response)
        except HTTPException as exc:
            await generation_idempotency_registry.fail(
                claim.entry,
                status_code=exc.status_code,
                detail=exc.detail,
                headers=exc.headers,
            )
            raise
        except BaseException:
            await generation_idempotency_registry.abort(claim.entry)
            raise

        await generation_idempotency_registry.complete(
            claim.entry,
            validated_response.model_dump(mode="json"),
        )
        return validated_response


@app.get("/api/saas/requests/{request_id}")
async def get_saas_generation_request(request_id: str):
    """Return an internal reconciliation snapshot for one SaaS submission."""
    normalized_request_id = _normalize_saas_request_id(request_id)
    if normalized_request_id is None:
        raise HTTPException(status_code=400, detail="Invalid SaaS request ID")
    return await generation_idempotency_registry.lookup(normalized_request_id)


def _verbose_log_window(
    session: Dict[str, Any],
    verbose_since: Optional[int],
) -> Dict[str, Any]:
    """Build the verbose-log slice and its cursor metadata (C12).

    The engine keeps a ring buffer of at most VERBOSE_MAX_ENTRIES entries;
    ``verbose_log_next_offset`` counts every entry ever appended and never
    rewinds, so the offset of the oldest retained entry is exactly
    ``next_offset - len(buffer)``. Pruning happens on the append path and in
    periodic maintenance, and both only drop the oldest entries, which is what
    lets a caller poll with ``?verbose_since=<seq>`` and receive only what is
    new. A caller whose cursor fell behind the retained window learns it
    through ``verbose_gap``/``verbose_dropped`` instead of silently skipping
    entries; the complete history lives in the debugger, never in engine
    memory.
    """
    buffer = session.get("verbose_log") or []
    next_offset = int(session.get("verbose_log_next_offset", len(buffer)))
    base_offset = next_offset - len(buffer)

    if verbose_since is None:
        returned_from = base_offset
        entries = list(buffer)
        dropped = 0
    else:
        requested = max(0, int(verbose_since))
        returned_from = max(requested, base_offset)
        start_index = min(max(returned_from - base_offset, 0), len(buffer))
        entries = buffer[start_index:]
        returned_from = next_offset - len(entries)
        dropped = max(0, base_offset - requested)

    return {
        "verbose_log": entries,
        "verbose_returned_from": returned_from,
        "verbose_base_offset": base_offset,
        "verbose_next_offset": next_offset,
        "verbose_gap": dropped > 0,
        "verbose_dropped": dropped,
    }


@app.get("/status/{session_id}")
async def get_status(
    session_id: str,
    verbose_since: Annotated[
        Optional[int],
        Query(
            ge=0,
            description=(
                "Return only verbose log entries from this sequence number "
                "onwards. Use the verbose_next_offset of the previous response "
                "as the cursor"
            ),
        ),
    ] = None,
):
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
        **_verbose_log_window(session, verbose_since),
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
        "finish_reason": (
            (session.get("final_result") or {}).get("finish_reason")
            or session.get("finish_reason")
        ),
        "finish_reason_category": (
            (session.get("final_result") or {}).get("finish_reason_category")
            or session.get("finish_reason_category")
        ),
        "retryable": (
            (session.get("final_result") or {}).get("retryable")
            if (session.get("final_result") or {}).get("retryable") is not None
            else session.get("retryable")
        ),
        "provider_error": (
            (session.get("final_result") or {}).get("provider_error")
            or session.get("provider_error")
        ),
        "context_admission": (
            (session.get("final_result") or {}).get("context_admission")
            or session.get("context_admission")
        ),
        "failure_reason": _client_safe_failure_reason(
            session,
            session.get("final_result"),
        ),
        "approved": session.get("approved"),
    })
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return summary


@app.get("/result/{session_id}")
async def get_result(session_id: str):
    '''Get the final result of a completed or failed generation session'''

    def _result_payload(session: Dict[str, Any]):
        built = _build_final_result(session)
        if (
            built[0] == "ok"
            and isinstance(built[1], dict)
            and "final_result" not in session
        ):
            # Legacy fallback serves session["error"] verbatim; keep raw
            # internal failure detail out of the client payload (audit GEN-07).
            built[1]["failure_reason"] = _client_safe_failure_reason(session, None)
            if session.get("error_code") and "error_code" not in built[1]:
                built[1]["error_code"] = session["error_code"]
        return built

    final_payload = await mutate_session(session_id, _result_payload)
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
    if not await session_exists(session_id) and not await cancellation_registry.has_session_runtime(
        session_id
    ):
        raise HTTPException(status_code=404, detail="Session not found")

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
