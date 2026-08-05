"""
Streaming endpoints for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, FrozenSet, List, Optional
from uuid import uuid4

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

import json_utils as json
from ai_service import StreamChunk, is_content_filter_stop
from attachments_router import get_attachment_manager
from config import config
from content_events import (
    GENERATION_SESSION_PHASES,
    GRAN_SABIO_SESSION_PHASES,
    QA_SESSION_PHASES,
    ContentEvent,
    ContentEventTemplate,
    ContentPhase,
    ContentStatus,
    ContentSubphase,
    SessionPhase,
    live_attempt_key,
    parse_visibility,
)
from content_stream import flush_content_events, publish_content_event
from llm_routing import resolve_call
from models import GenerationStatus
from request_timeouts import (
    get_request_default_timeout,
    get_request_timeout_field,
    resolve_request_timeout,
)
from services.attachment_manager import AttachmentError
from services.project_stream import (
    ProjectStreamManager,
    SessionStreamManager,
    SubscriptionError,
    parse_phases,
)
from services.runtime_console import bind_console_context, reset_console_context
from word_count_utils import build_word_count_instructions

# Route order is a contract, not an accident: /stream/console and
# /stream/{session_id} have the same shape, and FastAPI resolves them in
# registration order. Importing the console routes HERE guarantees the literal
# path is registered before the catch-all below, whatever order the package
# imports its route modules in.
from . import console_routes  # noqa: F401
from .app_state import (
    _ensure_services,
    app,
    logger,
    mutate_session,
)
from .cancellation import CancellationToken, cancellation_registry
from .generation_processor import ai_service, build_context_prompt

# ---------------------------------------------------------------------------
# Direct stream hardening (API-04 / API-09)
# ---------------------------------------------------------------------------

# Maximum number of paid direct-stream runs allowed per session.
MAX_DIRECT_STREAMS_PER_SESSION = 3
_DIRECT_STREAM_CONTENT_FILTER_TERMINAL = (
    b"Error: Provider content policy stopped generation."
)
# Subphase carried by the labelled copy of the raw direct-stream chunks.
_DIRECT_STREAM_SUBPHASE = "direct_stream"

# Live SSE polling of the per-session endpoints.
_SESSION_STREAM_POLL_SECONDS = 0.1
_SESSION_STREAM_HEARTBEAT_SECONDS = 15

_TERMINAL_SESSION_STATUSES = {
    GenerationStatus.COMPLETED,
    GenerationStatus.REJECTED,
    GenerationStatus.FAILED,
    GenerationStatus.CANCELLED,
}


def _direct_stream_max_per_session() -> int:
    """Return the per-session direct-stream cap (env DIRECT_STREAM_MAX_PER_SESSION)."""
    raw = os.getenv("DIRECT_STREAM_MAX_PER_SESSION", "")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return MAX_DIRECT_STREAMS_PER_SESSION


def _claim_direct_stream(session: Dict[str, Any]) -> Dict[str, Any]:
    """Validate direct-stream eligibility and mark the session atomically.

    Runs inside the session lock via mutate_session, so the state checks and
    the "active" marker cannot race between concurrent requests (API-04).
    """
    status_value = session.get("status")
    if status_value == GenerationStatus.CANCELLED:
        raise HTTPException(status_code=409, detail="Session is stopped")
    if status_value in _TERMINAL_SESSION_STATUSES:
        raise HTTPException(status_code=409, detail="Session already finished")
    if session.get("direct_stream_active"):
        raise HTTPException(status_code=409, detail="Direct stream already active for this session")
    count = int(session.get("direct_stream_count", 0) or 0)
    if count >= _direct_stream_max_per_session():
        raise HTTPException(status_code=429, detail="Direct stream limit reached for this session")
    session["direct_stream_count"] = count + 1
    session["direct_stream_active"] = True
    return {
        "request": session.get("request"),
        "resolved_context": session.get("resolved_context") or [],
        "status": session.get("status"),
        "project_id": session.get("project_id"),
        "project_epoch": session.get("project_epoch"),
        "request_name": session.get("request_name"),
        "cancelled": session.get("cancelled", False),
        "hard_cancelled": session.get("hard_cancelled", False),
        "usage_tracker": session.get("usage_tracker"),
        "direct_stream_count": count + 1,
    }


async def _release_direct_stream(session_id: str) -> None:
    """Clear the active direct-stream marker for a session (API-04)."""
    await mutate_session(session_id, lambda session: session.update({"direct_stream_active": False}))


def _client_stream_timeout_ceiling() -> float:
    """Ceiling applied to client-supplied stream timeouts (API-09)."""
    raw = os.getenv("STREAM_TIMEOUT_CLIENT_MAX_SECONDS", "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value
    return float(getattr(config, "REQUEST_TIMEOUT", 12000) or 12000)


def _resolve_direct_stream_timeout(request_data: Any) -> float:
    """Resolve the direct-stream timeout, clamping client-supplied values (API-09).

    Client timeout fields keep their precedence (see
    request_timeouts.resolve_request_timeout) but are capped at
    STREAM_TIMEOUT_CLIENT_MAX_SECONDS; operator config keeps its value.
    """
    resolved = resolve_request_timeout(
        request_data,
        "stream_seconds",
        settings=getattr(config, "REQUEST_TIMEOUTS", {}) or {},
        config_path=("streaming", "stream_seconds"),
        fallback=float(getattr(config, "REQUEST_TIMEOUT", 12000) or 12000),
    )
    client_supplied = get_request_timeout_field(request_data, "stream_seconds")
    if client_supplied is None:
        client_supplied = get_request_default_timeout(request_data)
    if client_supplied is None:
        return resolved
    return min(resolved, _client_stream_timeout_ceiling())


@app.get("/stream-content-direct/{session_id}")
async def stream_content_direct_v2(session_id: str):
    '''Direct provider content using a streaming transport, bypassing complex QA.

    Restricted to active sessions: terminal sessions are rejected and each
    session gets a bounded number of paid direct-stream runs (API-04).
    Every provider chunk is delivered as it arrives (C1) so the caller can stop
    a derailed generation before paying for the whole output. The same chunks
    go through the central content handler, which labels them
    ``content_status="draft"`` on the project stream and persists the verbatim
    trace in the debugger. If the provider stops the response by content
    policy, emission stops and a terminal marker is appended: what was already
    delivered is not "unsent", which is accepted by design - those tokens had
    already been handed to the engine and the caller owns the session.
    '''
    snapshot = await mutate_session(session_id, _claim_direct_stream)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        request_data = snapshot.get("request")
        resolved_context = snapshot.get("resolved_context") or []
        project_id = snapshot.get("project_id")
        project_epoch = snapshot.get("project_epoch")

        if (
            snapshot.get("cancelled")
            or snapshot.get("hard_cancelled")
            or await cancellation_registry.is_hard_cancelled(session_id)
        ):
            raise HTTPException(status_code=409, detail="Session is stopped")
        if project_id and await cancellation_registry.is_project_cancelled_or_stopping(project_id):
            raise HTTPException(status_code=409, detail="Project is stopped")

        context_prompt = ""
        if resolved_context:
            try:
                context_prompt = build_context_prompt(get_attachment_manager(), resolved_context)
            except AttachmentError as exc:
                logger.warning("Failed to compose context prompt for streaming: %s", exc)
                context_prompt = ""

        if not request_data:
            raise HTTPException(status_code=400, detail="No request data found")
    except Exception:
        # Release the direct-stream claim when validation fails after claiming.
        await _release_direct_stream(session_id)
        raise

    logger.info(
        "Direct stream claimed for session %s (run %d/%d)",
        session_id,
        snapshot["direct_stream_count"],
        _direct_stream_max_per_session(),
    )

    async def direct_content_generator():
        console_context_tokens = bind_console_context(
            session_id=session_id,
            project_id=project_id,
            phase="direct_stream",
        )
        current_task = asyncio.current_task()
        registered_task = False
        content_filter_stopped = False
        content_template = ContentEventTemplate(
            phase=ContentPhase.GENERATION,
            session_id=session_id,
            project_id=project_id,
            project_epoch=project_epoch,
            request_name=snapshot.get("request_name"),
            attempt=1,
            subphase=_DIRECT_STREAM_SUBPHASE,
        )

        async def _publish_filtered_terminal() -> None:
            """Close the trace with the filtered verdict before the marker."""
            await publish_content_event(
                ContentEvent(
                    phase=ContentPhase.GENERATION,
                    subphase=_DIRECT_STREAM_SUBPHASE,
                    session_id=session_id,
                    project_id=project_id,
                    project_epoch=project_epoch,
                    request_name=snapshot.get("request_name"),
                    attempt=1,
                    content_status=ContentStatus.FILTERED,
                    reason="Provider content policy stopped generation.",
                )
            )

        try:
            await cancellation_registry.register_current_task(
                session_id,
                "direct_content_stream",
                project_id=project_id,
                project_epoch=project_epoch,
            )
            registered_task = current_task is not None
            cancellation_token = CancellationToken(
                session_id=session_id,
                project_id=project_id,
                phase="streaming",
                operation="direct_content_stream",
                registry=cancellation_registry,
            )
            _ensure_services()
            usage_tracker = snapshot.get("usage_tracker")
            prompt = request_data.prompt
            word_instructions = build_word_count_instructions(request_data) if (
                hasattr(request_data, 'max_words') and request_data.max_words
            ) else ""

            prompt_sections: List[str] = []
            if context_prompt:
                prompt_sections.append(context_prompt)
            prompt_sections.append(prompt)
            if word_instructions:
                prompt_sections.append(word_instructions)
            final_prompt = "\n\n".join(section.strip() for section in prompt_sections if section)
            route = resolve_call("generation.direct_stream", request=request_data)
            route_params = route.params
            token_resolution = config.resolve_output_max_tokens(
                route.model,
                requested_max_tokens=getattr(request_data, 'max_tokens', None),
                routed_max_tokens=route_params.get("max_tokens"),
                call_id="generation.direct_stream",
            )
            requested_max_tokens = token_resolution["max_tokens"]

            async def direct_stream_cancelled() -> bool:
                return await cancellation_token.any_cancelled()

            stream_timeout = _resolve_direct_stream_timeout(request_data)

            async for chunk in ai_service.generate_content_stream(
                prompt=final_prompt,
                model=route.model,
                temperature=route_params.get("temperature", request_data.temperature),
                max_tokens=requested_max_tokens,
                reasoning_effort=route_params.get("reasoning_effort", getattr(request_data, 'reasoning_effort', None)),
                thinking_budget_tokens=route_params.get("thinking_budget_tokens", getattr(request_data, 'thinking_budget_tokens', None)),
                content_type=getattr(request_data, 'content_type', 'biography'),
                llm_routing=getattr(request_data, "_llm_routing", None),
                request_timeout=stream_timeout,
                usage_callback=usage_tracker.create_callback(
                    phase="direct_stream",
                    role="generator",
                    operation="direct_content_stream",
                ) if usage_tracker else None,
                cancellation_token=cancellation_token,
                cancel_callback=direct_stream_cancelled,
            ):
                if await direct_stream_cancelled():
                    raise asyncio.CancelledError()
                # Keep thinking and content together on this raw endpoint; the
                # labelled separation lives on /stream/project (C8).
                is_thinking = False
                if isinstance(chunk, StreamChunk):
                    finish_category = str(
                        (chunk.metadata or {}).get("finish_reason_category") or ""
                    ).strip().casefold()
                    if finish_category == "content_filter":
                        content_filter_stopped = True
                        continue
                    chunk_text = chunk.text
                    is_thinking = bool(chunk.is_thinking)
                else:
                    chunk_text = chunk
                if chunk_text and not content_filter_stopped:
                    await publish_content_event(
                        content_template.chunk(
                            chunk_text,
                            is_thinking=is_thinking,
                            subphase=(
                                ContentSubphase.THINKING
                                if is_thinking
                                else _DIRECT_STREAM_SUBPHASE
                            ),
                        )
                    )
                    yield chunk_text.encode("utf-8")

            if content_filter_stopped:
                await _publish_filtered_terminal()
                yield _DIRECT_STREAM_CONTENT_FILTER_TERMINAL

        except asyncio.CancelledError:
            logger.info("Direct content stream cancelled for session %s", session_id)
        except Exception as exc:
            if content_filter_stopped or is_content_filter_stop(exc):
                logger.warning(
                    "Direct streaming stopped by provider content policy for session %s",
                    session_id,
                )
                await _publish_filtered_terminal()
                yield _DIRECT_STREAM_CONTENT_FILTER_TERMINAL
                return
            error_id = uuid4().hex[:8]
            logger.exception(
                "Direct streaming failed for session %s [error_id=%s]",
                session_id,
                error_id,
            )
            yield f"Error: direct streaming failed (error_id={error_id})".encode('utf-8')
        finally:
            await flush_content_events(session_id, phase=ContentPhase.GENERATION)
            await _release_direct_stream(session_id)
            if registered_task and current_task is not None:
                await cancellation_registry.unregister_task(session_id, current_task)
            reset_console_context(console_context_tokens)

    return StreamingResponse(
        direct_content_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


_UNSET = object()


async def _stream_live_session_field(
    session_id: str,
    *,
    field_name: str,
    session_phases: Optional[FrozenSet[str]] = None,
    final_field: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Serve one live session text field as SSE, cursor-safe across attempts.

    Live fields mean "current attempt" (C2): a new attempt clears them and
    refills them, so a plain length cursor goes silent the moment a shorter
    attempt replaces a longer one. The cursor is reset here whenever the
    attempt id changes or the field stops continuing what was already emitted,
    and the first chunk emitted after a reset carries ``"reset": true`` so a
    consumer that concatenates knows it must drop what it had instead of
    appending to it.

    Length alone cannot decide continuity: a normalized or guard-repaired body
    can be LONGER than the draft it replaces without sharing its prefix (JSON
    re-spacing, for instance), and slicing it by the cursor would splice new
    text onto a stale prefix. The text already emitted is therefore kept and
    prefix-compared on every poll; the fields are bounded per session, so
    holding it costs nothing.

    When the replacement is EMPTY the reset travels on its own, as a chunk with
    ``"content": ""`` and ``"reset": true``: a failed attempt that clears the
    field leaves no text to carry the flag, and staying silent would leave the
    consumer showing a draft the engine already discarded. It is only sent when
    something HAD been emitted, so a stream that never delivered anything never
    asks a consumer to drop what it does not have.

    Args:
        session_id: session that owns the field.
        field_name: live session field to follow.
        session_phases: session phases with a live producer for this field;
            ``None`` follows the field during the whole session.
        final_field: field flushed once when the session reaches a terminal
            status (only the accepted content, never a draft).
    """
    attempt_key = live_attempt_key(field_name)
    # Verbatim text already handed to the client: the continuity reference.
    emitted = ""
    last_attempt: Any = _UNSET
    pending_reset = False
    last_heartbeat = time.time()

    def _read(session: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": session.get("status"),
            "current_phase": session.get("current_phase", SessionPhase.INITIALIZING),
            "value": session.get(field_name) or "",
            "attempt_id": session.get(attempt_key),
            "final_value": (session.get(final_field) or "") if final_field else "",
        }

    def _encode(chunk: str, *, reset: bool) -> bytes:
        payload: Dict[str, Any] = {"type": "chunk", "content": chunk}
        if reset:
            payload["reset"] = True
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    while True:
        snapshot = await mutate_session(session_id, _read)
        if snapshot is None:
            break

        streaming_phase = (
            session_phases is None or snapshot["current_phase"] in session_phases
        )
        if streaming_phase:
            attempt_id = snapshot["attempt_id"]
            if last_attempt is _UNSET:
                last_attempt = attempt_id
            elif attempt_id != last_attempt:
                last_attempt = attempt_id
                pending_reset = pending_reset or bool(emitted)
                emitted = ""

            current = snapshot["value"]
            if not current.startswith(emitted):
                # The field stopped continuing what the client already has (a
                # shorter attempt, or a normalized/repaired body of any length):
                # replace it instead of splicing new text onto a stale prefix.
                pending_reset = True
                emitted = ""
            if len(current) > len(emitted):
                yield _encode(current[len(emitted):], reset=pending_reset)
                pending_reset = False
                emitted = current
            elif pending_reset:
                # The replacement is EMPTY (a failed attempt cleared the field,
                # or the next attempt has not produced a token yet). There is no
                # chunk to carry the flag on, so the reset travels alone: the
                # client must drop dead text now, not keep it until something
                # new arrives - which may never happen.
                yield _encode("", reset=True)
                pending_reset = False

        if snapshot["status"] in _TERMINAL_SESSION_STATUSES:
            final_value = snapshot["final_value"]
            if final_value:
                if not final_value.startswith(emitted):
                    # The accepted body does not continue what was streamed
                    # live: replay it whole instead of going silent or splicing.
                    pending_reset = True
                    emitted = ""
                if len(final_value) > len(emitted):
                    yield _encode(final_value[len(emitted):], reset=pending_reset)
                    pending_reset = False
                    emitted = final_value
            elif emitted and not snapshot["value"]:
                # Nothing accepted AND the live field is gone: the text already
                # delivered belongs to an attempt the engine discarded, and no
                # further poll will ever correct it. A content-filter stop keeps
                # its draft in the live field (C16), so it does not land here.
                pending_reset = True
                emitted = ""
            if pending_reset:
                yield _encode("", reset=True)
            break

        now = time.time()
        if now - last_heartbeat >= _SESSION_STREAM_HEARTBEAT_SECONDS:
            yield b": heartbeat\n\n"
            last_heartbeat = now

        await asyncio.sleep(_SESSION_STREAM_POLL_SECONDS)


def _live_field_stream_response(
    session_id: str,
    *,
    field_name: str,
    session_phases: Optional[FrozenSet[str]] = None,
    final_field: Optional[str] = None,
) -> StreamingResponse:
    """Wrap a live session field generator in an SSE streaming response."""
    return StreamingResponse(
        _stream_live_session_field(
            session_id,
            field_name=field_name,
            session_phases=session_phases,
            final_field=final_field,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def _require_session(session_id: str) -> None:
    """Reject the request when the session does not exist."""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/stream-content/{session_id}")
async def stream_content_direct(session_id: str):
    """Stream the live generation text of a session in SSE format.

    Emits the draft of the attempt in progress as it is produced and the
    accepted content once the session reaches a terminal status. Provider
    reasoning is NOT mixed in here: thinking travels labelled on
    ``/stream/project`` (subphase ``thinking``).
    """
    await _require_session(session_id)
    return _live_field_stream_response(
        session_id,
        field_name="partial_content",
        final_field="last_generated_content",
    )


@app.get("/stream-generation/{session_id}")
async def stream_generation_content(session_id: str):
    """Stream the live generation content of a session in SSE format."""
    await _require_session(session_id)
    return _live_field_stream_response(
        session_id,
        field_name="generation_content",
        session_phases=GENERATION_SESSION_PHASES,
    )


@app.get("/stream-qa/{session_id}")
async def stream_qa_content(session_id: str):
    """Stream live QA evaluation content of a session in SSE format."""
    await _require_session(session_id)
    return _live_field_stream_response(
        session_id,
        field_name="qa_content",
        session_phases=QA_SESSION_PHASES,
    )


@app.get("/stream-preflight/{session_id}")
async def stream_preflight_content(session_id: str):
    """Stream live preflight validation content of a session in SSE format."""
    await _require_session(session_id)
    return _live_field_stream_response(
        session_id,
        field_name="preflight_content",
    )


@app.get("/stream-gransabio/{session_id}")
async def stream_gran_sabio_content(session_id: str):
    """Stream live Gran Sabio processing content of a session in SSE format."""
    await _require_session(session_id)
    return _live_field_stream_response(
        session_id,
        field_name="gransabio_content",
        session_phases=GRAN_SABIO_SESSION_PHASES,
    )


# --- Project-scoped unified stream ---

@app.get("/stream/project/{project_id}")
async def stream_project_unified(
    project_id: str,
    phases: str = Query(
        default="all",
        description="CSV of phases: auto_qa, preflight, generation, qa, consensus, gransabio/gran_sabio, status, or 'all'"
    ),
    visibility: str = Query(
        default="full",
        description="full (default, everything labelled) or canonical (no drafts, no thinking)"
    )
):
    """
    Unified real-time project event stream (SSE).

    Allows selecting which phases to monitor via the `phases` parameter.
    Receives all phases by default.

    **Available phases:**
    - `auto_qa` - Auto-QA planning chunks
    - `preflight` - Preflight validation chunks
    - `generation` - Content generation chunks
    - `qa` - QA evaluation chunks
    - `consensus` - Consensus updates
    - `gransabio` or `gran_sabio` - Gran Sabio chunks
    - `status` - Aggregated status (includes initial snapshot)
    - `all` - All of the above (default)

    **Visibility (convenience filter, not a permission):**
    - `full` (default) - everything the engine produces, each emission labelled
      with `content_status` (draft/accepted/rejected/filtered), `iteration` and
      `attempt_id`
    - `canonical` - omits `content_status="draft"` chunks and thinking, keeps
      status, control, accepted and terminal events; the initial late-join
      snapshot follows the same rule

    **Behavior:**
    - All events include "phase" field to identify origin
    - Content events include a `sequence` monotonic per (session, phase);
      delivery is best effort, so a gap means events were dropped
    - If "status" is included, sends initial snapshot after connection
    - Heartbeat every 15 seconds keeps connection alive
    - Immediate close on cancel (no queue draining)

    **Examples:**
    - `/stream/project/abc123` -> all phases
    - `/stream/project/abc123?phases=generation,qa` -> only those phases
    - `/stream/project/abc123?phases=status` -> status only
    - `/stream/project/abc123?visibility=canonical` -> committed content only

    **Errors:**
    - 400: Invalid project_id, invalid phases or invalid visibility
    - an `error` event on the stream when no phase could be subscribed: the
      SSE response has already started by then, so it is not an HTTP status
    """
    # Validate project_id
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")

    # Parse and validate phases
    try:
        parsed_phases = parse_phases(phases)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        parsed_visibility = parse_visibility(visibility)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _unified_stream_response(
        ProjectStreamManager(project_id, parsed_phases, parsed_visibility)
    )


def _unified_stream_response(manager: ProjectStreamManager) -> StreamingResponse:
    """Wrap a stream manager in an SSE response that reports subscription errors."""

    async def safe_stream():
        try:
            async for chunk in manager.stream():
                yield chunk
        except SubscriptionError as e:
            # This shouldn't happen often since we check in stream(),
            # but handle gracefully if it does
            logger.error("Subscription error during stream: %s", e)
            error_event = {
                "type": "error",
                "phase": "status",
                "error": str(e),
                "timestamp": int(time.time() * 1000),
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=True)}\n\n".encode("utf-8")

    return StreamingResponse(
        safe_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# --- Session-scoped unified stream ---
#
# Registered AFTER /stream/console (imported explicitly at the top of this
# module): both paths have the same shape and FastAPI matches in registration
# order, so the literal console route must always win.

@app.get("/stream/{session_id}")
async def stream_session_unified(
    session_id: str,
    phases: str = Query(
        default="all",
        description="CSV of phases: auto_qa, preflight, generation, qa, consensus, gransabio/gran_sabio, status, or 'all'"
    ),
    visibility: str = Query(
        default="full",
        description="full (default, everything labelled) or canonical (no drafts, no thinking)"
    )
):
    """
    Real-time event stream of ONE session (SSE).

    Session-scoped adapter over the project bus: it subscribes to the same
    phases as `/stream/project/{project_id}` and delivers only what belongs to
    this session, on a flat contract.

    **Events:**
    - `connected` - carries `session_id`, `project_id`, subscribed phases and
      the effective `visibility`
    - `status_snapshot` (on connect) and `status_change` - `session` object plus
      the flat `status`, `current_phase`, `iteration`, `max_iterations`; the
      `project` block only carries project identity/lifecycle
    - content events of the session, labelled with `content_status`,
      `iteration`, `attempt_id` and a `sequence` monotonic per (session, phase)
    - `session_end` for this session, followed by the closing `stream_end`

    **Isolation:** sibling sessions of the same project are never exposed, not
    even through project aggregates.

    **Close:** the stream drains what this session had already produced and
    closes after its own `session_end`; it also closes if the project ends
    first. `session_end` travels on the content phases AND on the status
    channel, so every subscription - `phases=status` included - closes on its
    own session terminal, exactly once.

    **Visibility (convenience filter, not a permission):** same as the project
    stream - `full` (default) or `canonical` (no drafts, no thinking, snapshot
    included).

    **Errors:**
    - 400: invalid session_id, phases or visibility
    - 404: unknown session
    - 409: session without a project stream
    - an `error` event on the stream when no phase could be subscribed: the
      SSE response has already started by then, so it is not an HTTP status
    """
    if not session_id or len(session_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        parsed_phases = parse_phases(phases)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        parsed_visibility = parse_visibility(visibility)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    project_id = await mutate_session(
        session_id,
        lambda session: session.get("project_id") or "",
    )
    if project_id is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not project_id:
        # /generate assigns project_id = session_id when the caller does not
        # provide one, so a session without a project is an engine invariant
        # violation, not a client-recoverable state.
        raise HTTPException(status_code=409, detail="Session has no project stream")

    return _unified_stream_response(
        SessionStreamManager(project_id, session_id, parsed_phases, parsed_visibility)
    )
