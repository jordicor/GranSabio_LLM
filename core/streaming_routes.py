"""
Streaming endpoints for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Dict, FrozenSet, Optional

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

import json_utils as json
from content_events import (
    GENERATION_SESSION_PHASES,
    GRAN_SABIO_SESSION_PHASES,
    QA_SESSION_PHASES,
    SessionPhase,
    live_attempt_key,
    parse_visibility,
)
from models import GenerationStatus
from services.project_stream import (
    ProjectStreamManager,
    SessionStreamManager,
    SubscriptionError,
    parse_phases,
)

# Route order is a contract, not an accident: /stream/console and
# /stream/{session_id} have the same shape, and FastAPI resolves them in
# registration order. Importing the console routes HERE guarantees the literal
# path is registered before the catch-all below, whatever order the package
# imports its route modules in.
from . import console_routes  # noqa: F401
from .app_state import (
    app,
    logger,
    mutate_session,
)

# Live SSE polling of the per-session endpoints.
_SESSION_STREAM_POLL_SECONDS = 0.1
_SESSION_STREAM_HEARTBEAT_SECONDS = 15

_TERMINAL_SESSION_STATUSES = {
    GenerationStatus.COMPLETED,
    GenerationStatus.REJECTED,
    GenerationStatus.FAILED,
    GenerationStatus.CANCELLED,
}


@app.get("/stream-content-direct/{session_id}", status_code=410, deprecated=True)
async def stream_content_direct_retired(session_id: str) -> None:
    """Reject the retired endpoint that started a separate generation."""
    raise HTTPException(
        status_code=410,
        detail=(
            "This endpoint is retired because it started a separate generation "
            "outside the normal input and QA pipeline. "
            f"Use GET /stream-content/{session_id} to observe the existing generation "
            f"or GET /stream/{session_id} for session events. "
            "Start a new generation with POST /generate."
        ),
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
async def stream_content(session_id: str) -> StreamingResponse:
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
