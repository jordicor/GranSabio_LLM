"""
Debugger API routes for Gran Sabio LLM.

Note: These endpoints are protected by require_internal_ip dependency,
ensuring they remain internal-only even after the global IP middleware
is replaced with user authentication.
"""

import os
from typing import Optional

from fastapi import Depends, HTTPException, Query

from debug_logger import DebugLogger, get_debug_logger

from . import app_state
from .app_state import _parse_debug_json, app, config
from .security import require_internal_ip


def require_debugger_exposed() -> None:
    """FastAPI dependency: 404 when the debugger API is disabled (API-11).

    The debugger endpoints return full request payloads (prompts, business
    data) of every session; the IP filter gives no isolation between users
    of the same deployment. Multi-user setups can set
    DEBUGGER_EXPOSE_ENABLED=false to take the debugger API off the wire
    without disabling capture. Default is enabled so the local UI keeps
    working.
    """
    if os.getenv("DEBUGGER_EXPOSE_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=404, detail="Debugger endpoints are disabled")


async def _get_active_debug_logger() -> DebugLogger:
    """Return an initialized debug logger instance."""
    if app_state.debug_logger is None:
        app_state.debug_logger = await get_debug_logger(
            enabled=config.DEBUGGER.enabled,
            db_path=config.DEBUGGER.db_path,
            retention_days=config.DEBUGGER.retention_days,
        )
        if config.DEBUGGER.enabled:
            await app_state.debug_logger.initialize()
    return app_state.debug_logger


def _ensure_debugger_storage_available(logger_instance: DebugLogger) -> None:
    """Fail fast when debugger storage has been disabled after a terminal I/O error."""
    if logger_instance.storage_disabled_reason:
        raise HTTPException(
            status_code=503,
            detail=f"Debugger storage unavailable: {logger_instance.storage_disabled_reason}",
        )


@app.get("/debugger/sessions")
async def list_debugger_sessions(
    limit: int = Query(25, ge=1),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = Query(default=None),
    _client_ip: str = Depends(require_internal_ip),
    _exposed: None = Depends(require_debugger_exposed),
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
    _ensure_debugger_storage_available(logger_instance)
    max_limit = max(1, config.DEBUGGER.max_session_list)
    limit = min(limit, max_limit)
    normalized_project_id = project_id.strip() if project_id else None
    if normalized_project_id and len(normalized_project_id) > 128:
        raise HTTPException(status_code=400, detail="project_id must be 128 characters or fewer")
    sessions = await logger_instance.list_sessions(
        limit=limit,
        offset=offset,
        project_id=normalized_project_id,
    )
    # request_name is now denormalized in the sessions table
    enriched_sessions = []
    for session in sessions:
        enriched_sessions.append({
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "status": session["status"],
            "phase": session.get("phase"),
            "project_id": session["project_id"],
            "request_name": session.get("request_name"),
        })
    return {"sessions": enriched_sessions, "project_id": normalized_project_id}


@app.get("/debugger/sessions/{session_id}")
async def debugger_session_details(
    session_id: str,
    _client_ip: str = Depends(require_internal_ip),
    _exposed: None = Depends(require_debugger_exposed),
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
    _ensure_debugger_storage_available(logger_instance)
    details = await logger_instance.get_session_details(session_id)
    if not details:
        raise HTTPException(status_code=404, detail="Session not found")

    request_data = _parse_debug_json(details.get("request_json"))
    request_name = None
    if isinstance(request_data, dict):
        request_name = request_data.get("request_name")

    response_payload = {
        "session_id": details["session_id"],
        "created_at": details["created_at"],
        "updated_at": details["updated_at"],
        "status": details["status"],
        "phase": details.get("phase"),
        "project_id": details.get("project_id"),
        "request_name": request_name,
        "request": request_data,
        "preflight": _parse_debug_json(details.get("preflight_json")),
        "final": _parse_debug_json(details.get("final_json")),
        "usage": _parse_debug_json(details.get("usage_json")),
        "notes": details.get("notes"),
    }
    return response_payload


@app.get("/debugger/sessions/{session_id}/events")
async def debugger_session_events(
    session_id: str,
    limit: int = Query(200, ge=1),
    offset: int = Query(0, ge=0),
    _client_ip: str = Depends(require_internal_ip),
    _exposed: None = Depends(require_debugger_exposed),
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
    _ensure_debugger_storage_available(logger_instance)
    max_limit = max(1, config.DEBUGGER.max_session_list * 5)
    limit = min(limit, max_limit)
    events = await logger_instance.get_session_events(session_id, offset=offset, limit=limit)
    parsed_events = [
        {
            "event_order": item["event_order"],
            "event_type": item["event_type"],
            "created_at": item["created_at"],
            "payload": _parse_debug_json(item.get("payload_json")),
        }
        for item in events
    ]
    return {"events": parsed_events}


@app.get("/debugger/sessions/{session_id}/content-traces")
async def debugger_session_content_traces(
    session_id: str,
    limit: int = Query(25, ge=1),
    offset: int = Query(0, ge=0),
    _client_ip: str = Depends(require_internal_ip),
    _exposed: None = Depends(require_debugger_exposed),
):
    """Read the persisted content trace of a session (C0.8 / §8.2).

    The trace is what makes "look back at what happened" work when nobody was
    subscribed to the stream: one row per (session, phase, attempt, model,
    layer) with the verbatim text and its labels. Rows carry whole generation
    bodies, so pages are small by default.
    """
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
    _ensure_debugger_storage_available(logger_instance)
    max_limit = max(1, config.DEBUGGER.max_session_list)
    limit = min(limit, max_limit)
    traces = await logger_instance.get_content_traces(
        session_id,
        limit=limit,
        offset=offset,
    )
    return {"traces": traces}
