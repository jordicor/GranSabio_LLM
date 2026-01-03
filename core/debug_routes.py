"""
Debugger API routes for Gran Sabio LLM.

Note: These endpoints are protected by require_internal_ip dependency,
ensuring they remain internal-only even after the global IP middleware
is replaced with user authentication.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Query

from debug_logger import DebugLogger, get_debug_logger

from . import app_state
from .app_state import app, config, _parse_debug_json
from .security import require_internal_ip


async def _get_active_debug_logger() -> DebugLogger:
    """Return an initialized debug logger instance."""
    if app_state.debug_logger is None:
        app_state.debug_logger = await get_debug_logger(
            enabled=config.DEBUGGER.enabled,
            db_path=config.DEBUGGER.db_path,
        )
        if config.DEBUGGER.enabled:
            await app_state.debug_logger.initialize()
    return app_state.debug_logger


@app.get("/debugger/sessions")
async def list_debugger_sessions(
    limit: int = Query(25, ge=1),
    offset: int = Query(0, ge=0),
    project_id: Optional[str] = Query(default=None),
    _client_ip: str = Depends(require_internal_ip),
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
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
    # Extract request_name from request_json for each session
    enriched_sessions = []
    for session in sessions:
        request_data = _parse_debug_json(session.get("request_json"))
        request_name = None
        if isinstance(request_data, dict):
            request_name = request_data.get("request_name")
        enriched_sessions.append({
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "status": session["status"],
            "project_id": session["project_id"],
            "request_name": request_name,
        })
    return {"sessions": enriched_sessions, "project_id": normalized_project_id}


@app.get("/debugger/sessions/{session_id}")
async def debugger_session_details(
    session_id: str,
    _client_ip: str = Depends(require_internal_ip),
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
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
):
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")

    logger_instance = await _get_active_debug_logger()
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

