"""
Monitor API routes for Gran Sabio LLM.
Provides endpoints for listing active projects and sessions.
"""

from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("/active")
async def list_active_connections(
    status: str = "running",
    include_reserved: bool = False,
    limit: int = 100,
):
    """
    List all active projects and standalone sessions for the monitor UI.

    Returns:
        {
            "projects": [
                {
                    "project_id": str,
                    "status": str,  # idle, running, completed, rejected, failed, cancelled
                    "session_count": int,
                    "active_sessions": int,
                    "last_activity": str (ISO timestamp)
                }
            ],
            "standalone_sessions": [
                {
                    "session_id": str,
                    "request_name": str | None,
                    "status": str,
                    "phase": str,
                    "created_at": str (ISO timestamp)
                }
            ],
            "timestamp": str (ISO timestamp)
        }
    """
    from .app_state import active_sessions, active_sessions_lock, reserved_project_ids

    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    requested_status = (status or "running").lower()
    valid_filters = {"running", "terminal", "idle", "all"}
    if requested_status not in valid_filters:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter '{status}'. Valid values: {sorted(valid_filters)}",
        )

    projects_map: Dict[str, Dict] = {}
    standalone: List[Dict] = []

    lock = active_sessions_lock

    def _process_sessions(items):
        for session_id, session in items:
            project_id = session.get("project_id")
            status_enum = session.get("status")
            status_str = status_enum.value if hasattr(status_enum, "value") else str(status_enum)
            phase = session.get("current_phase", "unknown")

            if project_id:
                # Group by project
                if project_id not in projects_map:
                    projects_map[project_id] = {
                        "project_id": project_id,
                        "status": "idle",
                        "session_count": 0,
                        "active_sessions": 0,
                        "last_activity": None
                    }

                proj = projects_map[project_id]
                proj["session_count"] += 1

                # Count active
                if (
                    status_str in ("generating", "qa_evaluation", "consensus", "gran_sabio_review", "initializing", "running")
                    or phase in ("inline_deal_breaker_review", "gran_sabio_review", "gran_sabio_regeneration")
                ):
                    proj["active_sessions"] += 1
                    proj["status"] = "running"
                elif status_str == "completed" and proj["status"] != "running":
                    proj["status"] = "completed"
                elif status_str == "rejected" and proj["status"] not in ("running", "completed"):
                    proj["status"] = "rejected"
                elif status_str in ("failed", "error") and proj["status"] not in ("running", "completed", "rejected"):
                    proj["status"] = "failed"
                elif status_str == "cancelled" and proj["status"] not in ("running", "completed", "rejected", "failed"):
                    proj["status"] = "cancelled"

                # Track last activity
                activity = session.get("last_activity_at") or session.get("created_at")
                if activity:
                    activity_iso = activity.isoformat() if isinstance(activity, datetime) else str(activity)
                    if proj["last_activity"] is None or activity_iso > proj["last_activity"]:
                        proj["last_activity"] = activity_iso
            else:
                # Standalone session (no project_id)
                created = session.get("created_at")
                standalone.append({
                    "session_id": session_id,
                    "request_name": session.get("request_name"),
                    "status": status_str,
                    "phase": phase,
                    "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None
                })

    if lock is None:
        _process_sessions(list(active_sessions.items()))
    else:
        async with lock:
            _process_sessions(list(active_sessions.items()))

    # Reserved ids can be useful for manual connection, but they are noisy for
    # the default active-only operator view.
    if include_reserved:
        for pid in reserved_project_ids:
            if pid not in projects_map:
                projects_map[pid] = {
                    "project_id": pid,
                    "status": "idle",
                    "session_count": 0,
                    "active_sessions": 0,
                    "last_activity": None
                }

    terminal_statuses = {"completed", "rejected", "failed", "error", "cancelled"}

    def _matches_filter(project: Dict) -> bool:
        project_status = str(project.get("status") or "idle")
        if requested_status == "all":
            return True
        if requested_status == "running":
            return project_status == "running" or int(project.get("active_sessions") or 0) > 0
        if requested_status == "terminal":
            return project_status in terminal_statuses
        if requested_status == "idle":
            return project_status == "idle"
        return False

    projects = [project for project in projects_map.values() if _matches_filter(project)]
    projects.sort(key=lambda project: project.get("last_activity") or "", reverse=True)
    projects.sort(key=lambda project: project.get("status") != "running")

    return {
        "projects": projects[:limit],
        "standalone_sessions": standalone,
        "filter": requested_status,
        "include_reserved": include_reserved,
        "timestamp": datetime.utcnow().isoformat()
    }
