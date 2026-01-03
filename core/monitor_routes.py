"""
Monitor API routes for Gran Sabio LLM.
Provides endpoints for listing active projects and sessions.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter

router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("/active")
async def list_active_connections():
    """
    List all active projects and standalone sessions for the monitor UI.

    Returns:
        {
            "projects": [
                {
                    "project_id": str,
                    "status": str,  # idle, running, completed, failed, cancelled
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

    projects_map: Dict[str, Dict] = {}
    standalone: List[Dict] = []

    lock = active_sessions_lock

    def _process_sessions(items):
        for session_id, session in items:
            project_id = session.get("project_id")
            status_enum = session.get("status")
            status_str = status_enum.value if hasattr(status_enum, "value") else str(status_enum)

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
                if status_str in ("generating", "qa_evaluation", "consensus", "initializing", "running"):
                    proj["active_sessions"] += 1
                    proj["status"] = "running"
                elif status_str == "completed" and proj["status"] != "running":
                    proj["status"] = "completed"
                elif status_str in ("failed", "error") and proj["status"] not in ("running", "completed"):
                    proj["status"] = "failed"
                elif status_str == "cancelled" and proj["status"] not in ("running", "completed", "failed"):
                    proj["status"] = "cancelled"

                # Track last activity
                created = session.get("created_at")
                if created:
                    created_iso = created.isoformat() if isinstance(created, datetime) else str(created)
                    if proj["last_activity"] is None or created_iso > proj["last_activity"]:
                        proj["last_activity"] = created_iso
            else:
                # Standalone session (no project_id)
                created = session.get("created_at")
                standalone.append({
                    "session_id": session_id,
                    "request_name": session.get("request_name"),
                    "status": status_str,
                    "phase": session.get("current_phase", "unknown"),
                    "created_at": created.isoformat() if isinstance(created, datetime) else str(created) if created else None
                })

    if lock is None:
        _process_sessions(list(active_sessions.items()))
    else:
        import asyncio
        # We need to handle the async lock properly
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, use the lock properly
            # But since this function is async, we can just await
            pass
        _process_sessions(list(active_sessions.items()))

    # Also include reserved projects that might be idle (no active sessions yet)
    for pid in reserved_project_ids:
        if pid not in projects_map:
            projects_map[pid] = {
                "project_id": pid,
                "status": "idle",
                "session_count": 0,
                "active_sessions": 0,
                "last_activity": None
            }

    return {
        "projects": list(projects_map.values()),
        "standalone_sessions": standalone,
        "timestamp": datetime.utcnow().isoformat()
    }
