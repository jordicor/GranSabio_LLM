"""
HTML interface routes for Gran Sabio LLM.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from .app_state import app, config, templates
from .security import require_internal_ip


@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/monitor", response_class=HTMLResponse)
async def monitor_interface(request: Request):
    """Serve the live stream monitor interface"""
    return templates.TemplateResponse("stream_monitor.html", {"request": request})


@app.get("/debugger", response_class=HTMLResponse)
async def debugger_interface(
    request: Request,
    project_id: Optional[str] = Query(default=None),
    _client_ip: str = Depends(require_internal_ip),
):
    """Serve the debugger interface when enabled."""
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")
    normalized_project_id = project_id.strip() if project_id else None
    if normalized_project_id and len(normalized_project_id) > 128:
        raise HTTPException(status_code=400, detail="project_id must be 128 characters or fewer")
    return templates.TemplateResponse(
        "debugger.html",
        {
            "request": request,
            "project_id": normalized_project_id,
        },
    )


@app.get("/debugger/project/{project_id}", response_class=HTMLResponse)
async def debugger_project_interface(
    request: Request,
    project_id: str,
    _client_ip: str = Depends(require_internal_ip),
):
    """Serve the debugger filtered to a specific project identifier."""
    if not config.DEBUGGER.enabled:
        raise HTTPException(status_code=404, detail="Debugger interface is disabled")
    normalized_project_id = project_id.strip() or None
    if normalized_project_id and len(normalized_project_id) > 128:
        raise HTTPException(status_code=400, detail="project_id must be 128 characters or fewer")
    return templates.TemplateResponse(
        "debugger.html",
        {
            "request": request,
            "project_id": normalized_project_id,
        },
    )
