"""Stateless context preflight route (``gransabio_context_preflight.v2``).

``POST /context/preflight`` measures a candidate generation request without
creating sessions, projects, provider_sessions, or any LLM/provider call.
It answers either an accepted budget report (per-participant arithmetic plus
the residual evidence budget) or a structured rejection, always with HTTP 200;
only structurally invalid payloads are HTTP 400 (a caller bug, not a budget
outcome). See ``core/context_preflight.py`` for the contract rules.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import HTTPException, Request

from . import context_preflight
from .app_state import app


@app.post("/context/preflight")
async def context_preflight_endpoint(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    try:
        return context_preflight.build_context_preflight_response(payload)
    except context_preflight.MalformedRequest as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
