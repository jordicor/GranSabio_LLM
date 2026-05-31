"""Runtime console API routes.

These endpoints expose the raw process output captured by
``services.runtime_console`` for operator and agent observability.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

import json_utils as json
from services.runtime_console import (
    HEARTBEAT_INTERVAL_SECONDS,
    console_stats,
    get_recent_console_events,
    subscribe_console,
    unsubscribe_console,
)

from .app_state import app
from .security import require_internal_ip


def _validate_filter_value(label: str, value: Optional[str], *, max_length: int = 128) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise HTTPException(status_code=400, detail=f"{label} must be {max_length} characters or fewer")
    return normalized


def _validate_stream_filter(value: Optional[str]) -> Optional[str]:
    normalized = _validate_filter_value("stream", value, max_length=16)
    if normalized and normalized not in {"stdout", "stderr"}:
        raise HTTPException(status_code=400, detail="stream must be 'stdout' or 'stderr'")
    return normalized


def _console_filters(
    *,
    project_id: Optional[str],
    session_id: Optional[str],
    phase: Optional[str],
    stream: Optional[str],
    level: Optional[str],
) -> dict[str, Optional[str]]:
    return {
        "project_id": _validate_filter_value("project_id", project_id),
        "session_id": _validate_filter_value("session_id", session_id),
        "phase": _validate_filter_value("phase", phase, max_length=64),
        "stream": _validate_stream_filter(stream),
        "level": _validate_filter_value("level", level, max_length=32),
    }


def _format_sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


@app.get("/monitor/console/recent")
async def recent_console_output(
    limit: int = Query(200, ge=1, le=1000),
    project_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    phase: Optional[str] = Query(default=None),
    stream: Optional[str] = Query(default=None),
    level: Optional[str] = Query(default=None),
    _client_ip: str = Depends(require_internal_ip),
):
    """Return recent captured stdout/stderr/logging output."""

    filters = _console_filters(
        project_id=project_id,
        session_id=session_id,
        phase=phase,
        stream=stream,
        level=level,
    )
    return {
        "events": get_recent_console_events(limit=limit, **filters),
        "filters": filters,
        "stats": console_stats(),
    }


@app.get("/stream/console")
async def stream_console_output(
    tail: int = Query(200, ge=0, le=1000),
    project_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    phase: Optional[str] = Query(default=None),
    stream: Optional[str] = Query(default=None),
    level: Optional[str] = Query(default=None),
    _client_ip: str = Depends(require_internal_ip),
):
    """Stream captured runtime console output as Server-Sent Events."""

    filters = _console_filters(
        project_id=project_id,
        session_id=session_id,
        phase=phase,
        stream=stream,
        level=level,
    )

    async def event_generator():
        subscription = await subscribe_console(**filters)
        sent_tail_sequences: set[int] = set()
        try:
            connected_event = {
                "type": "console_connected",
                "timestamp": int(time.time() * 1000),
                "filters": filters,
                "tail": tail,
                "stats": console_stats(),
            }
            yield _format_sse(connected_event)

            if tail:
                for event in get_recent_console_events(limit=tail, **filters):
                    seq = event.get("seq")
                    if isinstance(seq, int):
                        sent_tail_sequences.add(seq)
                    yield _format_sse(event)

            while True:
                try:
                    event = await asyncio.wait_for(
                        subscription.queue.get(),
                        timeout=HEARTBEAT_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    yield b": heartbeat\n\n"
                    continue
                seq = event.get("seq")
                if isinstance(seq, int) and seq in sent_tail_sequences:
                    continue
                yield _format_sse(event)
        finally:
            await unsubscribe_console(subscription)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
