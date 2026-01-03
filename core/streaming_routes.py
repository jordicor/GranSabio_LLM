"""
Streaming endpoints for Gran Sabio LLM.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import List

from fastapi import HTTPException, Query
from fastapi.responses import StreamingResponse

from attachments_router import get_attachment_manager
from models import GenerationStatus
from services.attachment_manager import AttachmentError
from services.project_stream import ProjectStreamManager, parse_phases, SubscriptionError

from word_count_utils import build_word_count_instructions

from .app_state import (
    _ensure_services,
    app,
    logger,
    mutate_session,
)
from ai_service import StreamChunk
from .generation_processor import build_context_prompt, ai_service


@app.get("/stream-content-direct/{session_id}")
async def stream_content_direct_v2(session_id: str):
    '''Direct streaming using ai_service - bypasses complex QA for real-time streaming'''
    snapshot = await mutate_session(
        session_id,
        lambda session: {
            "request": session.get("request"),
            "resolved_context": session.get("resolved_context") or []
        }
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Session not found")

    request_data = snapshot.get("request")
    resolved_context = snapshot.get("resolved_context") or []

    context_prompt = ""
    if resolved_context:
        try:
            context_prompt = build_context_prompt(get_attachment_manager(), resolved_context)
        except AttachmentError as exc:
            logger.warning("Failed to compose context prompt for streaming: %s", exc)
            context_prompt = ""

    if not request_data:
        raise HTTPException(status_code=400, detail="No request data found")

    async def direct_content_generator():
        try:
            _ensure_services()
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

            async for chunk in ai_service.generate_content_stream(
                prompt=final_prompt,
                model=request_data.generator_model,
                temperature=request_data.temperature,
                max_tokens=getattr(request_data, 'max_tokens', 4000),
                reasoning_effort=getattr(request_data, 'reasoning_effort', None),
                thinking_budget_tokens=getattr(request_data, 'thinking_budget_tokens', None),
                content_type=getattr(request_data, 'content_type', 'biography')
            ):
                # Handle StreamChunk (Claude with thinking) vs plain string
                # In direct streaming, we send EVERYTHING (thinking + content) to client
                # TODO: Consider adding visual markers for thinking differentiation
                if isinstance(chunk, StreamChunk):
                    chunk_text = chunk.text
                else:
                    chunk_text = chunk
                if chunk_text:
                    yield chunk_text.encode('utf-8')

        except Exception as exc:
            error_msg = f"Error in direct streaming: {str(exc)}"
            yield error_msg.encode('utf-8')

    return StreamingResponse(
        direct_content_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/stream-content/{session_id}")
async def stream_content_direct(session_id: str):
    """Stream only content generation in real-time using SSE format"""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def content_generator():
        last_content_length = 0
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep connection alive

        while True:
            snapshot = await mutate_session(
                session_id,
                lambda session: {
                    "status": session["status"],
                    "partial_content": session.get("partial_content", ""),
                    "last_generated_content": session.get("last_generated_content", "")
                }
            )
            if snapshot is None:
                break

            current_content = snapshot["partial_content"]
            if len(current_content) > last_content_length:
                new_chunk = current_content[last_content_length:]
                chunk_event = json.dumps({"type": "chunk", "content": new_chunk})
                yield f"data: {chunk_event}\n\n".encode("utf-8")
                last_content_length = len(current_content)

            status = snapshot["status"]
            if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                final_content = snapshot.get("last_generated_content", "")
                if len(final_content) > last_content_length:
                    final_chunk = final_content[last_content_length:]
                    final_event = json.dumps({"type": "chunk", "content": final_chunk})
                    yield f"data: {final_event}\n\n".encode("utf-8")
                break

            # Send SSE heartbeat to keep connection alive
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n".encode("utf-8")
                last_heartbeat = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        content_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream-generation/{session_id}")
async def stream_generation_content(session_id: str):
    """Stream only generation content in real-time using SSE format"""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generation_content_generator():
        last_content_length = 0
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep connection alive

        while True:
            snapshot = await mutate_session(
                session_id,
                lambda session: {
                    "status": session["status"],
                    "current_phase": session.get("current_phase", "initializing"),
                    "generation_content": session.get("generation_content", "")
                }
            )
            if snapshot is None:
                break

            # Only stream when in generation phase
            if snapshot["current_phase"] == "generating":
                current_content = snapshot["generation_content"]
                if len(current_content) > last_content_length:
                    new_chunk = current_content[last_content_length:]
                    chunk_event = json.dumps({"type": "chunk", "content": new_chunk})
                    yield f"data: {chunk_event}\n\n".encode("utf-8")
                    last_content_length = len(current_content)

            status = snapshot["status"]
            if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                break

            # Send SSE heartbeat to keep connection alive
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n".encode("utf-8")
                last_heartbeat = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        generation_content_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream-qa/{session_id}")
async def stream_qa_content(session_id: str):
    """Stream only QA evaluation content in real-time using SSE format"""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def qa_content_generator():
        last_content_length = 0
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep connection alive

        while True:
            snapshot = await mutate_session(
                session_id,
                lambda session: {
                    "status": session["status"],
                    "current_phase": session.get("current_phase", "initializing"),
                    "qa_content": session.get("qa_content", "")
                }
            )
            if snapshot is None:
                break

            # Only stream when in QA phase
            if snapshot["current_phase"] == "qa_evaluation":
                current_content = snapshot["qa_content"]
                if len(current_content) > last_content_length:
                    new_chunk = current_content[last_content_length:]
                    chunk_event = json.dumps({"type": "chunk", "content": new_chunk})
                    yield f"data: {chunk_event}\n\n".encode("utf-8")
                    last_content_length = len(current_content)

            status = snapshot["status"]
            if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                break

            # Send SSE heartbeat to keep connection alive
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n".encode("utf-8")
                last_heartbeat = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        qa_content_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream-preflight/{session_id}")
async def stream_preflight_content(session_id: str):
    """Stream only preflight validation content in real-time using SSE format"""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def preflight_content_generator():
        last_content_length = 0
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep connection alive

        while True:
            snapshot = await mutate_session(
                session_id,
                lambda session: {
                    "status": session["status"],
                    "current_phase": session.get("current_phase", "initializing"),
                    "preflight_content": session.get("preflight_content", "")
                }
            )
            if snapshot is None:
                break

            # Stream preflight content (available during and after preflight validation)
            current_content = snapshot["preflight_content"]
            if len(current_content) > last_content_length:
                new_chunk = current_content[last_content_length:]
                chunk_event = json.dumps({"type": "chunk", "content": new_chunk})
                yield f"data: {chunk_event}\n\n".encode("utf-8")
                last_content_length = len(current_content)

            status = snapshot["status"]
            if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                break

            # Send SSE heartbeat to keep connection alive
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n".encode("utf-8")
                last_heartbeat = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        preflight_content_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.get("/stream-gransabio/{session_id}")
async def stream_gran_sabio_content(session_id: str):
    """Stream only Gran Sabio processing content in real-time using SSE format"""
    exists = await mutate_session(session_id, lambda session: True)
    if exists is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def gran_sabio_content_generator():
        last_content_length = 0
        last_heartbeat = time.time()
        HEARTBEAT_INTERVAL = 15  # Send heartbeat every 15 seconds to keep connection alive

        while True:
            snapshot = await mutate_session(
                session_id,
                lambda session: {
                    "status": session["status"],
                    "current_phase": session.get("current_phase", "initializing"),
                    "gransabio_content": session.get("gransabio_content", "")
                }
            )
            if snapshot is None:
                break

            # Stream when in Gran Sabio phases (review or regeneration)
            if snapshot["current_phase"] in ["gran_sabio_review", "gran_sabio_regeneration"]:
                current_content = snapshot["gransabio_content"]
                if len(current_content) > last_content_length:
                    new_chunk = current_content[last_content_length:]
                    chunk_event = json.dumps({"type": "chunk", "content": new_chunk})
                    yield f"data: {chunk_event}\n\n".encode("utf-8")
                    last_content_length = len(current_content)

            status = snapshot["status"]
            if status in [GenerationStatus.COMPLETED, GenerationStatus.FAILED, GenerationStatus.CANCELLED]:
                break

            # Send SSE heartbeat to keep connection alive
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n".encode("utf-8")
                last_heartbeat = now

            await asyncio.sleep(0.1)

    return StreamingResponse(
        gran_sabio_content_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# --- Project-scoped unified stream ---

@app.get("/stream/project/{project_id}")
async def stream_project_unified(
    project_id: str,
    phases: str = Query(
        default="all",
        description="CSV of phases: preflight, generation, qa, consensus, gransabio/gran_sabio, status, or 'all'"
    )
):
    """
    Unified real-time project event stream (SSE).

    Allows selecting which phases to monitor via the `phases` parameter.
    Receives all phases by default.

    **Available phases:**
    - `preflight` - Preflight validation chunks
    - `generation` - Content generation chunks
    - `qa` - QA evaluation chunks
    - `consensus` - Consensus updates
    - `gransabio` or `gran_sabio` - Gran Sabio chunks
    - `status` - Aggregated status (includes initial snapshot)
    - `all` - All of the above (default)

    **Behavior:**
    - All events include "phase" field to identify origin
    - If "status" is included, sends initial snapshot after connection
    - Heartbeat every 15 seconds keeps connection alive
    - Immediate close on cancel (no queue draining)
    - Returns 503 if unable to subscribe to any phase

    **Examples:**
    - `/stream/project/abc123` -> all phases
    - `/stream/project/abc123?phases=generation,qa` -> only those phases
    - `/stream/project/abc123?phases=status` -> status only

    **Errors:**
    - 400: Invalid project_id or invalid phases
    - 503: Unable to subscribe to any phase (service temporarily unavailable)
    """
    # Validate project_id
    if not project_id or len(project_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid project_id")

    # Parse and validate phases
    try:
        parsed_phases = parse_phases(phases)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    manager = ProjectStreamManager(project_id, parsed_phases)

    # Wrap stream to catch SubscriptionError during iteration
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
