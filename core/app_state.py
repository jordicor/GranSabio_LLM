"""
Gran Sabio LLM Engine - Advanced Content Generation API
=====================================================

A sophisticated content generation engine that uses multiple AI models
for generation, multi-layer QA evaluation, and consensus-based approval.

Features:
- Multi-provider AI support (GPT, Claude, Gemini)
- Multi-layer QA evaluation system
- Deal-breaker detection for critical issues
- Gran Sabio escalation for conflict resolution
- Verbose progress tracking
- Flexible content type support

Author: Gran Sabio LLM Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query, Body
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable, TypeVar, Set
import asyncio
import uuid
import ipaddress
# Use optimized JSON (3.6x faster than standard json)
import json_utils as json
import time
import logging
import os
import sys
from datetime import datetime
from debug_logger import (
    DebugLogger,
    get_debug_logger,
    shutdown_debug_logger,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

# Silence noisy third-party loggers to avoid cluttering output
# This prevents low-level HTTP connection logs from httpcore, httpx, etc.
_noisy_loggers = [
    'httpcore',
    'httpcore.connection',
    'httpcore.http11',
    'httpcore.http2',
    'httpx',
    'urllib3',
    'urllib3.connectionpool',
    'openai._base_client',
    'anthropic._base_client',
    'google.auth',
    'google.auth.transport',
    'googleapiclient',
    'aiosqlite',
]
for _logger_name in _noisy_loggers:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
FILE_LOGGING_ENV_VAR = "GRANSABIO_FILE_LOGGING"
FORCE_VERBOSE_ENV_VAR = "GRANSABIO_FORCE_VERBOSE"
FORCE_EXTRA_VERBOSE_ENV_VAR = "GRANSABIO_FORCE_EXTRA_VERBOSE"

if os.getenv(FILE_LOGGING_ENV_VAR, "").lower() in TRUTHY_ENV_VALUES:
    try:
        from file_logger import activate_file_logging, TeeOutput
        if not isinstance(sys.stdout, TeeOutput):
            activate_file_logging()
    except Exception as exc:
        logger.error(f"Failed to activate file logging in worker process: {exc}")

from ai_service import get_ai_service
from qa_engine import QAEngine, QAProcessCancelled
from consensus_engine import ConsensusEngine
from gran_sabio import GranSabioEngine, GranSabioInvocationError, GranSabioProcessCancelled
from deal_breaker_tracker import get_tracker
from preflight_validator import run_preflight_validation
from attachments_router import router as attachments_router, get_attachment_manager
from analysis_router import router as analysis_router
from services.attachment_manager import (
    AttachmentManager,
    AttachmentError,
    AttachmentNotFoundError,
    AttachmentValidationError,
    ResolvedAttachment,
)
from config import Config, config
from models import (
    ContentRequest,
    QALayer,
    GenerationStatus,
    ProgressUpdate,
    ContentResponse,
    GenerationInitResponse,
    ProjectInitRequest,
    ProjectInitResponse,
)
from word_count_utils import (
    validate_word_count_config, 
    create_word_count_qa_layer,
    count_words,
    word_count_config_to_dict,
    is_word_count_enforcement_enabled,
)
from tools.ai_json_cleanroom import validate_ai_json, ValidationResult
from feedback_memory import get_feedback_manager, initialize_feedback_system, shutdown_feedback_system
from usage_tracking import (
    UsageTracker,
    inject_costs_into_json_payload,
    merge_costs_into_json_string,
)

T = TypeVar("T")

app = FastAPI(
    title="Gran Sabio LLM Engine",
    description="Advanced Content Generation API with Multi-Layer QA",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add GZip compression middleware (compresses responses > 1000 bytes)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add IP filter middleware (restricts access to internal networks only)
# This provides temporary protection until user authentication is implemented
# See core/security.py for configuration and allowed networks
from .security import IPFilterMiddleware
app.add_middleware(IPFilterMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount documentation fragments for SPA loading
app.mount("/templates/docs", StaticFiles(directory="templates/docs"), name="docs")

# Routers
app.include_router(attachments_router)
app.include_router(analysis_router)

# Smart Edit routes
from smart_edit.router import router as smart_edit_router
app.include_router(smart_edit_router, prefix="/smart-edit")

# Monitor routes
from .monitor_routes import router as monitor_router
app.include_router(monitor_router)

# Templates
templates = Jinja2Templates(directory="templates")

# Global services - initialized lazily to avoid event loop issues
ai_service = None
qa_engine = None
consensus_engine = None
gran_sabio = None
debug_logger: Optional[DebugLogger] = None

def _ensure_services():
    """Initialize services lazily when needed (within async context)"""
    global ai_service, qa_engine, consensus_engine, gran_sabio
    if ai_service is None:
        ai_service = get_ai_service()
        qa_engine = QAEngine(ai_service=ai_service)
        consensus_engine = ConsensusEngine(ai_service=ai_service)
        gran_sabio = GranSabioEngine(ai_service=ai_service)

# Active sessions storage
active_sessions: Dict[str, Dict] = {}
# Initialized during FastAPI startup to bind the lock to the running event loop
active_sessions_lock: Optional[asyncio.Lock] = None

# Track project identifiers allocated ahead of generation requests
reserved_project_ids: Set[str] = set()

# Track project identifiers that have been cancelled (reject new requests)
cancelled_project_ids: Set[str] = set()

# Project-phase streaming subscribers.
# For each project and phase we keep a list of asyncio.Queue subscribers.
# We intentionally do NOT buffer historical chunks; queues only carry live events.
project_phase_streams: Dict[str, Dict[str, List[asyncio.Queue[str]]]] = {}
project_phase_streams_lock: Optional[asyncio.Lock] = None

# Project status streaming subscribers (project-level snapshots on status changes)
project_status_streams: Dict[str, List[asyncio.Queue[str]]] = {}
project_status_streams_lock: Optional[asyncio.Lock] = None



def _serialize_for_debug(value: Any) -> Any:
    """Convert complex objects into JSON-serializable structures for debugger."""
    if value is None:
        return None

    if isinstance(value, BaseModel):
        try:
            return value.model_dump(mode="json", exclude_unset=False)  # type: ignore[attr-defined]
        except Exception:
            return value.dict()  # type: ignore[attr-defined]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {key: _serialize_for_debug(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_debug(item) for item in value]

    summary_callable = getattr(value, "summary", None)
    if callable(summary_callable):
        try:
            return summary_callable()
        except Exception:
            pass

    if hasattr(value, "__dict__") and not isinstance(value, str):
        return {
            key: _serialize_for_debug(val)
            for key, val in value.__dict__.items()
            if not key.startswith("_")
        }

    return value


def _parse_debug_json(value: Optional[str]) -> Any:
    """Best-effort JSON parsing for debugger API responses."""
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


async def _debug_session_start(
    session_id: str,
    *,
    request_payload: Any,
    preflight_payload: Any,
    project_id: Optional[str] = None,
    attachments: Optional[List[ResolvedAttachment]] = None,
    preflight_context: Optional[List[str]] = None,
) -> None:
    """Persist session initialization metadata."""
    if not debug_logger:
        return
    try:
        await debug_logger.record_session_start(
            session_id,
            request_payload=_serialize_for_debug(request_payload),
            preflight_payload=_serialize_for_debug(preflight_payload),
            status="initializing",
            project_id=project_id,
        )
        details = {
            "request": _serialize_for_debug(request_payload),
            "preflight": _serialize_for_debug(preflight_payload),
            "project_id": project_id,
            "attachments": [_serialize_for_debug(item) for item in (attachments or [])],
            "preflight_context": preflight_context or [],
        }
        await debug_logger.record_event(
            session_id,
            event_type="session_initialized",
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "details": details,
            },
        )
    except Exception:
        logger.exception("Failed to record debugger session start for %s", session_id)


async def _debug_record_event(session_id: str, event_type: str, payload: Any) -> None:
    """Safely store chronological events for a session."""
    if not debug_logger:
        return
    try:
        await debug_logger.record_event(
            session_id,
            event_type=event_type,
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "data": _serialize_for_debug(payload),
            },
        )
    except Exception:
        logger.exception("Failed to record debugger event %s for %s", event_type, session_id)


async def _debug_update_status(
    session_id: str,
    *,
    status: Optional[str] = None,
    final_payload: Any = None,
) -> None:
    """Update debugger session status."""
    if not debug_logger:
        return
    try:
        await debug_logger.update_session_status(
            session_id,
            status=status,
            final_payload=_serialize_for_debug(final_payload) if final_payload is not None else None,
        )
    except Exception:
        logger.exception("Failed to update debugger status for %s", session_id)


async def _debug_record_usage(session_id: str, usage_payload: Any) -> None:
    """Store usage/cost metadata."""
    if not debug_logger:
        return
    try:
        await debug_logger.record_usage_summary(session_id, _serialize_for_debug(usage_payload))
    except Exception:
        logger.exception("Failed to record debugger usage summary for %s", session_id)

@app.on_event("startup")
async def startup_event():
    """Initialize services and background tasks on startup"""
    global active_sessions_lock, project_phase_streams_lock, project_status_streams_lock, debug_logger
    active_sessions_lock = asyncio.Lock()
    project_phase_streams_lock = asyncio.Lock()
    project_status_streams_lock = asyncio.Lock()

    # Initialize services early to catch any configuration issues
    try:
        _ensure_services()
        logger.info("AI services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI services: {e}")
        # Don't fail startup, services will be initialized on first use

    # Initialize debugger persistence
    try:
        debug_logger = await get_debug_logger(
            enabled=config.DEBUGGER.enabled,
            db_path=config.DEBUGGER.db_path,
        )
        await debug_logger.initialize()
        if config.DEBUGGER.enabled:
            logger.info("Debugger persistence initialized")
            # Cleanup old sessions on startup (older than 7 days)
            cleanup_result = await debug_logger.cleanup_old_sessions(retention_days=7)
            if cleanup_result.get("deleted_sessions", 0) > 0:
                logger.info(
                    "Debugger cleanup: removed %d sessions and %d events",
                    cleanup_result["deleted_sessions"],
                    cleanup_result["deleted_events"]
                )
    except Exception as e:
        logger.error(f"Failed to initialize debugger persistence: {e}")
        debug_logger = None

    # Initialize session cleanup
    await perform_session_cleanup()
    asyncio.create_task(session_cleanup_loop())

    # Initialize feedback memory system
    try:
        await initialize_feedback_system()
        logger.info("Feedback memory system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize feedback memory system: {e}")
        # Don't fail startup, system can work without feedback memory

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Gran Sabio LLM Engine...")

    # Close the shared AI service instance connections
    try:
        # Get the shared AI service instance
        shared_ai_service = get_ai_service()
        if shared_ai_service and hasattr(shared_ai_service, 'close'):
            if asyncio.iscoroutinefunction(shared_ai_service.close):
                await shared_ai_service.close()
                logger.info("AI service connections closed successfully")
            else:
                shared_ai_service.close()
                logger.info("AI service connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing AI service: {e}")

    # Shutdown feedback memory system
    try:
        await shutdown_feedback_system()
        logger.info("Feedback memory system closed successfully")
    except Exception as e:
        logger.error(f"Error closing feedback memory system: {e}")

    # Shutdown debugger persistence
    try:
        await shutdown_debug_logger()
        logger.info("Debugger persistence closed successfully")
    except Exception as e:
        logger.error(f"Error shutting down debugger persistence: {e}")

    logger.info("Shutdown complete")


async def register_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """Store or replace a session under the shared lock."""
    lock = active_sessions_lock
    if lock is None:
        active_sessions[session_id] = session_data
        return
    async with lock:
        active_sessions[session_id] = session_data


async def mutate_session(session_id: str, mutator: Callable[[Dict[str, Any]], T]) -> Optional[T]:
    """Run a mutator while holding the session lock."""
    lock = active_sessions_lock
    if lock is None:
        session = active_sessions.get(session_id)
        if session is None:
            return None
        return mutator(session)
    async with lock:
        session = active_sessions.get(session_id)
        if session is None:
            return None
        return mutator(session)


async def pop_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Remove a session safely and return it if present."""
    lock = active_sessions_lock
    if lock is None:
        return active_sessions.pop(session_id, None)
    async with lock:
        return active_sessions.pop(session_id, None)


async def session_exists(session_id: str) -> bool:
    """Check if a session exists under lock."""
    lock = active_sessions_lock
    if lock is None:
        return session_id in active_sessions
    async with lock:
        return session_id in active_sessions


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a session safely under lock."""
    lock = active_sessions_lock
    if lock is None:
        return active_sessions.get(session_id)
    async with lock:
        return active_sessions.get(session_id)


def _store_final_result(session: Dict[str, Any], final_result: Dict[str, Any], session_id: Optional[str] = None) -> None:
    """Persist the final_result for a session while attaching project metadata."""
    project_id = session.get("project_id")
    if project_id:
        final_result.setdefault("project_id", project_id)
    session["final_result"] = final_result
    status = final_result.get("status") or session.get("status")
    session_identifier = session_id or session.get("session_id")  # session_id not stored; caller should pass when available
    if status and session_identifier:
        queue_project_session_end(session, session_identifier, str(status))


def _reserve_project_id(project_id: str) -> None:
    """Track project identifiers that clients intend to reuse."""
    if project_id:
        reserved_project_ids.add(project_id)


def _is_project_reserved(project_id: str) -> bool:
    """Check if a project identifier has already been reserved."""
    if not project_id:
        return False
    if project_id in reserved_project_ids:
        return True
    # Fallback check across active sessions to treat in-flight IDs as reserved
    # Use defensive copy to avoid RuntimeError if dict changes during iteration
    return any(session.get("project_id") == project_id for session in list(active_sessions.values()))


def _start_project(project_id: str) -> bool:
    """
    Activate a project, removing it from cancelled state if present.

    This should be called when a client wants to start/resume generation
    for a project that may have been previously cancelled.

    Args:
        project_id: The project identifier to activate.

    Returns:
        True if the project was previously cancelled (and is now active),
        False if it was already active.
    """
    if not project_id:
        return False
    was_cancelled = project_id in cancelled_project_ids
    cancelled_project_ids.discard(project_id)
    return was_cancelled


def _stop_project(project_id: str) -> int:
    """
    Cancel a project and mark all its active sessions as cancelled.

    This should be called when a client wants to stop all generation
    activity for a project. New requests with this project_id will be
    rejected until _start_project() is called.

    Args:
        project_id: The project identifier to cancel.

    Returns:
        Number of active sessions that were cancelled.
    """
    if not project_id:
        return 0

    cancelled_project_ids.add(project_id)

    # Broadcast project_end to all phases and close their streams
    phases = ["preflight", "generation", "qa", "smart_edit", "consensus", "gran_sabio"]
    for phase in phases:
        # Fire-and-forget; if no loop running, just skip
        try:
            asyncio.create_task(
                publish_project_phase_chunk(
                    project_id,
                    phase,
                    content=None,
                    event="project_end",
                    status="cancelled",
                    end_stream=True,
                )
            )
        except RuntimeError:
            logger.debug("Unable to publish project_end for project %s phase %s (no running loop)", project_id, phase)

    # Cancel all active sessions belonging to this project
    # Use defensive copy to avoid RuntimeError if dict changes during iteration
    cancelled_count = 0
    for session_id, session in list(active_sessions.items()):
        if session.get("project_id") == project_id:
            if not session.get("cancelled"):
                session["cancelled"] = True
                update_session_status(session, session_id, GenerationStatus.CANCELLED)
                cancelled_count += 1

    # Notify status subscribers and close status streams
    try:
        asyncio.create_task(publish_project_status_event(project_id, "project", "project_cancelled"))
        asyncio.create_task(close_project_status_stream(project_id, "project_cancelled"))
    except RuntimeError:
        logger.debug("Unable to publish project cancelled event (no running loop)")

    return cancelled_count


def _is_project_cancelled(project_id: str) -> bool:
    """Check if a project has been cancelled and should reject new requests."""
    if not project_id:
        return False
    return project_id in cancelled_project_ids


# --- Project-phase streaming utilities ---

async def subscribe_project_phase(project_id: str, phase: str) -> asyncio.Queue[str]:
    """
    Register a subscriber queue for a project/phase stream.

    The queue carries live chunks only (no history). Caller is responsible for
    calling unsubscribe_project_phase when finished.
    """
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
    if not project_id:
        return queue

    lock = project_phase_streams_lock
    if lock is None:
        phase_map = project_phase_streams.setdefault(project_id, {})
        phase_map.setdefault(phase, []).append(queue)
        return queue

    async with lock:
        phase_map = project_phase_streams.setdefault(project_id, {})
        phase_map.setdefault(phase, []).append(queue)
    return queue


async def unsubscribe_project_phase(project_id: str, phase: str, queue: asyncio.Queue[str]) -> None:
    """Remove a subscriber queue from a project/phase stream."""
    if not project_id:
        return

    lock = project_phase_streams_lock
    if lock is None:
        phase_map = project_phase_streams.get(project_id)
        if not phase_map:
            return
        subscribers = phase_map.get(phase, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers:
            phase_map.pop(phase, None)
        if not phase_map:
            project_phase_streams.pop(project_id, None)
        return

    async with lock:
        phase_map = project_phase_streams.get(project_id)
        if not phase_map:
            return
        subscribers = phase_map.get(phase, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers:
            phase_map.pop(phase, None)
        if not phase_map:
            project_phase_streams.pop(project_id, None)


def _build_project_phase_event(
    *,
    event: str,
    phase: str,
    session_id: Optional[str] = None,
    request_name: Optional[str] = None,
    content: Optional[str] = None,
    status: Optional[str] = None,
    # Retry-specific fields
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    reason: Optional[str] = None,
    retry_in_seconds: Optional[float] = None,
    is_api_error: Optional[bool] = None,
    provider: Optional[str] = None,
    # QA-specific fields for filtering
    qa_layer: Optional[str] = None,
    qa_model: Optional[str] = None,
    # Smart-edit specific fields
    edit_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a JSON event string for project/phase streams."""
    obj: Dict[str, Any] = {
        "type": event,
        "phase": phase,
        "timestamp": int(time.time() * 1000),
    }
    if session_id:
        obj["session_id"] = session_id
    if request_name:
        obj["request_name"] = request_name
    if status:
        obj["status"] = status
    if content is not None:
        obj["content"] = content
    # Retry fields
    if attempt is not None:
        obj["attempt"] = attempt
    if max_attempts is not None:
        obj["max_attempts"] = max_attempts
    if reason is not None:
        obj["reason"] = reason
    if retry_in_seconds is not None:
        obj["retry_in_seconds"] = retry_in_seconds
    if is_api_error is not None:
        obj["is_api_error"] = is_api_error
    if provider is not None:
        obj["provider"] = provider
    # QA filter fields
    if qa_layer is not None:
        obj["qa_layer"] = qa_layer
    if qa_model is not None:
        obj["qa_model"] = qa_model
    # Smart-edit data
    if edit_data is not None:
        obj["edit_data"] = edit_data
    return json.dumps(obj, ensure_ascii=True)


async def publish_project_phase_chunk(
    project_id: Optional[str],
    phase: str,
    content: Optional[str],
    *,
    session_id: Optional[str] = None,
    request_name: Optional[str] = None,
    event: str = "chunk",
    status: Optional[str] = None,
    end_stream: bool = False,
    # Retry-specific fields
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    reason: Optional[str] = None,
    retry_in_seconds: Optional[float] = None,
    is_api_error: Optional[bool] = None,
    provider: Optional[str] = None,
    # QA-specific fields for filtering
    qa_layer: Optional[str] = None,
    qa_model: Optional[str] = None,
    # Smart-edit specific fields
    edit_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Broadcast a live event to all subscribers of a project/phase stream.

    - If end_stream=True, a sentinel is sent after the event to close the stream.
    - If the queue is full, the event is dropped to avoid blocking producers.
    """
    if not project_id:
        return

    lock = project_phase_streams_lock
    targets: List[asyncio.Queue[str]] = []

    if lock is None:
        phase_map = project_phase_streams.get(project_id)
        if phase_map:
            targets = list(phase_map.get(phase, []))
    else:
        async with lock:
            phase_map = project_phase_streams.get(project_id)
            if phase_map:
                targets = list(phase_map.get(phase, []))

    if not targets:
        return

    payload = _build_project_phase_event(
        event=event,
        phase=phase,
        session_id=session_id,
        request_name=request_name,
        content=content,
        status=status,
        attempt=attempt,
        max_attempts=max_attempts,
        reason=reason,
        retry_in_seconds=retry_in_seconds,
        is_api_error=is_api_error,
        provider=provider,
        qa_layer=qa_layer,
        qa_model=qa_model,
        edit_data=edit_data,
    )

    for q in targets:
        try:
            q.put_nowait(payload)
            if end_stream:
                q.put_nowait(None)  # type: ignore[arg-type]
        except asyncio.QueueFull:
            logger.debug("Dropping project-phase event (queue full) for project %s phase %s", project_id, phase)


async def publish_project_session_end(
    project_id: Optional[str],
    session_id: str,
    status: str,
    request_name: Optional[str] = None,
) -> None:
    """Publish a session_end event to all phases for a project."""
    if not project_id:
        return
    phases = ["preflight", "generation", "qa", "smart_edit", "consensus", "gran_sabio"]
    for phase in phases:
        await publish_project_phase_chunk(
            project_id,
            phase,
            content=None,
            session_id=session_id,
            request_name=request_name,
            event="session_end",
            status=status,
            end_stream=False,
        )


def queue_project_session_end(session: Dict[str, Any], session_id: str, status: str) -> None:
    """Schedule session_end events without blocking current flow."""
    project_id = session.get("project_id")
    if not project_id:
        return
    request_name = session.get("request_name")
    try:
        asyncio.create_task(publish_project_session_end(project_id, session_id, status, request_name))
    except RuntimeError:
        # If no running loop, skip publishing
        logger.debug("Unable to schedule project session end event (no running loop)")


async def subscribe_project_status(project_id: str) -> asyncio.Queue[str]:
    """Register a subscriber queue for project-level status updates."""
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
    if not project_id:
        return queue

    lock = project_status_streams_lock
    if lock is None:
        project_status_streams.setdefault(project_id, []).append(queue)
        return queue

    async with lock:
        project_status_streams.setdefault(project_id, []).append(queue)
    return queue


async def unsubscribe_project_status(project_id: str, queue: asyncio.Queue[str]) -> None:
    """Remove a subscriber queue from project status stream."""
    if not project_id:
        return

    lock = project_status_streams_lock
    if lock is None:
        subscribers = project_status_streams.get(project_id, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers:
            project_status_streams.pop(project_id, None)
        return

    async with lock:
        subscribers = project_status_streams.get(project_id, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers:
            project_status_streams.pop(project_id, None)


async def publish_project_status_event(
    project_id: Optional[str],
    session_id: str,
    event_type: str = "status_change",
) -> None:
    """Broadcast a project status update to all subscribers."""
    if not project_id:
        return

    lock = project_status_streams_lock
    targets: List[asyncio.Queue[str]] = []

    if lock is None:
        targets = list(project_status_streams.get(project_id, []))
    else:
        async with lock:
            targets = list(project_status_streams.get(project_id, []))

    if not targets:
        return

    status_data = await get_project_status(project_id)
    event_payload = {
        "type": event_type,
        "timestamp": int(time.time() * 1000),
        "trigger_session": session_id,
        "project": status_data,
    }
    payload_str = json.dumps(event_payload, ensure_ascii=True, default=str)

    for q in targets:
        try:
            q.put_nowait(payload_str)
        except asyncio.QueueFull:
            logger.debug("Dropping project status event (queue full) for project %s", project_id)


async def close_project_status_stream(project_id: str, reason: str = "project_ended") -> None:
    """Send stream_end and close all project status subscribers."""
    if not project_id:
        return

    lock = project_status_streams_lock
    targets: List[asyncio.Queue[str]] = []

    if lock is None:
        targets = list(project_status_streams.get(project_id, []))
    else:
        async with lock:
            targets = list(project_status_streams.get(project_id, []))

    if not targets:
        return

    final_event = {
        "type": "stream_end",
        "timestamp": int(time.time() * 1000),
        "reason": reason,
    }
    final_str = json.dumps(final_event, ensure_ascii=True)

    for q in targets:
        try:
            q.put_nowait(final_str)
            q.put_nowait(None)  # type: ignore[arg-type]
        except asyncio.QueueFull:
            try:
                q.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                logger.debug("Unable to close status stream queue for project %s (queue full)", project_id)


def queue_project_status_event(session: Dict[str, Any], session_id: str, event_type: str = "status_change") -> None:
    """Schedule a project-level status update without blocking."""
    project_id = session.get("project_id")
    if not project_id:
        return
    try:
        asyncio.create_task(publish_project_status_event(project_id, session_id, event_type))
    except RuntimeError:
        logger.debug("Unable to schedule project status event (no running loop)")


def update_session_status(
    session: Dict[str, Any],
    session_id: str,
    status: "GenerationStatus",
    phase: Optional[str] = None,
) -> None:
    """Set session status (and optional phase) and trigger project status hook."""
    session["status"] = status
    if phase is not None:
        session["current_phase"] = phase
    queue_project_status_event(session, session_id)


def update_session_phase(session: Dict[str, Any], session_id: str, phase: str) -> None:
    """Set current phase and trigger project status hook."""
    session["current_phase"] = phase
    queue_project_status_event(session, session_id)


def update_session_iteration(session: Dict[str, Any], session_id: str, iteration: int) -> None:
    """Set current iteration and trigger project status hook."""
    session["current_iteration"] = iteration
    queue_project_status_event(session, session_id)


def update_qa_progress_reset(
    session: Dict[str, Any],
    session_id: str,
    total_evaluations: int,
) -> None:
    """Reset QA progress tracking for a new QA phase."""
    session["qa_evaluations_total"] = total_evaluations
    session["qa_evaluations_completed"] = 0
    session["current_qa_model"] = None
    session["current_qa_layer"] = None
    queue_project_status_event(session, session_id)


def update_qa_evaluation_started(session: Dict[str, Any], session_id: str, model: str, layer: str) -> None:
    """
    Track current QA model/layer and trigger event only when it changes.
    """
    prev_model = session.get("current_qa_model")
    prev_layer = session.get("current_qa_layer")
    if prev_model == model and prev_layer == layer:
        return
    session["current_qa_model"] = model
    session["current_qa_layer"] = layer
    queue_project_status_event(session, session_id)


def update_qa_evaluation_completed(session: Dict[str, Any], session_id: str) -> None:
    """Increment QA completion counter and trigger project status hook."""
    session["qa_evaluations_completed"] = session.get("qa_evaluations_completed", 0) + 1
    queue_project_status_event(session, session_id)


def update_consensus_result(
    session: Dict[str, Any],
    session_id: str,
    score: Optional[float],
    approved: bool = False,
) -> None:
    """Store consensus score, approval flag, clear QA tracking, and trigger hook."""
    session["last_consensus_score"] = score
    if approved:
        session["approved"] = True
    session["current_qa_model"] = None
    session["current_qa_layer"] = None
    queue_project_status_event(session, session_id)


async def perform_session_cleanup() -> None:
    """Remove expired sessions and trim verbose logs respecting configuration."""
    lock = active_sessions_lock
    now = datetime.now()
    timeout_seconds = getattr(config, 'SESSION_TIMEOUT', 3600)
    max_entries = max(1, getattr(config, 'VERBOSE_MAX_ENTRIES', 100))
    if timeout_seconds <= 0 and max_entries <= 0:
        return

    expired_sessions: List[str] = []
    trimmed_logs: List[tuple[str, int]] = []

    def _cleanup_sessions(items):
        nonlocal expired_sessions, trimmed_logs
        for session_id, session in items:
            created_at = session.get('created_at')
            status = session.get('status')
            if isinstance(created_at, datetime) and timeout_seconds > 0:
                try:
                    age_seconds = (now - created_at).total_seconds()
                except Exception:
                    age_seconds = timeout_seconds + 1
                is_final = status in {
                    GenerationStatus.COMPLETED,
                    GenerationStatus.FAILED,
                    GenerationStatus.CANCELLED
                }
                if age_seconds > timeout_seconds and is_final:
                    expired_sessions.append(session_id)
                    continue

            verbose_log = session.get('verbose_log')
            if isinstance(verbose_log, list) and len(verbose_log) > max_entries:
                trim_count = len(verbose_log) - max_entries
                session['verbose_log'] = verbose_log[-max_entries:]
                trimmed_logs.append((session_id, trim_count))

    if lock is None:
        _cleanup_sessions(list(active_sessions.items()))
        for session_id in expired_sessions:
            active_sessions.pop(session_id, None)
    else:
        async with lock:
            _cleanup_sessions(list(active_sessions.items()))
            for session_id in expired_sessions:
                active_sessions.pop(session_id, None)

    if expired_sessions:
        logger.info('Cleaned up %d expired session(s): %s', len(expired_sessions), ', '.join(expired_sessions))
    if trimmed_logs:
        summary = ', '.join(f"{sid} (-{count})" for sid, count in trimmed_logs)
        logger.debug('Trimmed verbose logs: %s', summary)


async def session_cleanup_loop() -> None:
    """Periodic background task that performs session maintenance."""
    interval = max(1, getattr(config, 'SESSION_CLEANUP_INTERVAL', 300))
    logger.info('Session cleanup loop started (interval=%ss)', interval)
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await perform_session_cleanup()
            except Exception as exc:
                logger.exception('Session cleanup iteration failed: %s', exc)
    except asyncio.CancelledError:
        logger.info('Session cleanup loop cancelled')
        raise


# --- Project Status Utilities ---

async def get_project_status(project_id: str) -> dict:
    """
    Retrieve the status of all sessions belonging to a project.

    Returns a JSON-serializable dict with the structure:
    {
        "project_id": str,
        "status": "idle" | "running" | "completed" | "failed" | "cancelled",
        "sessions": [...],
        "summary": { "total_sessions": int, "active_sessions": int, "completed_sessions": int }
    }
    """
    from word_count_utils import count_words  # Avoid circular import

    sessions_data = []
    active_count = 0
    completed_count = 0
    last_status: Optional[str] = None
    last_activity: Optional[datetime] = None

    lock = active_sessions_lock

    def _extract_session_data(items):
        nonlocal active_count, completed_count, last_status, last_activity
        for session_id, session in items:
            if session.get("project_id") != project_id:
                continue

            request = session.get("request")
            status_enum = session.get("status")
            status_str = status_enum.value if hasattr(status_enum, "value") else str(status_enum)

            # Determine phase
            phase = session.get("current_phase", "initializing")

            # Check activity time
            activity_at = session.get("last_activity_at") or session.get("created_at")
            if activity_at and (last_activity is None or activity_at > last_activity):
                last_activity = activity_at
                last_status = status_str

            # Count session states
            if status_str in ("completed",):
                completed_count += 1
            elif status_str in ("generating", "qa_evaluation", "consensus", "running", "initializing"):
                active_count += 1

            # Get generation info
            content = session.get("last_generated_content")
            content_length = session.get("last_generated_content_length")
            word_count = session.get("last_generated_content_word_count")
            if not content:
                content = session.get("generation_content") or ""
                content_length = session.get("generation_content_length")
                word_count = session.get("generation_content_word_count")
            if content_length is None or (content and content_length == 0):
                content_length = len(content)
            if word_count is None or (content and word_count == 0):
                word_count = count_words(content) if content else 0

            # Get QA info
            qa_models_raw = session.get("qa_models_config") or []
            qa_model_names = []
            for m in qa_models_raw:
                if isinstance(m, str):
                    qa_model_names.append(m)
                elif hasattr(m, "model"):
                    qa_model_names.append(m.model)
                elif isinstance(m, dict):
                    qa_model_names.append(m.get("model", "unknown"))

            qa_layer_names = session.get("qa_layer_names") or []

            # Current QA progress
            current_qa_model = session.get("current_qa_model")
            current_qa_layer = session.get("current_qa_layer")
            qa_completed = session.get("qa_evaluations_completed", 0)
            qa_total = session.get("qa_evaluations_total", 0)

            # Consensus info
            last_consensus = session.get("last_consensus_score")
            min_required = session.get("min_global_score", 8.0)
            approved = session.get("approved", False)

            # Gran Sabio info
            gran_sabio_active = phase in ("gran_sabio_review", "gran_sabio_regeneration")
            gran_sabio_model = session.get("gran_sabio_model", "")
            escalation_count = session.get("gran_sabio_escalation_count", 0)

            session_info = {
                "session_id": session_id,
                "request_name": session.get("request_name"),
                "status": status_str,
                "phase": phase,
                "iteration": session.get("current_iteration", 0),
                "max_iterations": session.get("max_iterations", 0),
                "started_at": session.get("created_at").isoformat() if session.get("created_at") else None,
                "generation": {
                    "model": request.generator_model if request else None,
                    "content_length": content_length,
                    "word_count": word_count,
                },
                "qa": {
                    "models": qa_model_names,
                    "layers": qa_layer_names,
                    "current_model": current_qa_model,
                    "current_layer": current_qa_layer,
                    "progress": {
                        "completed": qa_completed,
                        "total": qa_total,
                    },
                },
                "consensus": {
                    "last_score": last_consensus,
                    "min_required": min_required,
                    "approved": approved,
                },
                "gran_sabio": {
                    "active": gran_sabio_active,
                    "model": gran_sabio_model,
                    "escalation_count": escalation_count,
                },
            }
            sessions_data.append(session_info)

    if lock is None:
        _extract_session_data(list(active_sessions.items()))
    else:
        async with lock:
            _extract_session_data(list(active_sessions.items()))

    # Determine overall project status
    total_sessions = len(sessions_data)
    if total_sessions == 0:
        # Check if project is reserved but idle
        if project_id in reserved_project_ids:
            project_status = "idle"
        elif project_id in cancelled_project_ids:
            project_status = "cancelled"
        else:
            project_status = "idle"
    elif active_count > 0:
        project_status = "running"
    elif project_id in cancelled_project_ids:
        project_status = "cancelled"
    elif last_status == "completed":
        project_status = "completed"
    elif last_status in ("failed", "error"):
        project_status = "failed"
    else:
        project_status = "idle"

    return {
        "project_id": project_id,
        "status": project_status,
        "sessions": sessions_data,
        "summary": {
            "total_sessions": total_sessions,
            "active_sessions": active_count,
            "completed_sessions": completed_count,
        },
    }
