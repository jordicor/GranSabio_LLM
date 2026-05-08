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

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.gzip import GZipMiddleware

# Use optimized JSON (3.6x faster than standard json)
import json_utils as json
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
        from file_logger import TeeOutput, activate_file_logging
        if not isinstance(sys.stdout, TeeOutput):
            activate_file_logging()
    except Exception as exc:
        logger.error(f"Failed to activate file logging in worker process: {exc}")

from ai_service import get_ai_service
from analysis_router import router as analysis_router
from attachments_router import router as attachments_router
from config import config
from consensus_engine import ConsensusEngine
from deal_breaker_tracker import get_tracker
from feedback_memory import get_feedback_manager, initialize_feedback_system, shutdown_feedback_system
from gran_sabio import GranSabioEngine
from models import (
    GenerationStatus,
)
from qa_engine import QAEngine
from services.attachment_manager import (
    ResolvedAttachment,
)
from .cancellation import CancelMode, cancellation_registry

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
        # GranSabioEngine is a process-wide singleton, so session context is
        # not available at construction time. Pass ``tool_event_callback=None``
        # here — the feature degrades gracefully (no live ``/stream/project``
        # events for GranSabio tool-loop turns). Per-session wiring would
        # require constructing a fresh engine per request, which is out of
        # scope for this change.
        gran_sabio = GranSabioEngine(ai_service=ai_service, tool_event_callback=None)

# Active sessions storage
active_sessions: Dict[str, Dict] = {}
# Initialized during FastAPI startup to bind the lock to the running event loop
active_sessions_lock: Optional[asyncio.Lock] = None
_project_hard_stop_completion_tasks: Set[asyncio.Task] = set()
_project_session_end_tasks: Set[asyncio.Task] = set()
_debug_status_update_tasks: Set[asyncio.Task] = set()
_BACKGROUND_DRAIN_TIMEOUT_SECONDS = 2.0
_DEBUG_STATUS_UPDATE_TIMEOUT_SECONDS = 2.0


def _track_background_task(task: asyncio.Task, task_set: Set[asyncio.Task]) -> None:
    task_set.add(task)
    task.add_done_callback(task_set.discard)


def _log_background_task_result(task: asyncio.Task) -> None:
    task_name = task.get_name() if hasattr(task, "get_name") else "background-task"
    try:
        task.result()
    except asyncio.CancelledError:
        logger.debug("Background task was cancelled: %s", task_name)
    except Exception:
        logger.exception("Background task failed: %s", task_name)


async def _drain_background_tasks(
    task_set: Set[asyncio.Task],
    *,
    label: str,
    timeout: float = _BACKGROUND_DRAIN_TIMEOUT_SECONDS,
) -> None:
    """Give tracked background writes a bounded chance to finish."""
    pending = [task for task in list(task_set) if not task.done()]
    if not pending:
        return
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in done:
        _log_background_task_result(task)
    for task in still_pending:
        task.cancel()
    if still_pending:
        await asyncio.gather(*still_pending, return_exceptions=True)
        logger.warning(
            "Cancelled %d pending %s task(s) during shutdown after %.1fs",
            len(still_pending),
            label,
            timeout,
        )


def _track_project_hard_stop_task(task: asyncio.Task) -> None:
    _track_background_task(task, _project_hard_stop_completion_tasks)


def _log_project_hard_stop_task_result(task: asyncio.Task) -> None:
    task_name = task.get_name() if hasattr(task, "get_name") else "project-hard-stop"
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("Detached project hard-stop task was cancelled: %s", task_name)
    except Exception:
        logger.exception("Detached project hard-stop task failed: %s", task_name)

# Track project identifiers allocated ahead of generation requests
reserved_project_ids: Set[str] = set()

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


def _status_to_debug_value(status: Any) -> Optional[str]:
    """Convert status enum/string values to debugger status text."""
    if status is None:
        return None
    return status.value if isinstance(status, GenerationStatus) else str(status)


async def _debug_update_status_with_timeout(
    session_id: str,
    *,
    status: Optional[str] = None,
    final_payload: Any = None,
    timeout: float = _DEBUG_STATUS_UPDATE_TIMEOUT_SECONDS,
) -> None:
    """Run a debugger status update with a small upper bound."""
    try:
        await asyncio.wait_for(
            _debug_update_status(
                session_id,
                status=status,
                final_payload=final_payload,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out updating debugger status for %s after %.1fs",
            session_id,
            timeout,
        )


async def _debug_record_session_transition(
    session_id: str,
    *,
    status: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    """Persist relevant non-terminal lifecycle transitions in the debugger DB."""
    if status is not None:
        await _debug_update_status_with_timeout(session_id, status=status)
    await _debug_record_event(
        session_id,
        "session_transition",
        {
            "status": status,
            "phase": phase,
        },
    )


async def _debug_record_session_cancelled(
    session_id: str,
    *,
    reason: str,
    cancel_mode: str,
    final_result: Dict[str, Any],
) -> None:
    """Persist terminal cancellation details in the debugger DB."""
    await _debug_record_event(
        session_id,
        "session_cancelled",
        {
            "reason": reason,
            "cancel_mode": cancel_mode,
            "final_result": final_result,
        },
    )
    await _debug_update_status_with_timeout(
        session_id,
        status=GenerationStatus.CANCELLED.value,
        final_payload=final_result,
    )


def queue_debug_session_transition(
    session_id: str,
    *,
    status: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    """Schedule debugger lifecycle persistence for synchronous session mutators."""
    if not debug_logger or (status is None and phase is None):
        return
    try:
        task = asyncio.create_task(
            _debug_record_session_transition(session_id, status=status, phase=phase),
            name=f"debug-session-transition:{session_id}",
        )
        _track_background_task(task, _debug_status_update_tasks)
        task.add_done_callback(_log_background_task_result)
    except RuntimeError:
        logger.debug("Unable to schedule debugger session transition (no running loop)")

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
        if config.DEBUGGER.enabled and debug_logger.storage_available:
            logger.info("Debugger persistence initialized")
            # Cleanup old sessions on startup (older than 7 days)
            cleanup_result = await debug_logger.cleanup_old_sessions(retention_days=7)
            if cleanup_result.get("deleted_sessions", 0) > 0:
                logger.info(
                    "Debugger cleanup: removed %d sessions and %d events",
                    cleanup_result["deleted_sessions"],
                    cleanup_result["deleted_events"]
                )
        elif config.DEBUGGER.enabled and debug_logger.storage_disabled_reason:
            logger.warning(
                "Debugger persistence disabled during startup: %s",
                debug_logger.storage_disabled_reason,
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
        # Run cleanup for feedback memory on startup
        feedback_mgr = get_feedback_manager()
        await feedback_mgr.cleanup_old_data()
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

    try:
        await _drain_background_tasks(
            _project_session_end_tasks,
            label="project session-end",
        )
        await _drain_background_tasks(
            _debug_status_update_tasks,
            label="debug status update",
        )
    except Exception as e:
        logger.error(f"Error draining lifecycle background tasks: {e}")

    # Shutdown debugger persistence
    try:
        await shutdown_debug_logger()
        logger.info("Debugger persistence closed successfully")
    except Exception as e:
        logger.error(f"Error shutting down debugger persistence: {e}")

    logger.info("Shutdown complete")


async def register_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """Store or replace a session under the shared lock."""
    guard = await cancellation_registry.get_session_commit_guard(session_id)
    session_data["_hard_cancel_event"] = guard.dispatch_sealed_event

    def _store() -> None:
        existing = active_sessions.get(session_id)
        if existing is not None and is_terminal_session(existing) and not is_terminal_session(session_data):
            existing["late_writes_blocked"] = existing.get("late_writes_blocked", 0) + 1
            logger.debug(
                "Blocked non-terminal session overwrite after terminal status for session %s",
                session_id,
            )
            return
        active_sessions[session_id] = session_data

    lock = active_sessions_lock
    if lock is None:
        _store()
        return
    async with lock:
        _store()


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


async def mutate_session_if_not_hard_cancelled(
    session_id: str,
    mutator: Callable[[Dict[str, Any]], T],
) -> Optional[T]:
    """Run a synchronous session mutation unless the registry hard-stop seal is set."""
    guard = await cancellation_registry.get_session_commit_guard(session_id)
    if guard.hard_cancelled():
        await increment_late_writes_blocked(session_id)
        return None

    def _guarded(session: Dict[str, Any]) -> Optional[T]:
        if guard.hard_cancelled() or session.get("hard_cancelled"):
            session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1
            return None
        return mutator(session)

    return await mutate_session(session_id, _guarded)


async def increment_late_writes_blocked(session_id: str) -> None:
    """Increment the hard-cancel late-write counter without running arbitrary mutations."""
    def _increment(session: Dict[str, Any]) -> None:
        session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1

    await mutate_session(session_id, _increment)


async def pop_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Remove a session safely and return it if present."""
    lock = active_sessions_lock
    if lock is None:
        session = active_sessions.pop(session_id, None)
    else:
        async with lock:
            session = active_sessions.pop(session_id, None)
    if session is not None:
        await cancellation_registry.unregister_session(session_id)
    return session


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


def _is_hard_cancel_sealed(session: Dict[str, Any]) -> bool:
    """Return True when the session is hard-cancelled locally or via registry seal."""
    event = session.get("_hard_cancel_event")
    event_sealed = bool(event is not None and hasattr(event, "is_set") and event.is_set())
    return bool(session.get("hard_cancelled") or event_sealed)


def _store_final_result(
    session: Dict[str, Any],
    final_result: Dict[str, Any],
    session_id: Optional[str] = None,
    final_status: Optional[str] = None,
    *,
    queue_session_end: bool = True,
) -> None:
    """Persist the final_result for a session while attaching project metadata."""
    status_obj = final_status if final_status is not None else final_result.get("status", "")
    final_status_value = status_obj.value if isinstance(status_obj, GenerationStatus) else str(status_obj)
    current_status = session.get("status")
    current_status_value = (
        current_status.value if isinstance(current_status, GenerationStatus) else str(current_status)
    )
    terminal_status_values = {
        GenerationStatus.COMPLETED.value,
        GenerationStatus.REJECTED.value,
        GenerationStatus.FAILED.value,
        GenerationStatus.CANCELLED.value,
    }
    if (
        session.get("final_result") is not None
        and current_status_value in terminal_status_values
        and final_status_value != current_status_value
        and str(final_result.get("status", "")) != current_status_value
    ):
        session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1
        logger.debug(
            "Blocked final_result overwrite after terminal status for session %s: %s -> %s",
            session_id or session.get("session_id"),
            current_status_value,
            final_status_value,
        )
        return
    if (
        _is_hard_cancel_sealed(session)
        and final_status_value != GenerationStatus.CANCELLED.value
        and final_result.get("status") != GenerationStatus.CANCELLED.value
    ):
        session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1
        logger.debug("Blocked final_result overwrite after hard stop for session %s", session_id or session.get("session_id"))
        return
    project_id = session.get("project_id")
    if project_id:
        final_result.setdefault("project_id", project_id)
    session["final_result"] = final_result
    status = final_result.get("status") or final_status or session.get("status")
    session_identifier = session_id or session.get("session_id")  # session_id not stored; caller should pass when available
    if queue_session_end and status and session_identifier:
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


async def start_project_runtime(project_id: str) -> Dict[str, Any]:
    """Activate a project runtime for new work."""
    if not project_id:
        return {"project_id": project_id, "was_cancelled": False, "project_epoch": 0}
    _reserve_project_id(project_id)
    result = await cancellation_registry.clear_project_cancellation(project_id)
    await publish_project_status_event(project_id, "project", "project_started")
    return result


def _build_cancelled_final_result(
    session: Dict[str, Any],
    *,
    reason: str,
) -> Dict[str, Any]:
    raw_content = session.get("last_generated_content") or session.get("generation_content") or ""
    if not raw_content:
        raw_content = "No content generated before cancellation"
    original_request = session.get("request")
    return {
        "content": raw_content,
        "final_iteration": session.get("current_iteration", 0),
        "final_score": 0.0,
        "approved": False,
        "failure_reason": reason,
        "generated_at": datetime.now().isoformat(),
        "project_id": session.get("project_id"),
        "status": GenerationStatus.CANCELLED.value,
        "cancel_mode": session.get("cancel_mode"),
        "request_name": session.get("request_name"),
        "content_type": getattr(original_request, "content_type", None) if original_request else None,
    }


def is_terminal_session(session: Dict[str, Any]) -> bool:
    """Return True when a session has already reached a final status."""
    status = session.get("status")
    status_value = status.value if isinstance(status, GenerationStatus) else str(status)
    return status_value in {
        GenerationStatus.COMPLETED.value,
        GenerationStatus.REJECTED.value,
        GenerationStatus.FAILED.value,
        GenerationStatus.CANCELLED.value,
    }


def apply_session_cancelled_state(
    session: Dict[str, Any],
    session_id: str,
    *,
    cancel_mode: str,
    reason: str,
    hard: bool,
    queue_events: bool = True,
) -> Dict[str, Any]:
    """Force a session into cancelled state and persist a partial final result."""
    session["cancelled"] = True
    session["cancel_mode"] = cancel_mode
    session["cancel_requested_at"] = datetime.now().isoformat()
    if hard:
        session["hard_cancelled"] = True
        session["hard_cancel_reason"] = reason
    final_result = _build_cancelled_final_result(session, reason=reason)
    final_result["cancel_mode"] = cancel_mode
    _store_final_result(
        session,
        final_result,
        session_id,
        final_status=GenerationStatus.CANCELLED.value,
        queue_session_end=queue_events,
    )
    session["status"] = GenerationStatus.CANCELLED
    session["current_phase"] = "cancelled"
    if queue_events:
        queue_project_status_event(session, session_id, "status_change")
    return final_result


async def pause_project_runtime(project_id: str) -> Dict[str, Any]:
    """Cooperatively pause all current sessions in a project and block new admissions."""
    snapshot = await cancellation_registry.request_project_soft_cancel(project_id)
    session_ids = set(snapshot.get("session_ids") or set())
    cancelled_count = 0
    cancelled_debug_updates: List[Tuple[str, Dict[str, Any], str, str]] = []

    def _pause_sessions() -> None:
        nonlocal cancelled_count
        for session_id, session in active_sessions.items():
            if session.get("project_id") != project_id:
                continue
            if session_ids and session_id not in session_ids:
                continue
            if not is_terminal_session(session):
                final_result = apply_session_cancelled_state(
                    session,
                    session_id,
                    cancel_mode=CancelMode.SOFT.value,
                    reason="Project paused by user",
                    hard=False,
                )
                cancelled_debug_updates.append(
                    (
                        session_id,
                        final_result,
                        CancelMode.SOFT.value,
                        "Project paused by user",
                    )
                )
                cancelled_count += 1

    lock = active_sessions_lock
    if lock is None:
        _pause_sessions()
    else:
        async with lock:
            _pause_sessions()

    for session_id, final_result, cancel_mode, reason in cancelled_debug_updates:
        await _debug_record_session_cancelled(
            session_id,
            reason=reason,
            cancel_mode=cancel_mode,
            final_result=final_result,
        )

    await publish_project_status_event(project_id, "project", "project_paused")
    return {
        "project_id": project_id,
        "status": "paused",
        "sessions_cancelled": cancelled_count,
        "mode": CancelMode.SOFT.value,
        "provider_calls_closed": snapshot.get("provider_calls_closed", 0),
    }


async def _hard_stop_project_runtime_impl(project_id: str) -> Dict[str, Any]:
    """Hard-stop all current sessions in a project."""
    snapshot = await cancellation_registry.seal_project_for_hard_cancel(project_id)
    hard_stop_id = snapshot.get("hard_stop_id")
    snapshot_epoch = snapshot.get("project_epoch")
    registry_session_ids = set(snapshot.get("session_ids") or set())

    async def _complete_hard_stop() -> Dict[str, Any]:
        total_tasks_cancelled = 0
        total_provider_calls_closed = 0
        cancelled_count = 0
        cancel_result = {"tasks_cancelled": 0, "provider_calls_closed": 0}
        session_ids: Set[str] = set(registry_session_ids)
        pending_session_ids: Set[str] = set(registry_session_ids)
        cancelled_session_end_events: List[Tuple[str, str, str, Optional[str], Optional[int]]] = []
        cancelled_debug_updates: List[Tuple[str, Dict[str, Any], str, str]] = []
        status_stream_closed = False

        async def _request_cancel_for_sessions(cancel_session_ids: Set[str]) -> None:
            nonlocal total_tasks_cancelled, total_provider_calls_closed
            if not cancel_session_ids:
                return
            ordered_session_ids = sorted(cancel_session_ids)
            retry_results = await asyncio.gather(
                *(
                    cancellation_registry.request_hard_cancel(session_id)
                    for session_id in ordered_session_ids
                ),
                return_exceptions=True,
            )
            for session_id, retry_result in zip(ordered_session_ids, retry_results):
                if isinstance(retry_result, BaseException):
                    logger.debug("Project hard-stop cleanup failed for session %s: %s", session_id, retry_result)
                    continue
                total_tasks_cancelled += int(retry_result.get("tasks_cancelled", 0))
                total_provider_calls_closed += int(retry_result.get("provider_calls_closed", 0))
                pending_session_ids.discard(session_id)

        async def _finalize_hard_stop() -> None:
            nonlocal status_stream_closed
            try:
                await _request_cancel_for_sessions(set(pending_session_ids))
                if not status_stream_closed:
                    try:
                        await publish_project_status_event(
                            project_id,
                            "project",
                            "project_cancelled",
                            expected_project_epoch=snapshot_epoch,
                        )
                        await close_project_status_stream(
                            project_id,
                            "project_cancelled",
                            expected_project_epoch=snapshot_epoch,
                        )
                        status_stream_closed = True
                    except Exception:
                        logger.debug("Unable to close project status stream during hard stop", exc_info=True)
            finally:
                await cancellation_registry.finish_project_hard_cancel(
                    project_id,
                    hard_stop_id=hard_stop_id,
                )

        async def _run_finalizer() -> None:
            finalizer_task = asyncio.create_task(
                _finalize_hard_stop(),
                name=f"project-hard-stop-finalize:{project_id}:{hard_stop_id or 'unknown'}",
            )
            _track_project_hard_stop_task(finalizer_task)
            try:
                await asyncio.shield(finalizer_task)
            except asyncio.CancelledError:
                finalizer_task.add_done_callback(_log_project_hard_stop_task_result)
                raise

        try:
            await _request_cancel_for_sessions(registry_session_ids)

            def _active_project_session_ids() -> Set[str]:
                return {
                    session_id
                    for session_id, session in active_sessions.items()
                    if session.get("project_id") == project_id
                    and not is_terminal_session(session)
                    and session.get("project_epoch") == snapshot_epoch
                }

            lock = active_sessions_lock
            if lock is None:
                active_project_session_ids = _active_project_session_ids()
            else:
                async with lock:
                    active_project_session_ids = _active_project_session_ids()

            session_ids = registry_session_ids | active_project_session_ids
            pending_session_ids.update(session_ids - registry_session_ids)
            await _request_cancel_for_sessions(session_ids - registry_session_ids)

            cancel_result = {
                "tasks_cancelled": total_tasks_cancelled,
                "provider_calls_closed": total_provider_calls_closed,
            }

            late_session_ids: Set[str] = set()

            def _cancel_sessions() -> None:
                nonlocal cancelled_count
                for session_id, session in active_sessions.items():
                    if session.get("project_id") != project_id:
                        continue
                    if session.get("project_epoch") != snapshot_epoch:
                        continue
                    if not is_terminal_session(session):
                        if session_id not in session_ids:
                            session_ids.add(session_id)
                            pending_session_ids.add(session_id)
                            late_session_ids.add(session_id)
                        final_result = apply_session_cancelled_state(
                            session,
                            session_id,
                            cancel_mode=CancelMode.HARD.value,
                            reason="Project hard-stopped by user",
                            hard=True,
                            queue_events=False,
                        )
                        session["tasks_cancelled"] = cancel_result.get("tasks_cancelled", 0)
                        session["provider_calls_closed"] = cancel_result.get("provider_calls_closed", 0)
                        cancelled_session_end_events.append(
                            (
                                project_id,
                                session_id,
                                GenerationStatus.CANCELLED.value,
                                session.get("request_name"),
                                session.get("project_epoch"),
                            )
                        )
                        cancelled_debug_updates.append(
                            (
                                session_id,
                                final_result,
                                CancelMode.HARD.value,
                                "Project hard-stopped by user",
                            )
                        )
                        cancelled_count += 1

            if lock is None:
                _cancel_sessions()
            else:
                async with lock:
                    _cancel_sessions()

            await _request_cancel_for_sessions(late_session_ids)

            for (
                event_project_id,
                event_session_id,
                event_status,
                event_request_name,
                event_project_epoch,
            ) in cancelled_session_end_events:
                await publish_project_session_end(
                    event_project_id,
                    event_session_id,
                    event_status,
                    event_request_name,
                    event_project_epoch,
                )

            for event_session_id, final_result, cancel_mode, reason in cancelled_debug_updates:
                await _debug_record_session_cancelled(
                    event_session_id,
                    reason=reason,
                    cancel_mode=cancel_mode,
                    final_result=final_result,
                )

            phases = ["auto_qa", "preflight", "generation", "qa", "arbiter", "smart_edit", "consensus", "gran_sabio"]
            for phase in phases:
                await publish_project_phase_chunk(
                    project_id,
                    phase,
                    content=None,
                    project_epoch=snapshot.get("project_epoch"),
                    event="project_end",
                    status="cancelled",
                    end_stream=True,
                )
            await publish_project_status_event(
                project_id,
                "project",
                "project_cancelled",
                expected_project_epoch=snapshot.get("project_epoch"),
            )
            await close_project_status_stream(
                project_id,
                "project_cancelled",
                expected_project_epoch=snapshot.get("project_epoch"),
            )
            status_stream_closed = True
        finally:
            await _run_finalizer()

        return {
            "project_id": project_id,
            "status": "cancelled",
            "mode": CancelMode.HARD.value,
            "sessions_cancelled": cancelled_count,
            "tasks_cancelled": cancel_result.get("tasks_cancelled", 0),
            "provider_calls_closed": cancel_result.get("provider_calls_closed", 0),
        }

    return await _complete_hard_stop()


async def hard_stop_project_runtime(project_id: str) -> Dict[str, Any]:
    """Hard-stop all current sessions in a project."""
    completion_task = asyncio.create_task(
        _hard_stop_project_runtime_impl(project_id),
        name=f"project-hard-stop:{project_id}",
    )
    _track_project_hard_stop_task(completion_task)
    try:
        return await asyncio.shield(completion_task)
    except asyncio.CancelledError:
        completion_task.add_done_callback(_log_project_hard_stop_task_result)
        raise


async def is_project_cancelled_or_stopping(project_id: str) -> bool:
    """Return whether project admission should be rejected."""
    if not project_id:
        return False
    return await cancellation_registry.is_project_cancelled_or_stopping(project_id)


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


# Maximum byte size for the ``data`` field carried by project/phase events.
# Events exceeding this size are degraded to a preview to avoid blowing up
# SSE subscribers or the queue. Keep in sync with the docstring of
# ``publish_project_phase_chunk``.
_PROJECT_PHASE_DATA_MAX_BYTES = 8192


def _build_project_phase_event(
    *,
    event: str,
    phase: str,
    subphase: Optional[str] = None,
    session_id: Optional[str] = None,
    project_epoch: Optional[int] = None,
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
    # Free-form structured telemetry payload (e.g. tool-loop turn details).
    data: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a JSON event string for project/phase streams.

    The ``data`` field carries free-form structured telemetry (e.g. tool-loop
    turn events from QA / Arbiter / GranSabio). If the serialized payload
    exceeds ``_PROJECT_PHASE_DATA_MAX_BYTES``, the field is replaced by a
    degraded ``{"truncated": True, "preview_keys": [...], "size_bytes": N}``
    summary and a warning is logged. The full payload is still available via
    the debugger DB (when ``debug_event_callback`` is bound at the caller).
    """
    obj: Dict[str, Any] = {
        "type": event,
        "phase": phase,
        "timestamp": int(time.time() * 1000),
    }
    if subphase:
        obj["subphase"] = subphase
    if session_id:
        obj["session_id"] = session_id
    if project_epoch is not None:
        obj["project_epoch"] = project_epoch
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
    # Free-form telemetry data (size-guarded).
    if data is not None:
        try:
            serialized = json.dumps(data, ensure_ascii=True)
            size_bytes = len(serialized)
        except Exception:
            # If the payload cannot be serialized at all, degrade to a neutral
            # stub so telemetry never crashes the publisher.
            logger.warning(
                "publish_project_phase_chunk: data payload not JSON-serializable "
                "for phase=%s event=%s; degrading to stub",
                phase,
                event,
            )
            obj["data"] = {
                "truncated": True,
                "preview_keys": [],
                "size_bytes": 0,
            }
        else:
            if size_bytes <= _PROJECT_PHASE_DATA_MAX_BYTES:
                obj["data"] = data
            else:
                try:
                    preview_keys = list(data.keys())[:10]
                except AttributeError:
                    preview_keys = []
                logger.warning(
                    "publish_project_phase_chunk: data payload size %d bytes "
                    "exceeds %d limit for phase=%s event=%s; emitting preview",
                    size_bytes,
                    _PROJECT_PHASE_DATA_MAX_BYTES,
                    phase,
                    event,
                )
                obj["data"] = {
                    "truncated": True,
                    "preview_keys": preview_keys,
                    "size_bytes": size_bytes,
                }
    return json.dumps(obj, ensure_ascii=True)


async def publish_project_phase_chunk(
    project_id: Optional[str],
    phase: str,
    content: Optional[str],
    *,
    subphase: Optional[str] = None,
    session_id: Optional[str] = None,
    project_epoch: Optional[int] = None,
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
    # Free-form structured telemetry payload (size-guarded).
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Broadcast a live event to all subscribers of a project/phase stream.

    - If end_stream=True, a sentinel is sent after the event to close the stream.
    - If the queue is full, the event is dropped to avoid blocking producers.
    - Events with ``data`` > 8KB are truncated to a summary preview; the full
      payload is available via the debugger DB (when ``debug_event_callback``
      is bound at the caller site).
    """
    if not project_id:
        return

    terminal_event = event in {"session_end", "project_end", "stream_end"}
    try:
        project_state = await cancellation_registry.get_project_state(project_id)
        effective_project_epoch = project_epoch
        if effective_project_epoch is None and session_id:
            identity = await cancellation_registry.get_session_project_identity(session_id)
            if identity.get("project_id") and identity.get("project_id") != project_id:
                return
            effective_project_epoch = identity.get("project_epoch")
        if effective_project_epoch is None and session_id:
            session = active_sessions.get(session_id)
            if session and session.get("project_id") == project_id:
                effective_project_epoch = session.get("project_epoch")
        if effective_project_epoch is None:
            return
        if project_state.epoch != effective_project_epoch:
            return
        if not terminal_event:
            if session_id and await cancellation_registry.is_cancelled(session_id):
                return
            if project_state.cancelled or project_state.hard_stop_in_progress:
                return
    except Exception:
        logger.debug("Unable to evaluate project/session publish barrier", exc_info=True)
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
        subphase=subphase,
        session_id=session_id,
        project_epoch=effective_project_epoch,
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
        data=data,
    )

    for q in targets:
        if end_stream:
            event_enqueued = _put_project_phase_queue_item(q, payload)
            sentinel_enqueued = _put_project_phase_queue_item(q, None)
            if not event_enqueued or not sentinel_enqueued:
                logger.debug("Unable to enqueue project-phase close event for project %s phase %s", project_id, phase)
            continue
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug("Dropping project-phase event (queue full) for project %s phase %s", project_id, phase)


def _put_project_phase_queue_item(q: asyncio.Queue[str], item: Optional[str]) -> bool:
    """Best-effort queue put that preserves terminal sentinels under backpressure."""
    try:
        q.put_nowait(item)  # type: ignore[arg-type]
        return True
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            return False
        try:
            q.put_nowait(item)  # type: ignore[arg-type]
            return True
        except asyncio.QueueFull:
            return False


def _put_project_status_queue_item(q: asyncio.Queue[str], item: Optional[str]) -> bool:
    """Best-effort status queue put that preserves terminal stream_end sentinels."""
    return _put_project_phase_queue_item(q, item)


async def publish_project_session_end(
    project_id: Optional[str],
    session_id: str,
    status: str,
    request_name: Optional[str] = None,
    project_epoch: Optional[int] = None,
) -> None:
    """Publish a session_end event to all phases for a project."""
    if not project_id:
        return
    phases = ["auto_qa", "preflight", "generation", "qa", "arbiter", "smart_edit", "consensus", "gran_sabio"]
    for phase in phases:
        await publish_project_phase_chunk(
            project_id,
            phase,
            content=None,
            session_id=session_id,
            project_epoch=project_epoch,
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
    project_epoch = session.get("project_epoch")
    try:
        task = asyncio.create_task(
            publish_project_session_end(
                project_id,
                session_id,
                status,
                request_name,
                project_epoch,
            ),
            name=f"project-session-end:{project_id}:{session_id}",
        )
        _track_background_task(task, _project_session_end_tasks)
        task.add_done_callback(_log_background_task_result)
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
    expected_project_epoch: Optional[int] = None,
) -> None:
    """Broadcast a project status update to all subscribers."""
    if not project_id:
        return

    if expected_project_epoch is not None:
        project_state = await cancellation_registry.get_project_state(project_id)
        if project_state.epoch != expected_project_epoch:
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


async def close_project_status_stream(
    project_id: str,
    reason: str = "project_ended",
    expected_project_epoch: Optional[int] = None,
) -> None:
    """Send stream_end and close all project status subscribers."""
    if not project_id:
        return

    if expected_project_epoch is not None:
        project_state = await cancellation_registry.get_project_state(project_id)
        if project_state.epoch != expected_project_epoch:
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
        final_enqueued = _put_project_status_queue_item(q, final_str)
        sentinel_enqueued = _put_project_status_queue_item(q, None)
        if not final_enqueued or not sentinel_enqueued:
            logger.debug("Unable to close status stream queue for project %s", project_id)


def queue_project_status_event(session: Dict[str, Any], session_id: str, event_type: str = "status_change") -> None:
    """Schedule a project-level status update without blocking."""
    project_id = session.get("project_id")
    if not project_id:
        return
    project_epoch = session.get("project_epoch")
    try:
        asyncio.create_task(
            publish_project_status_event(
                project_id,
                session_id,
                event_type,
                expected_project_epoch=project_epoch,
            )
        )
    except RuntimeError:
        logger.debug("Unable to schedule project status event (no running loop)")


def update_session_status(
    session: Dict[str, Any],
    session_id: str,
    status: "GenerationStatus",
    phase: Optional[str] = None,
) -> None:
    """Set session status (and optional phase) and trigger project status hook."""
    final_status_values = {
        GenerationStatus.COMPLETED.value,
        GenerationStatus.REJECTED.value,
        GenerationStatus.FAILED.value,
        GenerationStatus.CANCELLED.value,
    }
    current_status = session.get("status")
    if isinstance(current_status, GenerationStatus):
        current_value = current_status.value
    elif current_status is None:
        current_value = None
    else:
        current_value = str(current_status)
    next_value = status.value if isinstance(status, GenerationStatus) else str(status)
    if _is_hard_cancel_sealed(session) and next_value != GenerationStatus.CANCELLED.value:
        session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1
        logger.debug(
            "Ignoring status transition after hard stop for session %s: %s -> %s",
            session_id,
            current_value,
            next_value,
        )
        return
    if current_value in final_status_values and next_value != current_value:
        logger.debug(
            "Ignoring status transition for terminal session %s: %s -> %s",
            session_id,
            current_value,
            next_value,
        )
        return
    session["status"] = status
    if phase is not None:
        session["current_phase"] = phase
    queue_project_status_event(session, session_id)
    queue_debug_session_transition(
        session_id,
        status=next_value,
        phase=session.get("current_phase"),
    )


def update_session_phase(session: Dict[str, Any], session_id: str, phase: str) -> None:
    """Set current phase and trigger project status hook."""
    session["current_phase"] = phase
    queue_project_status_event(session, session_id)
    queue_debug_session_transition(
        session_id,
        status=_status_to_debug_value(session.get("status")),
        phase=phase,
    )


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


def cleanup_session_sidecars(session_id: str) -> None:
    """Release per-session auxiliary state after the session is removed."""
    try:
        get_tracker().clear_session(session_id)
    except Exception as exc:
        logger.warning("Failed to clear deal-breaker tracker state for session %s: %s", session_id, exc)

    try:
        if qa_engine is not None:
            clear_state = getattr(qa_engine, "clear_session_state", None)
            if callable(clear_state):
                clear_state(session_id)
            else:
                getattr(qa_engine, "_qa_failure_tracker", {}).pop(session_id, None)
    except Exception as exc:
        logger.warning("Failed to clear QA runtime state for session %s: %s", session_id, exc)


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
                    GenerationStatus.REJECTED,
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
            cleanup_session_sidecars(session_id)
    else:
        async with lock:
            _cleanup_sessions(list(active_sessions.items()))
            for session_id in expired_sessions:
                active_sessions.pop(session_id, None)
                cleanup_session_sidecars(session_id)

    for session_id in expired_sessions:
        await cancellation_registry.unregister_session(session_id)

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
        "status": "idle" | "running" | "completed" | "rejected" | "failed" | "cancelled",
        "sessions": [...],
        "summary": { "total_sessions": int, "active_sessions": int, "completed_sessions": int }
    }
    """
    from word_count_utils import count_words  # Avoid circular import

    project_state = await cancellation_registry.get_project_state(project_id)
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
            elif (
                status_str in ("generating", "qa_evaluation", "consensus", "gran_sabio_review", "running", "initializing")
                or phase in ("inline_deal_breaker_review", "gran_sabio_review", "gran_sabio_regeneration")
            ):
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
            gran_sabio_active = (
                status_str == "gran_sabio_review"
                or phase in ("inline_deal_breaker_review", "gran_sabio_review", "gran_sabio_regeneration")
            )
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
    if project_state.hard_stop_in_progress:
        project_status = "cancelling"
    elif project_state.cancelled:
        project_status = "cancelled"
    elif total_sessions == 0:
        # Check if project is reserved but idle
        if project_id in reserved_project_ids:
            project_status = "idle"
        else:
            project_status = "idle"
    elif active_count > 0:
        project_status = "running"
    elif last_status == "completed":
        project_status = "completed"
    elif last_status == "rejected":
        project_status = "rejected"
    elif last_status in ("failed", "error"):
        project_status = "failed"
    else:
        project_status = "idle"

    return {
        "project_id": project_id,
        "status": project_status,
        "cancel_mode": project_state.cancel_mode,
        "project_epoch": project_state.epoch,
        "sessions": sessions_data,
        "summary": {
            "total_sessions": total_sessions,
            "active_sessions": active_count,
            "completed_sessions": completed_count,
        },
    }
