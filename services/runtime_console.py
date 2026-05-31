"""Runtime console capture and streaming utilities.

This module keeps a bounded in-memory tail of process console output and lets
internal clients subscribe to new output. It is intentionally independent from
the project phase stream so raw stdout/stderr remains observability data, not
generation content.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, Optional, TextIO


_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "runtime_console_session_id",
    default=None,
)
_project_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "runtime_console_project_id",
    default=None,
)
_phase_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "runtime_console_phase",
    default=None,
)

DEFAULT_MAX_EVENTS = 5000
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_QUEUE_SIZE = 1000
HEARTBEAT_INTERVAL_SECONDS = 15

_events: Deque[Dict[str, Any]] = deque()
_events_bytes = 0
_next_seq = 1
_subscribers: list["ConsoleSubscription"] = []
_lock = threading.RLock()
_capture_active = False


@dataclass(frozen=True)
class ConsoleContextTokens:
    """ContextVar tokens used to restore runtime console context."""

    session_id: contextvars.Token[Optional[str]]
    project_id: contextvars.Token[Optional[str]]
    phase: contextvars.Token[Optional[str]]


@dataclass
class ConsoleSubscription:
    """A live console subscriber bound to the event loop that created it."""

    queue: asyncio.Queue[Dict[str, Any]]
    loop: asyncio.AbstractEventLoop
    filters: Dict[str, Optional[str]]


class ConsoleCaptureOutput:
    """File-like wrapper that mirrors writes and publishes them to the console bus."""

    def __init__(self, stream: TextIO, stream_name: str):
        self.stream = stream
        self.stream_name = stream_name

    def write(self, message: str) -> int:
        written = self.stream.write(message)
        if message:
            publish_console_output(
                self.stream_name,
                message,
                source="stream",
            )
        return written

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if callable(close):
            close()

    def fileno(self) -> int:
        return self.stream.fileno()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self.stream, "encoding", None)

    @property
    def errors(self) -> Optional[str]:
        return getattr(self.stream, "errors", None)

    def writable(self) -> bool:
        writable = getattr(self.stream, "writable", None)
        return bool(writable()) if callable(writable) else True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


class RuntimeConsoleLoggingHandler(logging.Handler):
    """Logging handler that publishes formatted records without writing them."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record) + "\n"
            publish_console_output(
                "stderr",
                message,
                source="logging",
                level=record.levelname,
                logger_name=record.name,
            )
        except Exception:
            # Never let observability break application logging.
            return


def bind_console_context(
    *,
    session_id: Optional[str] = None,
    project_id: Optional[str] = None,
    phase: Optional[str] = None,
) -> ConsoleContextTokens:
    """Bind console context for the current task/thread."""

    return ConsoleContextTokens(
        session_id=_session_id_var.set(session_id),
        project_id=_project_id_var.set(project_id),
        phase=_phase_var.set(phase),
    )


def reset_console_context(tokens: ConsoleContextTokens) -> None:
    """Restore a previously bound console context."""

    _session_id_var.reset(tokens.session_id)
    _project_id_var.reset(tokens.project_id)
    _phase_var.reset(tokens.phase)


def set_console_phase(phase: Optional[str]) -> None:
    """Update the current task's phase context."""

    _phase_var.set(phase)


def current_console_context() -> Dict[str, Optional[str]]:
    """Return the context attached to newly captured console output."""

    return {
        "session_id": _session_id_var.get(),
        "project_id": _project_id_var.get(),
        "phase": _phase_var.get(),
    }


def activate_runtime_console_capture() -> None:
    """Capture stdout/stderr and logging records for runtime streaming."""

    global _capture_active
    if _capture_active:
        return

    if not isinstance(sys.stdout, ConsoleCaptureOutput):
        sys.stdout = ConsoleCaptureOutput(sys.stdout, "stdout")  # type: ignore[assignment]
    if not isinstance(sys.stderr, ConsoleCaptureOutput):
        sys.stderr = ConsoleCaptureOutput(sys.stderr, "stderr")  # type: ignore[assignment]

    root_logger = logging.getLogger()
    if not any(isinstance(handler, RuntimeConsoleLoggingHandler) for handler in root_logger.handlers):
        root_logger.addHandler(RuntimeConsoleLoggingHandler())

    _capture_active = True


def _event_size(event: Dict[str, Any]) -> int:
    text = str(event.get("text") or "")
    return len(text.encode("utf-8", errors="replace"))


def _trim_events_locked(max_events: int = DEFAULT_MAX_EVENTS, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
    global _events_bytes
    while len(_events) > max_events:
        removed = _events.popleft()
        _events_bytes -= _event_size(removed)
    while _events and _events_bytes > max_bytes:
        removed = _events.popleft()
        _events_bytes -= _event_size(removed)


def _matches_filters(event: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
    context = event.get("context") or {}
    for key in ("project_id", "session_id", "phase"):
        expected = filters.get(key)
        if expected and context.get(key) != expected:
            return False
    expected_stream = filters.get("stream")
    if expected_stream and event.get("stream") != expected_stream:
        return False
    expected_level = filters.get("level")
    if expected_level and str(event.get("level") or "").upper() != expected_level.upper():
        return False
    return True


def _enqueue_subscription_event(subscription: ConsoleSubscription, event: Dict[str, Any]) -> None:
    def _put() -> None:
        try:
            subscription.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                subscription.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                subscription.queue.put_nowait(event)
            except asyncio.QueueFull:
                return

    try:
        subscription.loop.call_soon_threadsafe(_put)
    except RuntimeError:
        return


def publish_console_output(
    stream: str,
    text: str,
    *,
    source: str = "stream",
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish a stdout/stderr/logging fragment to the in-memory console tail."""

    global _events_bytes, _next_seq
    timestamp_ms = int(time.time() * 1000)
    event = {
        "type": "console_output",
        "seq": 0,
        "timestamp": timestamp_ms,
        "created_at": datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat(),
        "stream": stream,
        "source": source,
        "text": text,
        "context": current_console_context(),
    }
    if level:
        event["level"] = level
    if logger_name:
        event["logger"] = logger_name

    with _lock:
        event["seq"] = _next_seq
        _next_seq += 1
        _events.append(event)
        _events_bytes += _event_size(event)
        _trim_events_locked()
        subscribers = list(_subscribers)

    for subscriber in subscribers:
        if _matches_filters(event, subscriber.filters):
            _enqueue_subscription_event(subscriber, event)

    return event


def get_recent_console_events(
    *,
    limit: int = 200,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    phase: Optional[str] = None,
    stream: Optional[str] = None,
    level: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Return a bounded, filtered copy of recent console events."""

    limit = max(1, min(limit, DEFAULT_MAX_EVENTS))
    filters = {
        "project_id": project_id,
        "session_id": session_id,
        "phase": phase,
        "stream": stream,
        "level": level,
    }
    with _lock:
        matched = [event.copy() for event in _events if _matches_filters(event, filters)]
    return matched[-limit:]


async def subscribe_console(
    *,
    project_id: Optional[str] = None,
    session_id: Optional[str] = None,
    phase: Optional[str] = None,
    stream: Optional[str] = None,
    level: Optional[str] = None,
) -> ConsoleSubscription:
    """Subscribe to new console events."""

    subscription = ConsoleSubscription(
        queue=asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE),
        loop=asyncio.get_running_loop(),
        filters={
            "project_id": project_id,
            "session_id": session_id,
            "phase": phase,
            "stream": stream,
            "level": level,
        },
    )
    with _lock:
        _subscribers.append(subscription)
    return subscription


async def unsubscribe_console(subscription: ConsoleSubscription) -> None:
    """Remove a console subscription."""

    with _lock:
        if subscription in _subscribers:
            _subscribers.remove(subscription)


def console_stats() -> Dict[str, Any]:
    """Return diagnostic stats for the runtime console bus."""

    with _lock:
        return {
            "events": len(_events),
            "bytes": _events_bytes,
            "subscribers": len(_subscribers),
            "next_seq": _next_seq,
            "capture_active": _capture_active,
        }


def _reset_runtime_console_for_tests() -> None:
    """Clear in-memory runtime console state for focused tests."""

    global _events_bytes, _next_seq
    with _lock:
        _events.clear()
        _events_bytes = 0
        _next_seq = 1
        _subscribers.clear()
