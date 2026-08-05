"""Incremental SSE parsing and unified-event routing for the GranSabio SDK.

Both clients (sync and async) consume the same Server-Sent Events framing and
the same unified event contract, so both use this module. It contains no HTTP
code on purpose: it is fed decoded bytes or text and it returns events, which
makes it testable offline against split and merged frames.

Two pieces:

- :class:`SSEParser` - incremental, line based, spec shaped (``data:``,
  ``event:``, ``id:``, ``retry:``, ``:comment``). Frames arrive split across
  transport chunks and several frames arrive merged in one chunk; both are
  handled, including a ``\\r\\n`` terminator split between two chunks.
- :class:`SessionEventRouter` - maps the engine's unified events onto the SDK
  callbacks. Routing is mechanical (by ``type``/``phase``/``content_status``
  fields), never by guessing meaning from message text.

Callback contract implemented here:

``progress_callback``
    Lifecycle only: connection, status snapshots/changes, attempt changes,
    control events, session end. It is NEVER called with an empty string.
``content_callback``
    Generation text of the session (``phase="generation"``). A ``draft``
    emission is a DELTA to append. Any other ``content_status`` is a COMMIT
    that carries the FULL body of its attempt (C0.7), so it REPLACES whatever
    was delivered before it instead of extending it.
``content_reset``
    Optional. Called when what was already delivered for the current phase is
    superseded: immediately before an accepted commit that replaces its own
    drafts or a chunk of a different attempt, and on a body-less verdict that
    closes the attempt with nothing to put in its place (a ``filtered`` terminal
    after a provider policy stop, a commit with an empty body). A consumer that
    provides it must drop its buffer and keep only what arrives next. A consumer
    that omits it keeps the historical append-everything behaviour, which glues
    the dead draft to the snapshot that replaced it (and, with ``json_output``,
    yields unparseable text).
``qa_callback``
    ``(text, agent, status)`` for every other text channel (QA, preflight,
    Arbiter, Gran Sabio, auto-QA). ``agent`` is the producing model when the
    engine reports one, otherwise the phase; ``status`` is the content label
    (``draft``/``accepted``/``rejected``/``filtered``, or ``thinking``).
"""

from __future__ import annotations

import codecs
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "DRAFT_CONTENT_STATUS",
    "QA_TEXT_PHASES",
    "TERMINAL_SESSION_STATUSES",
    "SSEEvent",
    "SSEParser",
    "SessionEventRouter",
    "StreamCallbacks",
    "StreamError",
    "decode_event_json",
]

# Session lifecycle values that end a streaming run.
TERMINAL_SESSION_STATUSES = frozenset(
    {"completed", "rejected", "failed", "error", "cancelled"}
)

# Phase whose text IS the requested content.
CONTENT_PHASE = "generation"

# The only content label that means "delta to append"; every other label closes
# the attempt and republishes its whole body (C0.7). Mirrors
# ``content_events.ContentStatus.DRAFT`` without importing the engine: the SDK
# is stdlib only so it can be vendored next to a caller.
DRAFT_CONTENT_STATUS = "draft"

# Phases whose text is engine work about the content, not the content itself.
QA_TEXT_PHASES = frozenset(
    {"qa", "auto_qa", "preflight", "arbiter", "gran_sabio", "smart_edit", "consensus"}
)


# ---------------------------------------------------------------------------
# Incremental SSE parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SSEEvent:
    """One dispatched Server-Sent Event (or one comment line).

    Attributes:
        data: concatenated ``data:`` lines (joined with newlines).
        event: ``event:`` name when the server sent one.
        event_id: ``id:`` value when the server sent one.
        retry: ``retry:`` reconnection hint in milliseconds.
        comment: text of a ``:comment`` line (heartbeats arrive this way).
    """

    data: str = ""
    event: Optional[str] = None
    event_id: Optional[str] = None
    retry: Optional[int] = None
    comment: Optional[str] = None

    @property
    def is_comment(self) -> bool:
        """Return whether this is a comment line (e.g. a heartbeat)."""
        return self.comment is not None


def decode_event_json(event: SSEEvent) -> Optional[Dict[str, Any]]:
    """Return the JSON object carried by an event, or ``None``.

    A payload that is not a JSON object is reported at warning level: the
    engine only sends JSON objects on these streams, so anything else means
    the contract broke and must be visible instead of silently skipped.
    """
    if event.is_comment or not event.data:
        return None
    try:
        payload = json.loads(event.data)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Discarding non-JSON SSE payload: %s", event.data[:120])
        return None
    if not isinstance(payload, dict):
        logger.warning("Discarding non-object SSE payload: %s", event.data[:120])
        return None
    return payload


class SSEParser:
    """Incremental Server-Sent Events parser.

    Feed it whatever the transport hands over (``bytes`` or ``str``, any
    chunking) and it returns the events that became complete with that chunk.
    Bytes are decoded with an incremental UTF-8 decoder, so a multi-byte
    character split across two chunks is not corrupted.

    Usage::

        parser = SSEParser()
        for chunk in transport:
            for event in parser.feed(chunk):
                ...
        for event in parser.close():
            ...
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""
        self._data: List[str] = []
        self._event_name: Optional[str] = None
        self._event_id: Optional[str] = None
        self._retry: Optional[int] = None

    def feed(self, chunk: Union[str, bytes, bytearray, None]) -> List[SSEEvent]:
        """Consume one transport chunk and return the events it completed."""
        if chunk is None:
            return []
        if isinstance(chunk, (bytes, bytearray)):
            text = self._decoder.decode(bytes(chunk))
        else:
            text = chunk
        if not text:
            return []

        self._buffer += text
        events: List[SSEEvent] = []
        while True:
            line, terminated, rest = _split_line(self._buffer)
            if not terminated:
                break
            self._buffer = rest
            event = self._handle_line(line)
            if event is not None:
                events.append(event)
        return events

    def close(self) -> List[SSEEvent]:
        """Flush the decoder and return an event left without its blank line.

        A partially received LINE is discarded (half an event is worse than no
        event); a complete event whose trailing blank line never arrived
        because the connection ended is still delivered.
        """
        tail = self._decoder.decode(b"", True)
        if tail:
            self._buffer += tail
        events: List[SSEEvent] = []
        while True:
            line, terminated, rest = _split_line(self._buffer)
            if not terminated:
                break
            self._buffer = rest
            event = self._handle_line(line)
            if event is not None:
                events.append(event)
        if self._buffer:
            logger.debug("Discarding incomplete SSE line at end of stream")
            self._buffer = ""
        pending = self._dispatch()
        if pending is not None:
            events.append(pending)
        return events

    def _handle_line(self, line: str) -> Optional[SSEEvent]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return SSEEvent(comment=line[1:].strip())

        field, separator, value = line.partition(":")
        if not separator:
            field, value = line, ""
        elif value.startswith(" "):
            value = value[1:]

        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event_name = value
        elif field == "id":
            self._event_id = value
        elif field == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                logger.debug("Ignoring non-numeric SSE retry field: %s", value[:32])
        return None

    def _dispatch(self) -> Optional[SSEEvent]:
        if not self._data and self._event_name is None:
            return None
        event = SSEEvent(
            data="\n".join(self._data),
            event=self._event_name,
            event_id=self._event_id,
            retry=self._retry,
        )
        self._data = []
        self._event_name = None
        return event


def _split_line(buffer: str) -> Tuple[str, bool, str]:
    """Split the first complete SSE line off ``buffer``.

    Returns ``(line, terminated, rest)``. A trailing lone ``\\r`` is treated as
    incomplete: the ``\\n`` that would complete a CRLF may arrive in the next
    chunk.
    """
    index_n = buffer.find("\n")
    index_r = buffer.find("\r")
    if index_n < 0 and index_r < 0:
        return "", False, buffer
    if index_r < 0 or (0 <= index_n < index_r):
        return buffer[:index_n], True, buffer[index_n + 1:]
    if index_r == len(buffer) - 1:
        return "", False, buffer
    if buffer[index_r + 1] == "\n":
        return buffer[:index_r], True, buffer[index_r + 2:]
    return buffer[:index_r], True, buffer[index_r + 1:]


# ---------------------------------------------------------------------------
# Unified event routing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamError:
    """A streaming failure surfaced to the caller.

    Streams are the visibility channel, not the result channel: a failure here
    never aborts a generation the engine is still running (the result is
    fetched by polling). It is logged at warning level, appended to the
    client's ``last_stream_errors`` and reported to ``error_callback``.
    """

    stream: str
    url: str
    error: str
    error_type: str

    def as_dict(self) -> Dict[str, str]:
        """Return the plain dict form exposed in client state."""
        return {
            "stream": self.stream,
            "url": self.url,
            "error": self.error,
            "error_type": self.error_type,
        }


@dataclass
class StreamCallbacks:
    """Callbacks a streaming run may drive.

    ``content_reset`` is optional and takes no argument: it announces that the
    content delivered so far for the current phase is dead, whether a
    replacement follows it or the attempt simply closed with nothing. Leaving it
    unset keeps the append-only behaviour described in the module docstring.
    """

    progress: Optional[Callable[[str], None]] = None
    content: Optional[Callable[[str], None]] = None
    qa: Optional[Callable[[str, str, str], None]] = None
    content_reset: Optional[Callable[[], None]] = None


class SessionEventRouter:
    """Route unified session-stream events onto the SDK callbacks.

    The router is transport agnostic and stateful only in what it must
    remember to report honestly: the generation attempt in flight (so a
    restarted draft is announced instead of silently concatenated), the
    ``(phase, attempt)`` whose content was already handed to the consumer (so a
    commit or a new attempt is announced as a REPLACEMENT, not appended) and
    whether a terminal status was already seen.

    Args:
        callbacks: the caller's callbacks.
        activity: optional ``ActivityMonitor`` marked on every event so the
            streaming deadline follows real server activity.
    """

    def __init__(
        self,
        callbacks: StreamCallbacks,
        activity: Optional[Any] = None,
    ) -> None:
        self._callbacks = callbacks
        self._activity = activity
        self._generation_attempt: Optional[int] = None
        self._content_delivered: Optional[Tuple[str, Optional[int]]] = None
        self.terminal_status: Optional[str] = None

    def handle(self, event: Dict[str, Any]) -> Optional[str]:
        """Dispatch one event; return the terminal status when the run ended."""
        event_type = str(event.get("type") or "")
        self._mark_activity(event, event_type)

        if event_type == "chunk":
            self._route_chunk(event)
            return None
        if event_type in ("status_snapshot", "status_change"):
            self._progress(self._render_status(event))
            return self._terminal_from(event.get("status"))
        if event_type == "session_end":
            status = str(event.get("status") or "session_end")
            self._progress(f"session_end | status={status}")
            return self._terminal_from(status) or status
        if event_type in ("stream_end", "project_end", "project_cancelled"):
            reason = event.get("reason") or event_type
            self._progress(f"{event_type} | reason={reason}")
            return None
        if event_type == "connected":
            self._progress(
                "connected | session={} | visibility={}".format(
                    event.get("session_id") or "?",
                    event.get("visibility") or "full",
                )
            )
            return None

        self._progress(self._render_control(event, event_type))
        return None

    # -- routing ------------------------------------------------------------

    def _route_chunk(self, event: Dict[str, Any]) -> None:
        phase = str(event.get("phase") or "")
        content = event.get("content") or ""

        if phase == CONTENT_PHASE:
            self._announce_attempt(event)

        if not content:
            # A chunk without text is a label-only event (filtered marker,
            # commit with empty body): it belongs to progress, never to the
            # content callback as an empty string. It still carries a verdict,
            # and a verdict that closes an attempt whose text was already
            # delivered leaves that text dead: the consumer is told to drop it
            # before the label reaches progress.
            if (
                phase == CONTENT_PHASE
                and not event.get("is_thinking")
                and self._closes_delivered_content(event)
            ):
                self._reset_content()
            self._progress(self._render_control(event, "chunk"))
            return

        if phase == CONTENT_PHASE and not event.get("is_thinking"):
            self._deliver_content(event, phase, content)
            return

        if self._callbacks.qa:
            agent = str(
                event.get("qa_model")
                or event.get("model")
                or (phase or "engine")
            )
            status = "thinking" if event.get("is_thinking") else str(
                event.get("content_status") or "chunk"
            )
            self._callbacks.qa(content, agent, status)

    def _deliver_content(
        self,
        event: Dict[str, Any],
        phase: str,
        content: str,
    ) -> None:
        """Hand generation text to the consumer, flagging replacements.

        Three situations make everything delivered so far for this phase dead
        text, and all three are the same instruction to the consumer:

        - a commit (any ``content_status`` other than ``draft``) republishes the
          whole body of its attempt, so the drafts it supersedes must go;
        - a chunk of a different attempt means the previous attempt was
          discarded by the engine;
        - an explicit ``reset`` flag, which the per-field endpoints set when a
          shorter attempt replaces a longer one.
        """
        if self._callbacks.content is None:
            return

        attempt = event.get("attempt_id")
        key = (phase, attempt if isinstance(attempt, int) else None)
        status = str(event.get("content_status") or DRAFT_CONTENT_STATUS)
        supersedes = (
            status != DRAFT_CONTENT_STATUS
            or bool(event.get("reset"))
            or key != self._content_delivered
        )
        if self._content_delivered is not None and supersedes:
            self._reset_content()

        self._content_delivered = key
        self._callbacks.content(content)

    def _closes_delivered_content(self, event: Dict[str, Any]) -> bool:
        """Return whether a body-less verdict kills the text already delivered.

        A ``draft`` chunk with no body supersedes nothing: it is telemetry. Any
        other label CLOSES the attempt (``filtered`` after a provider policy
        stop, ``rejected``, an ``accepted`` commit with an empty body), and an
        empty body cannot replace the text the consumer holds - the only honest
        instruction left is "discard it".

        Scoped to the attempt whose content was delivered, so a later attempt's
        verdict never wipes a body that had been committed. A verdict that
        carries no ``attempt_id`` can only refer to what is in flight.
        """
        if self._content_delivered is None:
            return False
        status = str(event.get("content_status") or DRAFT_CONTENT_STATUS)
        if status == DRAFT_CONTENT_STATUS:
            return False
        attempt = event.get("attempt_id")
        if not isinstance(attempt, int):
            return True
        return self._content_delivered == (CONTENT_PHASE, attempt)

    def _reset_content(self) -> None:
        """Announce that the content delivered so far must be discarded."""
        self._content_delivered = None
        if self._callbacks.content_reset:
            self._callbacks.content_reset()

    def _announce_attempt(self, event: Dict[str, Any]) -> None:
        """Report a restarted generation draft instead of concatenating it."""
        attempt = event.get("attempt_id")
        if event.get("reset") and self._generation_attempt is not None:
            self._progress("generation restarted | previous draft discarded")
            return
        if not isinstance(attempt, int):
            return
        if self._generation_attempt is None:
            self._generation_attempt = attempt
            return
        if attempt != self._generation_attempt:
            self._generation_attempt = attempt
            self._progress(
                f"generation attempt {attempt} started | previous draft discarded"
            )

    def _terminal_from(self, status: Any) -> Optional[str]:
        if not isinstance(status, str):
            return None
        normalized = status.strip().lower()
        if normalized in TERMINAL_SESSION_STATUSES:
            self.terminal_status = normalized
            return normalized
        return None

    def _progress(self, message: str) -> None:
        if not message or not self._callbacks.progress:
            return
        self._callbacks.progress(message)

    def _mark_activity(self, event: Dict[str, Any], event_type: str) -> None:
        if self._activity is None:
            return
        iteration = event.get("iteration")
        self._activity.mark(
            iteration=iteration if isinstance(iteration, int) else None,
            message=event_type or None,
        )

    # -- rendering ----------------------------------------------------------

    @staticmethod
    def _render_status(event: Dict[str, Any]) -> str:
        """Render a status event as a compact factual line."""
        session = event.get("session") or {}
        parts: List[str] = [
            "status={}".format(
                event.get("status") or session.get("status") or "unknown"
            )
        ]
        phase = event.get("current_phase") or session.get("phase")
        if phase:
            parts.append(f"phase={phase}")
        iteration = event.get("iteration")
        if isinstance(iteration, int) and iteration:
            parts.append(f"iteration={iteration}/{event.get('max_iterations') or '?'}")
        qa_progress = (session.get("qa") or {}).get("progress") or {}
        total = qa_progress.get("total")
        if total:
            parts.append(f"qa={qa_progress.get('completed') or 0}/{total}")
        score = (session.get("consensus") or {}).get("last_score")
        if score is not None:
            parts.append(f"score={score}")
        return " | ".join(parts)

    @staticmethod
    def _render_control(event: Dict[str, Any], event_type: str) -> str:
        """Render a control/label event as a compact factual line."""
        parts: List[str] = [event_type or "event"]
        for field in ("phase", "subphase", "content_status", "qa_layer", "qa_model"):
            value = event.get(field)
            if value:
                parts.append(f"{field}={value}")
        iteration = event.get("iteration")
        if isinstance(iteration, int):
            parts.append(f"iteration={iteration}")
        reason = event.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        return " | ".join(parts)
