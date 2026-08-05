"""Central coalescing publisher for live content events (C0.4 / C0.5 / C0.8).

Every producer that wants to show work in progress (generation drafts,
thinking, QA evaluations, Arbiter, Gran Sabio, preflight, tool loops) publishes
through :func:`publish_content_event`. This is the SINGLE entry point of the
visibility contract and it does three things at once:

1. **Coalesces drafts** with a 250 ms sampling window keyed by
   ``(session, phase, subphase, iteration, content_status)``. The project bus
   has no coalescing of its own: every event is enqueued in an
   ``asyncio.Queue(maxsize=1000)`` and silently dropped when full, so
   publishing one bus event per provider token would flood subscribers.
2. **Guarantees ordering**: any non-draft emission (accepted / rejected /
   filtered / non-chunk event) flushes the pending drafts of its session+phase
   BEFORE it is published, so a consumer never sees a commit followed by the
   tail of the draft it replaced.
3. **Persists the trace** in the debugger, batched at one row per
   ``(session, phase, attempt)`` with at least one second between writes per
   key. Every write is positioned at the row length it expects, so a delta a
   cancelled write already applied is never appended twice. The bus is volatile
   fan-out (it can return without publishing when nobody is subscribed or the
   queue is full); the debugger is the memory that makes "look back at what
   happened" work with no subscriber attached.

Delivery semantics are best effort by design. Subscribers detect loss through
the per-``(session, phase)`` ``sequence`` field stamped by
``core.app_state.publish_project_phase_chunk``; the trailing sub-interval of a
stream may reach the bus slightly late (never out of order).

This module lives at the repository root, not under ``core/``, because the
producers that must call it (``arbiter.py``, ``qa_evaluation_service.py``,
``gran_sabio.py``, ``ai_service.py``, ``preflight_validator.py``) are imported
BY ``core`` — importing ``core.*`` from them would close an import cycle. Every
``core`` dependency here is resolved lazily inside the functions that need it.

Wiring notes for producers (fiches C1-C6):

- Build a :class:`~content_events.ContentEvent` (or reuse a
  ``ContentEventTemplate``) and ``await publish_content_event(event)``.
  Never call ``publish_project_phase_chunk`` directly for content.
- Emit terminal/commit events through the same call so the flush ordering and
  the debugger trace stay consistent.
- ``ContentEvent.attempt`` is published as ``attempt_id``; bump it whenever a
  new provider attempt starts so consumers can reset their cursors.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from content_events import ContentEvent, ContentEventType, ContentStatus

logger = logging.getLogger(__name__)

# Sampling window for live draft publication on the project bus.
DRAFT_SAMPLING_INTERVAL_SECONDS = 0.25
# Minimum spacing between debugger writes for the same trace row.
TRACE_WRITE_INTERVAL_SECONDS = 1.0

# Bus buffer key: session, phase, subphase, iteration, content_status, model,
# layer. Model and layer are part of the key because a single phase can have
# several producers in flight at once (QA dispatches its evaluators
# concurrently): without them, chunks from two evaluators would be concatenated
# into one event and published under whichever model streamed last.
_BusKey = Tuple[str, str, str, int, str, str, str]
# Debugger trace key: session, phase, attempt, model, layer. Model and layer
# are part of the identity for the same reason they are part of the bus key:
# concurrent producers inside one phase (QA evaluators) must never interleave
# their verbatim text inside a single trace row.
_TraceKey = Tuple[str, str, int, str, str]

# Upper bound for trace text retained in memory while writes keep failing.
TRACE_MAX_PENDING_CHARS = 2_000_000

ChunkPublisher = Callable[..., Awaitable[None]]
# The trace writer reports the persisted row length after applying the delta so
# the coalescer can resynchronize its offsets instead of adding lengths blindly;
# ``None`` means "no row to stay aligned with" (persistence unavailable).
TraceWriter = Callable[..., Awaitable[Optional[int]]]

_NO_ITERATION = -1
_NO_ATTEMPT = 0


@dataclass(slots=True)
class _PendingChunks:
    """Draft text buffered for one bus key between sampled publications."""

    event: ContentEvent
    parts: List[str] = field(default_factory=list)
    last_publish_at: float = 0.0
    timer: Optional[asyncio.Task] = None


@dataclass(slots=True)
class _PendingTrace:
    """Verbatim text buffered for one debugger row between batched writes."""

    session_id: str
    phase: str
    attempt: int
    model: str = ""
    layer: str = ""
    project_id: Optional[str] = None
    subphase: Optional[str] = None
    iteration: Optional[int] = None
    content_status: str = ContentStatus.DRAFT
    parts: List[str] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    chars_written: int = 0
    # Length of the delta a write is currently trying to persist. It has left
    # ``parts`` but has not reached ``chars_written`` yet, so it must still count
    # towards the offsets stamped on chunks buffered during the write.
    in_flight_chars: int = 0
    pending_chars: int = 0
    last_write_at: float = 0.0
    write_in_flight: bool = False
    # Set when the write in flight resolves, so cleanup can wait for the delta
    # that write owns instead of dropping the row from under it.
    write_settled: Optional[asyncio.Event] = None
    timer: Optional[asyncio.Task] = None


def _trace_key(event: ContentEvent) -> _TraceKey:
    """Build the debugger-row identity for one content event."""
    return (
        event.session_id or "",
        event.phase,
        event.attempt if event.attempt is not None else _NO_ATTEMPT,
        event.model or "",
        event.layer or "",
    )


async def _default_publish_chunk(**kwargs: Any) -> None:
    """Publish one event on the project bus (lazy import breaks the cycle)."""
    from core.app_state import publish_project_phase_chunk

    await publish_project_phase_chunk(**kwargs)


async def _default_write_trace(**kwargs: Any) -> Optional[int]:
    """Append one batched trace row update to the debugger database."""
    from core import app_state

    debug_logger = app_state.debug_logger
    if debug_logger is None:
        return None
    return await debug_logger.record_content_trace(**kwargs)


class ContentStreamCoalescer:
    """Single entry point that publishes, samples and persists content events.

    Args:
        sampling_interval: seconds between sampled draft publications per key.
        trace_interval: minimum seconds between debugger writes per trace row.
        clock: monotonic clock, injectable for deterministic tests.
        publisher: bus publisher, injectable for deterministic tests.
        trace_writer: debugger writer, injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        sampling_interval: float = DRAFT_SAMPLING_INTERVAL_SECONDS,
        trace_interval: float = TRACE_WRITE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        publisher: ChunkPublisher = _default_publish_chunk,
        trace_writer: TraceWriter = _default_write_trace,
    ) -> None:
        self._sampling_interval = sampling_interval
        self._trace_interval = trace_interval
        self._clock = clock
        self._publisher = publisher
        self._trace_writer = trace_writer
        self._chunks: Dict[_BusKey, _PendingChunks] = {}
        self._traces: Dict[_TraceKey, _PendingTrace] = {}
        self._publish_lock: Optional[asyncio.Lock] = None
        self._publish_loop: Optional[asyncio.AbstractEventLoop] = None

    # -- public API ---------------------------------------------------------

    async def publish(self, event: ContentEvent) -> None:
        """Publish one content event through the coalescing handler.

        Draft chunks are sampled; everything else (accepted / rejected /
        filtered commits, ``iteration_rejected``, tool events, terminal
        markers) flushes the pending drafts of the same session+phase first and
        is then published immediately.
        """
        self._buffer_trace(event)

        if event.is_draft_chunk:
            if event.content:
                key = self._bus_key(event)
                pending = self._chunks.get(key)
                if pending is not None and pending.event.attempt != event.attempt:
                    # A new provider attempt reuses the same bus key: the
                    # GranSabio truncation retry restarts the envelope and keeps
                    # streaming into the same phase/subphase/iteration. Publish
                    # the previous attempt's tail under its own labels before
                    # buffering, or both attempts would be merged into a single
                    # event carrying the newest attempt_id.
                    await self._publish_pending(key)
                self._buffer_chunks(key, event)
                await self._maybe_publish(key)
            self._schedule_trace(event)
            return

        await self.flush(event.session_id, phase=event.phase)
        await self._publish_now(event, event.content)

    async def flush(
        self,
        session_id: Optional[str],
        *,
        phase: Optional[str] = None,
    ) -> None:
        """Flush every pending buffer of a session (optionally one phase only).

        Mandatory before any accepted/rejected/terminal emission; the publisher
        already does it for events routed through :meth:`publish`. Call it
        directly when a producer emits a terminal marker by other means.
        """
        if not session_id:
            return

        for key in [
            key
            for key in self._chunks
            if key[0] == session_id and (phase is None or key[1] == phase)
        ]:
            await self._publish_pending(key)

        for key in [
            key
            for key in self._traces
            if key[0] == session_id and (phase is None or key[1] == phase)
        ]:
            await self._write_trace(key, force=True)

    def discard_session(self, session_id: str) -> None:
        """Drop every buffer of a session, cancelling its timers.

        Called from session cleanup. Unwritten trace text is scheduled for one
        last write so an abandoned session still leaves its trace behind.
        """
        if not session_id:
            return

        for key in [key for key in self._chunks if key[0] == session_id]:
            entry = self._chunks.pop(key)
            _cancel_timer(entry.timer)

        pending_trace_keys: List[_TraceKey] = []
        for key in [key for key in self._traces if key[0] == session_id]:
            entry = self._traces[key]
            _cancel_timer(entry.timer)
            entry.timer = None
            # A row with a write in flight is NOT drained: that write owns a
            # delta it may still restore into ``parts``, and dropping the entry
            # now would leave the restore writing into an object nobody can
            # reach again. The drain waits for it instead.
            if entry.parts or entry.write_in_flight:
                pending_trace_keys.append(key)
            else:
                self._traces.pop(key, None)

        if not pending_trace_keys:
            return

        try:
            task = asyncio.create_task(self._drain_traces(pending_trace_keys))
        except RuntimeError:
            for key in pending_trace_keys:
                entry = self._traces.get(key)
                if entry is not None and entry.write_in_flight:
                    # No loop here to await the write that owns the delta; the
                    # row stays so the retry that write arms can still find it.
                    continue
                self._traces.pop(key, None)
            logger.debug("No running loop to drain content traces for %s", session_id)
            return
        task.add_done_callback(_log_task_failure)

    # -- bus coalescing -----------------------------------------------------

    def _bus_key(self, event: ContentEvent) -> _BusKey:
        return (
            event.session_id or "",
            event.phase,
            event.subphase or "",
            event.iteration if event.iteration is not None else _NO_ITERATION,
            event.content_status,
            event.model or "",
            event.layer or "",
        )

    def _buffer_chunks(self, key: _BusKey, event: ContentEvent) -> None:
        entry = self._chunks.get(key)
        if entry is None:
            entry = _PendingChunks(event=event, last_publish_at=self._clock())
            self._chunks[key] = entry
        else:
            # Keep the freshest context: the key pins session, phase, subphase,
            # iteration, content_status, model and layer, and ``publish`` flushes
            # on an attempt change, so only non-identity metadata (project epoch,
            # request name, payload) can advance inside one buffered entry.
            entry.event = event
        entry.parts.append(event.content)

    async def _maybe_publish(self, key: _BusKey) -> None:
        entry = self._chunks.get(key)
        if entry is None or not entry.parts:
            return
        elapsed = self._clock() - entry.last_publish_at
        if elapsed >= self._sampling_interval:
            await self._publish_pending(key)
            return
        self._arm_timer(entry, self._sampling_interval - elapsed, self._on_chunk_timer, key)

    async def _publish_pending(self, key: _BusKey) -> None:
        entry = self._chunks.pop(key, None)
        if entry is None:
            return
        _cancel_timer(entry.timer)
        entry.timer = None
        if not entry.parts:
            return
        content = "".join(entry.parts)
        entry.parts.clear()
        # An external cancellation while ``_publish_now`` is suspended drops this
        # delta on the bus, and that is the accepted trade: the entry is already
        # out of ``self._chunks`` so no accounting is corrupted, while re-queueing
        # it would let a draft tail land AFTER the commit that replaced it and
        # break the ordering guarantee this coalescer exists to provide. Bus
        # delivery is best effort (subscribers detect gaps by ``sequence``); the
        # durable copy is the debugger trace, which restores its delta on cancel.
        await self._publish_now(entry.event, content)

    async def _on_chunk_timer(self, key: _BusKey) -> None:
        await self._publish_pending(key)

    def _get_publish_lock(self) -> asyncio.Lock:
        """Return the publish lock bound to the running loop.

        The coalescer is an engine-wide singleton while every test (and every
        server run) owns its event loop, so the lock is rebound when the loop
        changes instead of raising "attached to a different loop".
        """
        loop = asyncio.get_running_loop()
        if self._publish_lock is None or self._publish_loop is not loop:
            self._publish_lock = asyncio.Lock()
            self._publish_loop = loop
        return self._publish_lock

    async def _publish_now(self, event: ContentEvent, content: str) -> None:
        async with self._get_publish_lock():
            await self._publisher(
                project_id=event.project_id,
                phase=event.phase,
                content=content if content else None,
                subphase=event.subphase,
                session_id=event.session_id,
                project_epoch=event.project_epoch,
                request_name=event.request_name,
                is_thinking=True if event.is_thinking else None,
                event=event.event,
                content_status=event.content_status,
                iteration=event.iteration,
                attempt_id=event.attempt,
                reason=event.reason,
                qa_layer=event.layer,
                qa_model=event.model,
                edit_data=event.edit_data,
                data=event.data,
            )

    # -- debugger trace batching -------------------------------------------

    def _buffer_trace(self, event: ContentEvent) -> None:
        if not event.session_id or not event.content:
            return

        key = _trace_key(event)
        entry = self._traces.get(key)
        if entry is None:
            entry = _PendingTrace(
                session_id=key[0],
                phase=key[1],
                attempt=key[2],
                model=key[3],
                layer=key[4],
                last_write_at=self._clock(),
            )
            self._traces[key] = entry

        if entry.project_id is None:
            entry.project_id = event.project_id
        entry.subphase = event.subphase
        entry.iteration = event.iteration
        entry.content_status = event.content_status

        # The offset base spans the three states a character can be in: already
        # persisted, taken by the write in flight, and still buffered. Leaving
        # ``in_flight_chars`` out would stamp every chunk buffered during a write
        # short by that delta, cumulatively, because the writer always suspends.
        offset = entry.chars_written + entry.in_flight_chars + entry.pending_chars
        last_segment = entry.segments[-1] if entry.segments else None
        if (
            last_segment is not None
            and last_segment["subphase"] == event.subphase
            and last_segment["iteration"] == event.iteration
            and last_segment["content_status"] == event.content_status
            and last_segment["is_thinking"] == event.is_thinking
        ):
            last_segment["end"] = offset + len(event.content)
        else:
            entry.segments.append(
                {
                    "subphase": event.subphase,
                    "iteration": event.iteration,
                    "content_status": event.content_status,
                    "is_thinking": event.is_thinking,
                    "start": offset,
                    "end": offset + len(event.content),
                }
            )
        entry.parts.append(event.content)
        entry.pending_chars += len(event.content)

    def _schedule_trace(self, event: ContentEvent) -> None:
        """Arm the batched write for a draft chunk that is still buffered."""
        if not event.session_id or not event.content:
            return
        key = _trace_key(event)
        entry = self._traces.get(key)
        if entry is None or entry.write_in_flight or not entry.parts:
            return
        elapsed = self._clock() - entry.last_write_at
        if elapsed >= self._trace_interval:
            self._spawn_trace_write(key)
            return
        self._arm_timer(entry, self._trace_interval - elapsed, self._on_trace_timer, key)

    def _spawn_trace_write(self, key: _TraceKey) -> None:
        try:
            task = asyncio.create_task(self._write_trace(key, force=True))
        except RuntimeError:
            logger.debug("No running loop to write content trace for %s", key[0])
            return
        task.add_done_callback(_log_task_failure)

    async def _on_trace_timer(self, key: _TraceKey) -> None:
        await self._write_trace(key, force=True)

    async def _write_trace(self, key: _TraceKey, *, force: bool) -> None:
        entry = self._traces.get(key)
        if entry is None:
            return
        _cancel_timer(entry.timer)
        entry.timer = None
        if not entry.parts:
            return
        if entry.write_in_flight:
            # A write is already in flight for this row; the buffered tail is
            # written by the follow-up scheduled when that write completes.
            return
        if not force and (self._clock() - entry.last_write_at) < self._trace_interval:
            return

        content = "".join(entry.parts)
        entry.parts.clear()
        entry.pending_chars = 0
        entry.in_flight_chars = len(content)
        expected_offset = entry.chars_written
        entry.write_in_flight = True
        entry.write_settled = asyncio.Event()
        segments = [dict(segment) for segment in entry.segments]
        persisted = False
        applied_chars: Optional[int] = None
        try:
            applied_chars = await self._trace_writer(
                session_id=entry.session_id,
                phase=entry.phase,
                attempt=entry.attempt,
                model=entry.model,
                layer=entry.layer,
                project_id=entry.project_id,
                subphase=entry.subphase,
                iteration=entry.iteration,
                content_status=entry.content_status,
                content_delta=content,
                expected_offset=expected_offset,
                segments=segments,
            )
            persisted = True
        except asyncio.CancelledError:
            # External cancellation (shutdown, hard stop) while the writer was
            # suspended: this call cannot know whether the delta reached the row,
            # so it goes back into the buffer with the offsets it was stamped
            # with and the retry re-sends it under the same ``expected_offset``.
            # That offset is what keeps the retry from appending twice when the
            # cancelled statement did execute on the aiosqlite worker thread.
            # The ``finally`` below releases ``write_in_flight``, so
            # ``flush``/``discard_session`` or the next chunk can still drain it;
            # arming a timer here would mean spawning a task on a loop that is
            # being torn down.
            _restore_unwritten_delta(entry, content)
            raise
        except Exception:
            logger.exception(
                "Failed to persist content trace for session %s phase %s",
                entry.session_id,
                entry.phase,
            )
        finally:
            entry.write_in_flight = False
            entry.last_write_at = self._clock()
            settled = entry.write_settled
            entry.write_settled = None
            if settled is not None:
                settled.set()

        if persisted:
            # ``applied_chars`` is the authoritative length of the row after an
            # append positioned at ``expected_offset``: text that a cancelled
            # attempt already got in is skipped instead of appended twice, so
            # trusting the writer's number is what keeps the offsets truthful.
            # ``None`` means there is no row at all (persistence unavailable) and
            # the in-memory accounting only has to stay monotonic.
            entry.chars_written = (
                entry.chars_written + len(content)
                if applied_chars is None
                else applied_chars
            )
            entry.in_flight_chars = 0
        else:
            # The row did not grow: keep the delta at the head of the buffer so
            # segment offsets stay aligned with the persisted text, unless the
            # backlog is already unreasonable.
            _restore_unwritten_delta(entry, content)
            if entry.pending_chars > TRACE_MAX_PENDING_CHARS:
                logger.error(
                    "Dropping %d buffered trace chars for session %s phase %s "
                    "after repeated write failures",
                    entry.pending_chars,
                    entry.session_id,
                    entry.phase,
                )
                entry.parts.clear()
                entry.pending_chars = 0

        if entry.parts:
            self._arm_timer(entry, self._trace_interval, self._on_trace_timer, key)

    async def _await_write_settled(self, key: _TraceKey) -> None:
        """Wait until no write is in flight for one trace row.

        ``_write_trace`` returns immediately when a write is already in flight,
        so any caller that intends to drain and then release the row has to let
        that write resolve first: it owns a delta it may restore, and it is the
        one that advances ``chars_written``.

        The re-check loop is what makes concurrent drains of the same row safe: a
        duplicate ``discard_session`` spawns a second drain, and when the write
        they both wait for settles, the first one immediately starts the follow-up
        write. Returning on the first wake would release the row from under THAT
        write and orphan the delta it may restore.
        """
        while True:
            entry = self._traces.get(key)
            if entry is None or not entry.write_in_flight:
                return
            settled = entry.write_settled
            if settled is None:
                return
            await settled.wait()

    async def _drain_traces(self, keys: List[_TraceKey]) -> None:
        """Write the last delta of abandoned rows and release them."""
        for key in keys:
            await self._await_write_settled(key)
            await self._write_trace(key, force=True)
            entry = self._traces.pop(key, None)
            if entry is None:
                continue
            # The row is gone now, so the retry timer that a failed write may
            # have armed would find nothing: cancel it and say out loud what is
            # being lost instead of leaving a silent gap in the trace.
            _cancel_timer(entry.timer)
            entry.timer = None
            if entry.parts:
                logger.error(
                    "Dropping %d unwritten trace chars for session %s phase %s "
                    "left behind by session cleanup",
                    entry.pending_chars,
                    entry.session_id,
                    entry.phase,
                )

    # -- timers -------------------------------------------------------------

    def _arm_timer(
        self,
        entry: Any,
        delay: float,
        callback: Callable[[Any], Awaitable[None]],
        key: Any,
    ) -> None:
        if entry.timer is not None and not entry.timer.done():
            return
        try:
            entry.timer = asyncio.create_task(_delayed(delay, callback, key))
        except RuntimeError:
            entry.timer = None
            logger.debug("No running loop to sample content events for %s", key)
            return
        entry.timer.add_done_callback(_log_task_failure)


async def _delayed(
    delay: float,
    callback: Callable[[Any], Awaitable[None]],
    key: Any,
) -> None:
    """Run ``callback(key)`` once after ``delay`` seconds."""
    if delay > 0:
        await asyncio.sleep(delay)
    await callback(key)


def _cancel_timer(timer: Optional[asyncio.Task]) -> None:
    """Cancel a pending timer task, never the task that is running right now.

    ``_on_chunk_timer`` and ``_on_trace_timer`` run INSIDE ``entry.timer``, and a
    running task is never ``done()``: cancelling it from there would schedule a
    CancelledError at the next suspending await, i.e. in the middle of the
    publication or the trace write whose buffers were already drained. Callers
    still reset ``entry.timer`` afterwards so a replacement timer can be armed.
    """
    if timer is None or timer.done():
        return
    try:
        running = asyncio.current_task()
    except RuntimeError:
        # Sync session cleanup with no running loop: nothing to self-cancel.
        running = None
    if timer is running:
        return
    timer.cancel()


def _restore_unwritten_delta(entry: _PendingTrace, content: str) -> None:
    """Return an unwritten delta to the FRONT of the trace buffer.

    ``chars_written`` only advances once the writer confirms the row grew, so a
    delta that did not reach the database must go back at the head of ``parts``
    (and into ``pending_chars``) or the segment offsets already stamped in
    memory would drift from the persisted text. ``in_flight_chars`` is cleared
    for the same reason: the delta keeps the offsets it was stamped with, only
    counted as buffered again instead of as in flight.
    """
    entry.parts.insert(0, content)
    entry.pending_chars += len(content)
    entry.in_flight_chars = 0


def _log_task_failure(task: "asyncio.Task[None]") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Content stream task failed: %s", exc)


_coalescer = ContentStreamCoalescer()


def get_content_stream_coalescer() -> ContentStreamCoalescer:
    """Return the engine-wide coalescing publisher."""
    return _coalescer


async def publish_content_event(event: ContentEvent) -> None:
    """Publish one content event through the engine-wide handler."""
    await _coalescer.publish(event)


async def flush_content_events(
    session_id: Optional[str],
    *,
    phase: Optional[str] = None,
) -> None:
    """Flush pending draft buffers before a terminal or commit emission."""
    await _coalescer.flush(session_id, phase=phase)


def discard_session_content_state(session_id: str) -> None:
    """Release buffers of a session that is leaving the runtime."""
    _coalescer.discard_session(session_id)


__all__ = [
    "DRAFT_SAMPLING_INTERVAL_SECONDS",
    "TRACE_WRITE_INTERVAL_SECONDS",
    "ContentEvent",
    "ContentEventType",
    "ContentStatus",
    "ContentStreamCoalescer",
    "discard_session_content_state",
    "flush_content_events",
    "get_content_stream_coalescer",
    "publish_content_event",
]
