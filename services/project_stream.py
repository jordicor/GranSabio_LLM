"""
Unified project streaming manager.

Multiplexes events from multiple phases into a single SSE stream.
Enriches status events with phase field for consistency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

import json_utils as json
from content_events import Visibility, is_hidden_by_canonical_visibility

logger = logging.getLogger(__name__)

# Note: core.app_state imports are done lazily in methods to avoid
# circular import (core/__init__.py imports streaming_routes which imports us)

VALID_CONTENT_PHASES = {"auto_qa", "preflight", "generation", "qa", "arbiter", "smart_edit", "consensus", "gran_sabio"}
VALID_PHASES = VALID_CONTENT_PHASES | {"status"}
PHASE_ALIASES = {"gransabio": "gran_sabio", "smartedit": "smart_edit"}
HEARTBEAT_INTERVAL = 15

# Event types that end a project stream for every subscriber.
PROJECT_CLOSING_EVENTS = ("stream_end", "project_cancelled", "project_end")


def _now_ms() -> int:
    """Current wall-clock milliseconds (event timestamps)."""
    return int(time.time() * 1000)


class SubscriptionError(Exception):
    """Raised when subscription to phases fails."""
    pass


def parse_phases(phases_param: str) -> Set[str]:
    """
    Parse and validate phases parameter.

    Handles:
    - "all" or empty -> all phases
    - CSV of phase names -> requested phases
    - Aliases (gransabio -> gran_sabio)
    - Empty result (e.g., "phases=,") -> all phases

    Raises:
        ValueError: If invalid phase names or "all" combined with others
    """
    if not phases_param or phases_param.strip().lower() == "all":
        return VALID_PHASES.copy()

    requested = set()
    has_all = False

    for p in phases_param.split(","):
        phase = p.strip().lower()
        if not phase:
            continue
        if phase == "all":
            has_all = True
            continue
        # Apply alias if exists
        phase = PHASE_ALIASES.get(phase, phase)
        requested.add(phase)

    # "all" combined with specific phases is ambiguous -> error
    if has_all and requested:
        raise ValueError("Cannot combine 'all' with specific phases. Use 'all' alone or list specific phases.")

    if has_all:
        return VALID_PHASES.copy()

    # Empty result (e.g., "phases=,") -> default to all
    if not requested:
        return VALID_PHASES.copy()

    # Validate all requested phases
    invalid = requested - VALID_PHASES
    if invalid:
        valid_list = sorted(VALID_PHASES | set(PHASE_ALIASES.keys()))
        raise ValueError(f"Invalid phases: {invalid}. Valid: {valid_list}")

    return requested


class ProjectStreamManager:
    """
    Manages unified SSE streaming for a project across multiple phases.

    Features:
    - Multiplexes events from selected phases into single stream
    - Enriches status events with "phase" field for consistency
    - Processes all simultaneous events (no event loss)
    - Immediate close on cancel (no queue draining)
    - Logs subscription failures for debugging
    - Raises SubscriptionError if no subscriptions succeed
    - Emits synthetic stream_end event to guarantee close reason
    - Applies the subscriber ``visibility`` filter (C0.6): ``canonical`` drops
      draft and thinking content, both live and in the late-join snapshot

    Delivery is best effort: content events carry a ``sequence`` that is
    monotonic per (session, phase), so a subscriber that cannot keep up detects
    the loss through a gap instead of silently missing content.
    """

    def __init__(
        self,
        project_id: str,
        phases: Set[str],
        visibility: str = Visibility.FULL,
    ):
        self.project_id = project_id
        self.phases = phases
        self.visibility = visibility
        self._content_queues: Dict[str, asyncio.Queue] = {}
        self._status_queue: Optional[asyncio.Queue] = None
        self._subscribed_phases: Set[str] = set()

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """
        Main streaming generator. Yields SSE-formatted bytes.

        Flow:
        1. Subscribe to all requested phases (raises if none succeed)
        2. Send connected event (built before subscribing, sent after it)
        3. If status included, send initial snapshot
        4. Multiplex events from all phases
        5. Emit synthetic stream_end if needed
        6. On close/cancel, unsubscribe immediately
        """
        # Lazy import to avoid circular dependency
        from core.app_state import get_project_status

        # The greeting is built from local state before subscribing, but it is
        # only sent once the subscriptions exist: announcing the stream first
        # would promise a delivery window that has not started yet, and events
        # published in between would be lost with no gap to detect them.
        connected_event = self._build_connected_event()

        try:
            # 1. Subscribe to phases (raises SubscriptionError if all fail)
            await self._subscribe_to_phases()

            # 2. Connected event (first payload on the wire)
            yield self._format_sse(connected_event)

            # 3. Status snapshot (only if status requested)
            if "status" in self.phases:
                snapshot = await get_project_status(
                    self.project_id,
                    include_content_snapshot=True,
                    visibility=self.visibility,
                )
                snapshot_event = self._build_snapshot_event(snapshot)
                if snapshot_event is not None:
                    yield self._format_sse(snapshot_event)

            # 4. Multiplex events from all subscribed phases
            async for event in self._multiplex_events():
                if event is None:
                    # Heartbeat signal
                    yield b": heartbeat\n\n"
                else:
                    yield self._format_sse(event)

        finally:
            # 6. Always cleanup on exit
            await self._unsubscribe_all()

    async def _subscribe_to_phases(self) -> None:
        """
        Subscribe to all requested phases.
        Tracks each successful subscription for proper cleanup.
        Logs warnings for failed subscriptions.
        Raises SubscriptionError if no subscriptions succeed.
        """
        # Lazy imports to avoid circular dependency
        from core.app_state import subscribe_project_phase, subscribe_project_status

        # Content phases
        content_phases = self.phases & VALID_CONTENT_PHASES
        for phase in content_phases:
            try:
                queue = await subscribe_project_phase(self.project_id, phase)
                self._content_queues[phase] = queue
                self._subscribed_phases.add(phase)
            except Exception as e:
                logger.warning(
                    "Failed to subscribe to phase %s for project %s: %s",
                    phase, self.project_id, e
                )

        # Status phase (separate subscription system)
        if "status" in self.phases:
            try:
                self._status_queue = await subscribe_project_status(self.project_id)
                self._subscribed_phases.add("status")
            except Exception as e:
                logger.warning(
                    "Failed to subscribe to status for project %s: %s",
                    self.project_id, e
                )

        # Validate at least one subscription succeeded
        if not self._subscribed_phases:
            logger.error(
                "Failed to subscribe to any phase for project %s (requested: %s)",
                self.project_id, self.phases
            )
            raise SubscriptionError(
                f"Failed to subscribe to any requested phase for project {self.project_id}"
            )

        # Log successful subscriptions
        logger.debug(
            "Subscribed to phases %s for project %s",
            self._subscribed_phases, self.project_id
        )

    async def _unsubscribe_all(self) -> None:
        """Unsubscribe from all phases that were successfully subscribed."""
        # Lazy imports to avoid circular dependency
        from core.app_state import unsubscribe_project_phase, unsubscribe_project_status

        for phase in list(self._subscribed_phases):
            if phase == "status":
                if self._status_queue:
                    try:
                        await unsubscribe_project_status(self.project_id, self._status_queue)
                    except Exception:
                        pass
                    self._status_queue = None
            else:
                queue = self._content_queues.get(phase)
                if queue:
                    try:
                        await unsubscribe_project_phase(self.project_id, phase, queue)
                    except Exception:
                        pass

        self._content_queues.clear()
        self._subscribed_phases.clear()

    async def _multiplex_events(self) -> AsyncGenerator[Optional[str], None]:
        """
        Multiplex events from all subscribed queues.

        Yields:
            - JSON string for real events (enriched with phase if needed)
            - None for heartbeat signal

        Features:
            - Processes ALL completed tasks in each iteration (no event loss)
            - Enriches status events with "phase": "status"
            - Closes immediately on sentinel or stream_end event
            - Emits synthetic stream_end if close was triggered by non-status event
        """
        active_tasks: Dict[asyncio.Task, Tuple[str, asyncio.Queue]] = {}

        def _create_task(phase: str, queue: asyncio.Queue) -> None:
            task = asyncio.create_task(queue.get())
            active_tasks[task] = (phase, queue)

        # Initialize tasks for all queues
        for phase, queue in self._content_queues.items():
            _create_task(phase, queue)
        if self._status_queue:
            _create_task("status", self._status_queue)

        # Track close state for synthetic event emission
        close_reason: Optional[str] = None
        closed_by_status: bool = False

        try:
            while active_tasks:
                # Use set() to avoid issues with dynamic dict view
                try:
                    done, _ = await asyncio.wait(
                        set(active_tasks.keys()),
                        timeout=HEARTBEAT_INTERVAL,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                except Exception:
                    close_reason = "wait_error"
                    break

                if not done:
                    # Timeout - send heartbeat
                    yield None
                    continue

                should_close = False

                # Process ALL completed tasks (not just one!)
                for task in done:
                    phase, queue = active_tasks.pop(task)

                    try:
                        result = task.result()
                    except Exception:
                        # Task failed, don't recreate
                        continue

                    if result is None:
                        # Sentinel received - record and close
                        close_reason = "sentinel_received"
                        should_close = True
                        break

                    prepared = self._prepare_event(phase, result)
                    if prepared is None:
                        # Not for this subscriber (canonical visibility drops
                        # drafts, a session stream drops foreign sessions); the
                        # queue must keep flowing, so re-arm and move on.
                        _create_task(phase, queue)
                        continue

                    yield prepared

                    # Check for stream end events
                    try:
                        parsed = json.loads(prepared)
                    except Exception:
                        parsed = None

                    if isinstance(parsed, dict):
                        event_type = parsed.get("type")
                        if event_type in PROJECT_CLOSING_EVENTS:
                            closed_by_status = True
                            close_reason = parsed.get("reason", event_type)
                            should_close = True
                            break
                        subscriber_close = self._closes_on_event(parsed)
                        if subscriber_close:
                            close_reason = subscriber_close
                            should_close = True
                            break

                    # Recreate task for next event from this queue
                    _create_task(phase, queue)

                if should_close:
                    for pending in await self._collect_pending_events(active_tasks):
                        yield pending
                    break

        finally:
            # Cancel any remaining tasks
            for task in active_tasks:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # 5. Emit synthetic stream_end if status didn't provide one
        if not closed_by_status and close_reason:
            yield self._build_stream_end_event(close_reason)

    # -- subscriber hooks ---------------------------------------------------

    def _prepare_event(self, phase: str, event_json: str) -> Optional[str]:
        """Adapt one raw queue event to what this subscriber must receive.

        Returns ``None`` when the event must not reach this subscriber.
        """
        if phase == "status":
            return self._enrich_status_event(event_json)
        if self._is_hidden_by_visibility(event_json):
            return None
        return self._transform_content_event(event_json)

    def _transform_content_event(self, event_json: str) -> Optional[str]:
        """Adapt a content-phase event. The project stream forwards it as is."""
        return event_json

    def _closes_on_event(self, parsed: Dict[str, Any]) -> Optional[str]:
        """Return a close reason when this subscriber ends on ``parsed``.

        Project subscribers only end on project lifecycle events, which are
        handled by the multiplexer itself, so nothing extra closes them.
        """
        return None

    async def _collect_pending_events(
        self,
        pending_tasks: Dict[asyncio.Task, Tuple[str, asyncio.Queue]],
    ) -> List[str]:
        """Events that must still be delivered before closing the stream.

        The project stream closes immediately on cancel and never drains its
        queues; session-scoped subscribers override this to deliver what their
        own session had already produced.
        """
        return []

    def _build_snapshot_event(self, snapshot: Dict[str, Any]) -> Optional[str]:
        """Serialize the late-join status snapshot for this subscriber."""
        snapshot_event = {
            "type": "status_snapshot",
            "phase": "status",
            "project": snapshot,
            "timestamp": _now_ms(),
        }
        return json.dumps(snapshot_event, ensure_ascii=True)

    def _build_stream_end_event(self, reason: str) -> str:
        """Build the synthetic close event emitted when nothing else did."""
        synthetic_end = {
            "type": "stream_end",
            "phase": "status",
            "reason": reason,
            "timestamp": _now_ms(),
            "synthetic": True,  # Indicates this was generated by the manager
        }
        return json.dumps(synthetic_end, ensure_ascii=True)

    def _is_hidden_by_visibility(self, event_json: str) -> bool:
        """Return whether a content event must be withheld from this subscriber.

        Only ``visibility=canonical`` filters anything; the default (``full``)
        never parses the payload, so the hot path stays untouched. Terminal and
        control events never carry ``content_status="draft"``, so they always
        survive the filter.
        """
        if self.visibility != Visibility.CANONICAL:
            return False
        try:
            parsed = json.loads(event_json)
        except Exception:
            return False
        return is_hidden_by_canonical_visibility(
            content_status=parsed.get("content_status"),
            subphase=parsed.get("subphase"),
            is_thinking=parsed.get("is_thinking"),
        )

    def _enrich_status_event(self, event_json: str) -> Optional[str]:
        """
        Add "phase": "status" to status events for consistency.

        Status events from app_state.py don't include phase field,
        but content events do. This ensures all events in the unified
        stream have phase for easy client-side filtering.
        """
        try:
            parsed = json.loads(event_json)
            if "phase" not in parsed:
                parsed["phase"] = "status"
                return json.dumps(parsed, ensure_ascii=True)
        except Exception:
            pass
        return event_json

    def _build_connected_event(self) -> str:
        """Build the initial connected event."""
        event = {
            "type": "connected",
            "project_id": self.project_id,
            "subscribed_phases": sorted(self.phases),
            "visibility": self.visibility,
            "timestamp": _now_ms(),
        }
        return json.dumps(event, ensure_ascii=True)

    @staticmethod
    def _format_sse(data: str) -> bytes:
        """Format data as SSE event."""
        return f"data: {data}\n\n".encode("utf-8")


class SessionStreamManager(ProjectStreamManager):
    """Session-scoped SSE adapter over the project bus (C9).

    This is NOT a second event bus: it subscribes to the same project queues
    and projects what it reads for the owner of ONE session.

    - Content events of sibling sessions are dropped. Isolation is the only
      security rule of the visibility model: each one sees their own.
    - The late-join snapshot and every status event are reduced to the
      requested session, so siblings are never exposed - not through
      ``project.sessions``, not through the project ``summary``, and not
      through ``trigger_session``.
    - The lifecycle is flattened onto a stable contract: ``status``,
      ``current_phase``, ``iteration`` and ``max_iterations`` travel at the top
      level of every status event, next to the full ``session`` object.
    - The stream drains what is still queued for the session and closes right
      after the ``session_end`` of THIS session; the project stream would stay
      open until the project itself ends. ``session_end`` is fanned out over
      the content phases AND over the status channel, so EVERY subscription -
      ``phases=status`` alone included - has a session terminal to close on,
      and the duplicated copies collapse into the single terminal the client
      sees.

    Content events keep the ``sequence`` stamped per (session, phase) by the
    publisher, which is already per-session granularity (C0.5): a gap means the
    bus dropped events, never that nothing happened. Delivery stays best
    effort, and ``visibility`` behaves exactly as on the project stream.
    """

    def __init__(
        self,
        project_id: str,
        session_id: str,
        phases: Set[str],
        visibility: str = Visibility.FULL,
    ):
        super().__init__(project_id, phases, visibility)
        self.session_id = session_id

    # -- projection ---------------------------------------------------------

    def _session_view(self, project: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return this session's entry inside a project status payload."""
        for session in project.get("sessions") or []:
            if session.get("session_id") == self.session_id:
                return session
        return None

    def _build_session_status_event(
        self,
        event_type: str,
        project: Dict[str, Any],
        timestamp: int,
    ) -> str:
        """Project a project-status payload onto the flat session contract."""
        session = self._session_view(project)
        payload: Dict[str, Any] = {
            "type": event_type,
            "phase": "status",
            "session_id": self.session_id,
            "session": session,
            # Only project-level identity/lifecycle: never `sessions` (sibling
            # payloads) nor `summary` (counters that describe siblings).
            "project": {
                key: project.get(key)
                for key in ("project_id", "status", "cancel_mode", "project_epoch")
                if key in project
            },
            "timestamp": timestamp,
        }
        if session is not None:
            payload["status"] = session.get("status")
            payload["current_phase"] = session.get("phase")
            payload["iteration"] = session.get("iteration")
            payload["max_iterations"] = session.get("max_iterations")
        return json.dumps(payload, ensure_ascii=True, default=str)

    def _build_snapshot_event(self, snapshot: Dict[str, Any]) -> Optional[str]:
        return self._build_session_status_event(
            "status_snapshot",
            snapshot,
            _now_ms(),
        )

    def _enrich_status_event(self, event_json: str) -> Optional[str]:
        """Reduce a project status event to this session, or drop it."""
        try:
            parsed = json.loads(event_json)
        except Exception:
            # An event we cannot inspect may carry sibling sessions: dropping
            # it is the only safe option for a session-scoped subscriber.
            logger.warning(
                "Dropping unparseable status event on session stream %s",
                self.session_id,
            )
            return None
        if not isinstance(parsed, dict):
            return None

        event_session_id = parsed.get("session_id")
        if event_session_id and event_session_id != self.session_id:
            # A session-scoped status event owned by a sibling (its
            # ``session_end``, for instance): forwarding it would expose the
            # sibling, and stamping it as ours would close this stream early.
            return None

        event_type = parsed.get("type") or "status_change"
        timestamp = parsed.get("timestamp") or _now_ms()
        project = parsed.get("project")
        if isinstance(project, dict):
            if (
                event_type not in PROJECT_CLOSING_EVENTS
                and self._session_view(project) is None
            ):
                # A sibling session moved: nothing happened to this one.
                return None
            return self._build_session_status_event(event_type, project, timestamp)

        payload: Dict[str, Any] = {
            "type": event_type,
            "phase": "status",
            "session_id": self.session_id,
            "timestamp": timestamp,
        }
        # Session-scoped identity fields the content-phase copy of the same
        # terminal already carries. Forwarding them here is what makes the shape
        # deterministic for a ``phases=all`` subscriber, where the copy that wins
        # the race decides what the client sees.
        for field in ("project_epoch", "request_name"):
            value = parsed.get(field)
            if value is not None:
                payload[field] = value
        status = parsed.get("status")
        if status:
            # The session terminal travels on this channel too: its ``status``
            # is what tells the subscriber how the session ended and what
            # ``_closes_on_event`` reports as the close reason.
            payload["status"] = status
        reason = parsed.get("reason")
        if reason:
            payload["reason"] = reason
        return json.dumps(payload, ensure_ascii=True)

    def _transform_content_event(self, event_json: str) -> Optional[str]:
        """Forward content events of this session only."""
        try:
            parsed = json.loads(event_json)
        except Exception:
            logger.warning(
                "Dropping unparseable content event on session stream %s",
                self.session_id,
            )
            return None
        if not isinstance(parsed, dict):
            return None
        session_id = parsed.get("session_id")
        if session_id == self.session_id:
            return event_json
        if session_id is None and parsed.get("type") in PROJECT_CLOSING_EVENTS:
            # Project lifecycle marker: it carries no sibling session data and
            # the subscriber must learn why the stream is about to end.
            return event_json
        return None

    def _closes_on_event(self, parsed: Dict[str, Any]) -> Optional[str]:
        if (
            parsed.get("type") == "session_end"
            and parsed.get("session_id") == self.session_id
        ):
            return str(parsed.get("status") or "session_end")
        return None

    async def _collect_pending_events(
        self,
        pending_tasks: Dict[asyncio.Task, Tuple[str, asyncio.Queue]],
    ) -> List[str]:
        """Drain what this session already produced before closing.

        ``session_end`` is published to every content phase of the project and
        to its status channel, so the moment the first copy closes the stream
        the other queues still hold content this session produced. Everything
        queued is delivered (the duplicated ``session_end`` copies excluded)
        and then the stream ends.
        """
        collected: List[str] = []
        drained_queues: Set[int] = set()
        for task, (phase, queue) in list(pending_tasks.items()):
            if task.done() and not task.cancelled():
                try:
                    self._append_pending(collected, phase, task.result())
                except Exception:
                    logger.debug("Pending stream task failed for %s", phase, exc_info=True)
            if id(queue) in drained_queues:
                continue
            drained_queues.add(id(queue))
            while True:
                try:
                    raw = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                self._append_pending(collected, phase, raw)
        return collected

    def _append_pending(
        self,
        collected: List[str],
        phase: str,
        raw: Optional[str],
    ) -> None:
        """Add one drained event to the closing batch when it belongs here."""
        if raw is None:
            return
        prepared = self._prepare_event(phase, raw)
        if prepared is None:
            return
        try:
            parsed = json.loads(prepared)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("type") in PROJECT_CLOSING_EVENTS:
                return
            if self._closes_on_event(parsed):
                # Duplicate session_end copy from another phase of the fan-out.
                return
        collected.append(prepared)

    def _build_stream_end_event(self, reason: str) -> str:
        synthetic_end = {
            "type": "stream_end",
            "phase": "status",
            "session_id": self.session_id,
            "reason": reason,
            "timestamp": _now_ms(),
            "synthetic": True,
        }
        return json.dumps(synthetic_end, ensure_ascii=True)

    def _build_connected_event(self) -> str:
        event = {
            "type": "connected",
            "session_id": self.session_id,
            "project_id": self.project_id,
            "subscribed_phases": sorted(self.phases),
            "visibility": self.visibility,
            "timestamp": _now_ms(),
        }
        return json.dumps(event, ensure_ascii=True)
