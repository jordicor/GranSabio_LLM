"""
Unified project streaming manager.

Multiplexes events from multiple phases into a single SSE stream.
Enriches status events with phase field for consistency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, Optional, Set, Tuple

import json_utils as json


logger = logging.getLogger(__name__)

# Note: core.app_state imports are done lazily in methods to avoid
# circular import (core/__init__.py imports streaming_routes which imports us)

VALID_CONTENT_PHASES = {"preflight", "generation", "qa", "arbiter", "smart_edit", "consensus", "gran_sabio"}
VALID_PHASES = VALID_CONTENT_PHASES | {"status"}
PHASE_ALIASES = {"gransabio": "gran_sabio", "smartedit": "smart_edit"}
HEARTBEAT_INTERVAL = 15


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
    """

    def __init__(self, project_id: str, phases: Set[str]):
        self.project_id = project_id
        self.phases = phases
        self._content_queues: Dict[str, asyncio.Queue] = {}
        self._status_queue: Optional[asyncio.Queue] = None
        self._subscribed_phases: Set[str] = set()

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """
        Main streaming generator. Yields SSE-formatted bytes.

        Flow:
        1. Send connected event
        2. Subscribe to all requested phases (raises if none succeed)
        3. If status included, send initial snapshot
        4. Multiplex events from all phases
        5. Emit synthetic stream_end if needed
        6. On close/cancel, unsubscribe immediately
        """
        # Lazy import to avoid circular dependency
        from core.app_state import get_project_status

        # 1. Connected event (always first)
        yield self._format_sse(self._build_connected_event())

        try:
            # 2. Subscribe to phases (raises SubscriptionError if all fail)
            await self._subscribe_to_phases()

            # 3. Status snapshot (only if status requested)
            if "status" in self.phases:
                snapshot = await get_project_status(self.project_id)
                snapshot_event = {
                    "type": "status_snapshot",
                    "phase": "status",
                    "project": snapshot,
                    "timestamp": int(time.time() * 1000),
                }
                yield self._format_sse(json.dumps(snapshot_event, ensure_ascii=True))

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

                    # Enrich status events with phase field for consistency
                    if phase == "status":
                        result = self._enrich_status_event(result)

                    yield result

                    # Check for stream end events
                    try:
                        parsed = json.loads(result)
                        event_type = parsed.get("type")
                        if event_type in ("stream_end", "project_cancelled"):
                            closed_by_status = True
                            close_reason = parsed.get("reason", event_type)
                            should_close = True
                            break
                    except Exception:
                        pass

                    # Recreate task for next event from this queue
                    _create_task(phase, queue)

                if should_close:
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
            synthetic_end = {
                "type": "stream_end",
                "phase": "status",
                "reason": close_reason,
                "timestamp": int(time.time() * 1000),
                "synthetic": True,  # Indicates this was generated by the manager
            }
            yield json.dumps(synthetic_end, ensure_ascii=True)

    def _enrich_status_event(self, event_json: str) -> str:
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
            "timestamp": int(time.time() * 1000),
        }
        return json.dumps(event, ensure_ascii=True)

    @staticmethod
    def _format_sse(data: str) -> bytes:
        """Format data as SSE event."""
        return f"data: {data}\n\n".encode("utf-8")
