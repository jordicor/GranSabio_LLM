"""Bounded idempotency state for generation admission requests."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Literal

IdempotencyState = Literal["pending", "completed", "error"]
ClaimAction = Literal["execute", "wait", "replay"]


class IdempotencyConflictError(Exception):
    """Raised when a request ID is reused with a different payload."""


class IdempotencyCapacityError(Exception):
    """Raised when all bounded registry slots contain active requests."""


@dataclass
class GenerationIdempotencyEntry:
    """One generation admission tracked by its SaaS request ID."""

    request_id: str
    fingerprint: str
    state: IdempotencyState = "pending"
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    response_payload: dict[str, Any] | None = None
    error_status_code: int | None = None
    error_detail: Any = None
    error_headers: dict[str, str] | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass(frozen=True)
class GenerationIdempotencyClaim:
    """Result of claiming an idempotency key."""

    action: ClaimAction
    entry: GenerationIdempotencyEntry


class GenerationIdempotencyRegistry:
    """Process-local, bounded registry for generation admission idempotency."""

    def __init__(self, *, ttl_seconds: int, max_entries: int) -> None:
        self._ttl_seconds = max(300, int(ttl_seconds))
        self._max_entries = max(100, int(max_entries))
        self._entries: dict[str, GenerationIdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    async def claim(self, request_id: str, fingerprint: str) -> GenerationIdempotencyClaim:
        """Claim a key, wait for its owner, or replay its terminal result."""
        now = time.monotonic()
        async with self._lock:
            self._remove_expired_locked(now)
            entry = self._entries.get(request_id)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise IdempotencyConflictError(request_id)
                action: ClaimAction = "wait" if entry.state == "pending" else "replay"
                return GenerationIdempotencyClaim(action=action, entry=entry)

            self._make_room_locked()
            entry = GenerationIdempotencyEntry(
                request_id=request_id,
                fingerprint=fingerprint,
                created_at=now,
                updated_at=now,
            )
            self._entries[request_id] = entry
            return GenerationIdempotencyClaim(action="execute", entry=entry)

    async def complete(
        self,
        entry: GenerationIdempotencyEntry,
        response_payload: dict[str, Any],
    ) -> None:
        """Store a successful admission response and release duplicate waiters."""
        async with self._lock:
            if self._entries.get(entry.request_id) is not entry:
                return
            entry.state = "completed"
            entry.response_payload = copy.deepcopy(response_payload)
            entry.updated_at = time.monotonic()
            entry.event.set()

    async def fail(
        self,
        entry: GenerationIdempotencyEntry,
        *,
        status_code: int,
        detail: Any,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store a deterministic HTTP failure and release duplicate waiters."""
        async with self._lock:
            if self._entries.get(entry.request_id) is not entry:
                return
            entry.state = "error"
            entry.error_status_code = status_code
            entry.error_detail = copy.deepcopy(detail)
            entry.error_headers = dict(headers) if headers else None
            entry.updated_at = time.monotonic()
            entry.event.set()

    async def abort(self, entry: GenerationIdempotencyEntry) -> None:
        """Release a key when admission ended without a replayable HTTP result."""
        async with self._lock:
            if self._entries.get(entry.request_id) is not entry:
                return
            self._entries.pop(entry.request_id, None)
            entry.event.set()

    async def wait(self, entry: GenerationIdempotencyEntry) -> None:
        """Wait until the owner stores a result or aborts its claim."""
        await entry.event.wait()

    async def lookup(self, request_id: str) -> dict[str, Any]:
        """Return a public reconciliation snapshot without request payload data."""
        async with self._lock:
            self._remove_expired_locked(time.monotonic())
            entry = self._entries.get(request_id)
            if entry is None:
                return {"found": False, "request_id": request_id}

            snapshot: dict[str, Any] = {
                "found": True,
                "request_id": request_id,
                "state": entry.state,
                "status": "pending",
                "session_id": None,
                "project_id": None,
                "init_response": None,
            }
            if entry.state == "completed":
                response_payload = copy.deepcopy(entry.response_payload or {})
                snapshot.update(
                    status=response_payload.get("status", "completed"),
                    session_id=response_payload.get("session_id"),
                    project_id=response_payload.get("project_id"),
                    init_response=response_payload,
                )
            elif entry.state == "error":
                snapshot.update(
                    status="error",
                    error={
                        "status_code": entry.error_status_code,
                        "detail": copy.deepcopy(entry.error_detail),
                    },
                )
            return snapshot

    async def clear(self) -> None:
        """Clear registry state. Intended for deterministic tests and shutdown."""
        async with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            for entry in entries:
                entry.event.set()

    def _remove_expired_locked(self, now: float) -> None:
        expired_ids = [
            request_id
            for request_id, entry in self._entries.items()
            if now - entry.updated_at >= self._ttl_seconds
        ]
        for request_id in expired_ids:
            entry = self._entries.pop(request_id)
            entry.event.set()

    def _make_room_locked(self) -> None:
        if len(self._entries) < self._max_entries:
            return
        terminal_entries = [
            entry for entry in self._entries.values() if entry.state != "pending"
        ]
        if not terminal_entries:
            raise IdempotencyCapacityError
        oldest = min(terminal_entries, key=lambda entry: entry.updated_at)
        self._entries.pop(oldest.request_id, None)
        oldest.event.set()
