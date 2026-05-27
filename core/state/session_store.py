"""Session storage helpers used by core.app_state."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


async def mutate_session(
    sessions: dict[str, dict[str, Any]],
    lock: Optional[asyncio.Lock],
    session_id: str,
    mutator: Callable[[dict[str, Any]], T],
) -> Optional[T]:
    """Run a mutator while holding the provided session lock."""

    if lock is None:
        session = sessions.get(session_id)
        if session is None:
            return None
        return mutator(session)
    async with lock:
        session = sessions.get(session_id)
        if session is None:
            return None
        return mutator(session)


async def increment_late_writes_blocked(
    sessions: dict[str, dict[str, Any]],
    lock: Optional[asyncio.Lock],
    session_id: str,
) -> None:
    """Increment the hard-cancel late-write counter."""

    def _increment(session: dict[str, Any]) -> None:
        session["late_writes_blocked"] = session.get("late_writes_blocked", 0) + 1

    await mutate_session(sessions, lock, session_id, _increment)


async def session_exists(
    sessions: dict[str, dict[str, Any]],
    lock: Optional[asyncio.Lock],
    session_id: str,
) -> bool:
    """Check if a session exists under the provided lock."""

    if lock is None:
        return session_id in sessions
    async with lock:
        return session_id in sessions


async def get_session(
    sessions: dict[str, dict[str, Any]],
    lock: Optional[asyncio.Lock],
    session_id: str,
) -> Optional[dict[str, Any]]:
    """Retrieve a session under the provided lock."""

    if lock is None:
        return sessions.get(session_id)
    async with lock:
        return sessions.get(session_id)


def is_terminal_session(session: dict[str, Any], final_status_values: set[str]) -> bool:
    """Return True when a session status is in the supplied final-status set."""

    status = session.get("status")
    status_value = getattr(status, "value", str(status))
    return status_value in final_status_values
