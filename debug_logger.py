"""
Debug logger for Gran Sabio LLM Engine.

Persists full request/response timelines to SQLite so developers can inspect
each session in chronological order via the /debugger interface.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

import json_utils as json

logger = logging.getLogger(__name__)

# Piggyback cleanup configuration
CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour between opportunistic cleanups
CLEANUP_RETENTION_DAYS = 7       # Delete sessions older than 7 days


class DebugLogger:
    """Centralized persistence for debugging sessions."""

    def __init__(self, *, enabled: bool, db_path: str):
        self.enabled = enabled
        self.db_path = Path(db_path)
        self._pool: Optional[aiosqlite.Connection] = None
        self._init_lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._event_counters: Dict[str, int] = {}
        self._last_cleanup_ts: float = 0.0
        self._cleanup_running: bool = False

    async def initialize(self) -> None:
        """Open SQLite connection and create schema if needed."""
        if not self.enabled:
            return

        async with self._init_lock:
            if self._pool is not None:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._pool = await aiosqlite.connect(self.db_path.as_posix())
            await self._pool.execute("PRAGMA journal_mode=DELETE")
            await self._pool.execute("PRAGMA synchronous=NORMAL")
            await self._pool.execute("PRAGMA foreign_keys=ON")
            await self._create_schema()
            await self._pool.commit()
            logger.info("Debug logger initialized with database at %s", self.db_path)

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _create_schema(self) -> None:
        """Create tables for session storage."""
        if self._pool is None:
            return

        await self._pool.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                status TEXT,
                request_json TEXT,
                request_name TEXT,
                preflight_json TEXT,
                final_json TEXT,
                usage_json TEXT,
                notes TEXT,
                project_id TEXT
            );

            CREATE TABLE IF NOT EXISTS session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_order INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_session_events_lookup
                ON session_events(session_id, event_order);

            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(updated_at DESC);

            CREATE INDEX IF NOT EXISTS idx_sessions_created
                ON sessions(created_at);
            """
        )
        await self._ensure_project_id_column()

    async def _ensure_project_id_column(self) -> None:
        """Ensure the sessions table contains project_id and request_name columns."""
        if self._pool is None:
            return

        async with self._pool.execute("PRAGMA table_info(sessions)") as cursor:
            columns = await cursor.fetchall()

        col_names = {col[1] for col in columns}

        if "project_id" not in col_names:
            await self._pool.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")

        if "request_name" not in col_names:
            await self._pool.execute("ALTER TABLE sessions ADD COLUMN request_name TEXT")

        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)"
        )
        await self._pool.commit()

    def _serialize(self, payload: Any) -> str:
        """Serialize payload to JSON string."""
        try:
            return json.dumps(payload, ensure_ascii=True)
        except Exception:
            logger.exception("Failed to serialize payload for debug logger")
            return "{}"

    async def _maybe_piggyback_cleanup(self):
        """Run cleanup opportunistically if enough time has passed."""
        now = time.monotonic()
        if self._cleanup_running or (now - self._last_cleanup_ts) < CLEANUP_INTERVAL_SECONDS:
            return
        self._cleanup_running = True
        self._last_cleanup_ts = now
        try:
            result = await self.cleanup_old_sessions(retention_days=CLEANUP_RETENTION_DAYS)
            deleted = result.get("deleted_sessions", 0)
            if deleted > 0:
                logger.info("Piggyback cleanup: removed %d old debug sessions", deleted)
        except Exception:
            logger.exception("Piggyback cleanup failed")
        finally:
            self._cleanup_running = False

    async def record_session_start(
        self,
        session_id: str,
        *,
        request_payload: Any,
        preflight_payload: Optional[Any] = None,
        status: str = "initializing",
        project_id: Optional[str] = None,
    ) -> None:
        """Persist initial session data."""
        if not self.enabled or self._pool is None:
            return

        # DEBUG: Log project_id being stored
        logger.info(f"DEBUG_LOGGER: Recording session {session_id[:8]}... with project_id: {project_id if project_id else 'NULL'}")

        # Extract request_name for denormalized column
        req_name = None
        if isinstance(request_payload, dict):
            req_name = request_payload.get("request_name")
        elif hasattr(request_payload, "request_name"):
            req_name = request_payload.request_name

        await self._pool.execute(
            """
            INSERT INTO sessions
            (session_id, created_at, updated_at, status, request_json, request_name, preflight_json, project_id)
            VALUES
            (?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = datetime('now'),
                status = excluded.status,
                request_json = excluded.request_json,
                request_name = excluded.request_name,
                preflight_json = excluded.preflight_json,
                project_id = COALESCE(excluded.project_id, sessions.project_id)
            """,
            (
                session_id,
                status,
                self._serialize(request_payload),
                req_name,
                self._serialize(preflight_payload) if preflight_payload is not None else None,
                project_id,
            ),
        )
        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def update_session_status(
        self,
        session_id: str,
        *,
        status: Optional[str] = None,
        final_payload: Optional[Any] = None,
    ) -> None:
        """Update stored session status and optional final payload."""
        if not self.enabled or self._pool is None:
            return

        await self._pool.execute(
            """
            UPDATE sessions
            SET
                status = COALESCE(?, status),
                final_json = COALESCE(?, final_json),
                updated_at = datetime('now')
            WHERE session_id = ?
            """,
            (
                status,
                self._serialize(final_payload) if final_payload is not None else None,
                session_id,
            ),
        )
        await self._pool.commit()

    async def record_usage_summary(self, session_id: str, usage_payload: Any) -> None:
        """Store aggregated usage metrics for a session."""
        if not self.enabled or self._pool is None:
            return

        await self._pool.execute(
            """
            UPDATE sessions
            SET usage_json = ?, updated_at = datetime('now')
            WHERE session_id = ?
            """,
            (self._serialize(usage_payload), session_id),
        )
        await self._pool.commit()

    async def record_event(
        self,
        session_id: str,
        *,
        event_type: str,
        payload: Any,
    ) -> None:
        """Insert chronological event for the session."""
        if not self.enabled or self._pool is None:
            return

        event_order = await self._next_event_order(session_id)
        await self._pool.execute(
            """
            INSERT INTO session_events (session_id, event_order, event_type, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                event_order,
                event_type,
                self._serialize(payload),
            ),
        )
        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def _next_event_order(self, session_id: str) -> int:
        """Compute the next chronological order value for the session."""
        async with self._event_lock:
            next_value = self._event_counters.get(session_id)
            if next_value is None:
                next_value = await self._lookup_existing_event_order(session_id)
            next_value += 1
            self._event_counters[session_id] = next_value
            return next_value

    async def _lookup_existing_event_order(self, session_id: str) -> int:
        """Fetch the latest event order from the database."""
        if self._pool is None:
            return 0

        async with self._pool.execute(
            "SELECT COALESCE(MAX(event_order), 0) FROM session_events WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent sessions ordered by last update descending."""
        if not self.enabled or self._pool is None:
            return []

        where_clause = ""
        params: List[Any] = []
        if project_id:
            where_clause = "WHERE project_id = ?"
            params.append(project_id)

        query = f"""
            SELECT session_id, created_at, updated_at, status, project_id, request_name
            FROM sessions
            {where_clause}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """

        params.extend([limit, offset])

        async with self._pool.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "status": row[3],
                "project_id": row[4],
                "request_name": row[5],
            }
            for row in rows
        ]

    async def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a session."""
        if not self.enabled or self._pool is None:
            return None

        async with self._pool.execute(
            """
            SELECT session_id, created_at, updated_at, status,
                   request_json, preflight_json, final_json, usage_json, notes,
                   project_id
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "session_id": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "status": row[3],
            "request_json": row[4],
            "preflight_json": row[5],
            "final_json": row[6],
            "usage_json": row[7],
            "notes": row[8],
            "project_id": row[9],
        }

    async def get_session_events(
        self,
        session_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch chronological events for display."""
        if not self.enabled or self._pool is None:
            return []

        async with self._pool.execute(
            """
            SELECT event_order, event_type, payload_json, created_at
            FROM session_events
            WHERE session_id = ?
            ORDER BY event_order ASC
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "event_order": row[0],
                "event_type": row[1],
                "payload_json": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    async def purge_session(self, session_id: str) -> None:
        """Delete all data for a specific session."""
        if not self.enabled or self._pool is None:
            return

        await self._pool.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._pool.commit()
        async with self._event_lock:
            self._event_counters.pop(session_id, None)

    async def cleanup_old_sessions(self, retention_days: int = 14) -> dict:
        """
        Delete sessions older than retention_days.

        Args:
            retention_days: Number of days to retain sessions (default: 14)

        Returns:
            Dictionary with cleanup statistics
        """
        if not self.enabled or self._pool is None:
            return {"status": "disabled", "deleted_sessions": 0, "deleted_events": 0}

        try:
            # Compute cutoff once in Python (ISO-8601 sorts correctly as text)
            cutoff_str = (datetime.utcnow() - timedelta(days=retention_days)).strftime('%Y-%m-%d %H:%M:%S')

            # Count before deletion for stats
            async with self._pool.execute(
                "SELECT COUNT(*) FROM sessions WHERE created_at < ?",
                (cutoff_str,)
            ) as cursor:
                row = await cursor.fetchone()
                sessions_to_delete = row[0] if row else 0

            async with self._pool.execute(
                """SELECT COUNT(*) FROM session_events
                   WHERE session_id IN (
                       SELECT session_id FROM sessions
                       WHERE created_at < ?
                   )""",
                (cutoff_str,)
            ) as cursor:
                row = await cursor.fetchone()
                events_to_delete = row[0] if row else 0

            if sessions_to_delete == 0:
                return {
                    "status": "ok",
                    "deleted_sessions": 0,
                    "deleted_events": 0,
                    "message": "No old sessions to clean up"
                }

            # Delete old events first (foreign key constraint)
            await self._pool.execute(
                """DELETE FROM session_events
                   WHERE session_id IN (
                       SELECT session_id FROM sessions
                       WHERE created_at < ?
                   )""",
                (cutoff_str,)
            )

            # Delete old sessions
            await self._pool.execute(
                "DELETE FROM sessions WHERE created_at < ?",
                (cutoff_str,)
            )

            await self._pool.commit()

            # VACUUM to reclaim disk space
            await self._pool.execute("VACUUM")

            # Clear event counters for deleted sessions
            async with self._event_lock:
                self._event_counters.clear()

            logger.info(
                "Debug logger cleanup: deleted %d sessions and %d events older than %d days",
                sessions_to_delete, events_to_delete, retention_days
            )

            return {
                "status": "ok",
                "deleted_sessions": sessions_to_delete,
                "deleted_events": events_to_delete,
                "retention_days": retention_days
            }

        except Exception as e:
            logger.error("Failed to cleanup old sessions: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "deleted_sessions": 0,
                "deleted_events": 0
            }


_shared_debug_logger: Optional[DebugLogger] = None
_debug_logger_lock = asyncio.Lock()


async def initialize_debug_logger(enabled: bool, db_path: str) -> None:
    """Initialize shared debug logger instance."""
    logger_instance = await get_debug_logger(enabled=enabled, db_path=db_path)
    await logger_instance.initialize()


async def shutdown_debug_logger() -> None:
    """Close shared debug logger."""
    global _shared_debug_logger
    if _shared_debug_logger is not None:
        await _shared_debug_logger.close()
        _shared_debug_logger = None


async def get_debug_logger(*, enabled: bool, db_path: str) -> DebugLogger:
    """Return shared debug logger (creating if needed)."""
    global _shared_debug_logger

    if _shared_debug_logger is not None:
        return _shared_debug_logger

    async with _debug_logger_lock:
        if _shared_debug_logger is None:
            _shared_debug_logger = DebugLogger(enabled=enabled, db_path=db_path)
    return _shared_debug_logger
