"""Operational logs in PostgreSQL, independent of debugger and billing state.

One dedicated writer owns ordered commits and optional outage journaling. Emit
normally only enqueues bounded chunks. A record larger than RAM uses bounded
producer backpressure, never truncation; exhaustion latches an explicit capture
failure and closes new dispatch outside logging.emit. See STORAGE_AND_LOGS.md.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg

import json_utils as json
from debugger_lease import InstanceLease
from debugger_spool import ZERO_HASH, DebuggerConflict, DebuggerPersistenceError, DebuggerSpool, Operation, validate_spool_path
from logging_policy import env_bool

LOG_JOURNAL_ID = 0x47534C4A  # GSLJ: cannot be opened as a debugger journal.


def log_schema(instance_id: str) -> str:
    """Only explicit installation identities may select an operational schema."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", instance_id):
        raise ValueError("LOG_INSTANCE_ID must be 1-40 lowercase letters/digits/underscores, starting with a letter")
    return "gransabio_logs_" + instance_id


@dataclass(frozen=True)
class LogSettings:
    """Independent destination and bounds; selecting PG never enables verbosity."""

    database_url: str = field(repr=False)
    instance_id: str
    spill: bool = True
    strict: bool = False
    ram_bytes: int = 64 * 1024 * 1024
    ram_seconds: float = 30.0
    spool_bytes: int = 1024 * 1024 * 1024
    min_free_bytes: int = 256 * 1024 * 1024
    spool_path: str = ""
    retention_days: int = 14

    def __post_init__(self):
        log_schema(self.instance_id)
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("LOG_DATABASE_URL must be an explicit PostgreSQL URI")
        if self.ram_bytes < 65536 or self.spool_bytes < 65536 or self.ram_seconds <= 0:
            raise ValueError("Log RAM/disk bounds must be at least 65536 bytes and RAM seconds positive")
        if self.min_free_bytes < 0 or self.retention_days < 0:
            raise ValueError("Log free-space reserve and retention cannot be negative")

    @classmethod
    def from_env(cls) -> LogSettings:
        """No fallback to debugger credentials or identity."""
        return cls(
            database_url=os.getenv("LOG_DATABASE_URL", ""), instance_id=os.getenv("LOG_INSTANCE_ID", ""),
            spill=env_bool("LOG_PG_SPILL", True), strict=env_bool("LOG_PG_STRICT"),
            ram_bytes=int(os.getenv("LOG_PG_RAM_BYTES", str(64 * 1024 * 1024))),
            ram_seconds=float(os.getenv("LOG_PG_RAM_SECONDS", "30")),
            spool_bytes=int(os.getenv("LOG_PG_SPOOL_BYTES", str(1024 * 1024 * 1024))),
            min_free_bytes=int(os.getenv("LOG_PG_MIN_FREE_BYTES", str(256 * 1024 * 1024))),
            spool_path=os.getenv("LOG_PG_SPOOL_PATH", ""), retention_days=int(os.getenv("LOG_RETENTION_DAYS", "14")),
        )


class LogCapture:
    """Thread-safe ingress and one writer; accepted operations are never evicted."""

    def __init__(self, settings: LogSettings, *, notice=None):
        self.settings = settings
        self.schema = log_schema(settings.instance_id)
        address = urlsplit(settings.database_url)
        self.target = hashlib.sha256(json.dumps(["logs", address.hostname, address.port or 5432,
                                                address.path, settings.instance_id]).encode()).hexdigest()
        base = Path(os.getenv("LOCALAPPDATA") or os.getenv("XDG_STATE_HOME") or str(Path.home() / ".local/state"))
        self.path = Path(settings.spool_path) if settings.spool_path else base / "GranSabio_LLM/logs/pending" / settings.instance_id / "capture.sqlite"
        validate_spool_path(self.path, Path(os.getenv("DEBUGGER_DB_PATH") or base / "GranSabio_LLM/debugger/debugger_history.db"))
        self._lease = InstanceLease(self.target)
        self._path_lease = InstanceLease(hashlib.sha256(str(self.path.resolve()).encode()).hexdigest())
        self.origin, self.owner_token = str(uuid4()), str(uuid4())
        self.store_id = self.expected_owner = None
        self.next_seq = self.ack_seq = 0
        self.last_hash = self.ack_hash = ZERO_HASH
        self.queue: deque[Operation] = deque()
        self.ram_bytes = self.replay_bytes = self.peak_ram_bytes = 0
        self.spool = None
        self.connection = None
        self._connected = False
        self.oldest_at = None
        self.state, self.reason = "buffering", None
        self.fatal = self.loss_unknown = self.capture_failed = False
        self.rejected_fragments = 0
        self._cv = threading.Condition(threading.RLock())
        self._capture_lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = None
        self._closing = self.closed = False
        self._close_deadline = 0.0
        self._retry_at = self._spill_retry_at = 0.0
        self._retry_count = 0
        self._last_notice = None
        self._next_retention = 0.0
        self._recovered_clean = False
        self._notice = notice

    def start(self) -> None:
        """Read local continuity before accepting messages; PG may remain offline."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main, name="gransabio-log-writer", daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("Operational log journal initialization did not finish within 10 seconds")
        if self.fatal:
            raise RuntimeError(self.reason)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            with self._cv:
                self.fatal = True
                self.state, self.reason = "blocked", "Log writer stopped: " + type(exc).__name__
                self._cv.notify_all()
            self._report()
        finally:
            self._ready.set()

    def capture(self, event: dict) -> None:
        """Retain exact text, including NUL; never throw into logging.emit.

        Each event is split into ordered, independently checksummed fragments.
        Capture never waits for storage from a logging caller. Exhausting the
        bounded queue latches an incomplete-capture failure and blocks new work.
        """
        if self._thread is not None and threading.current_thread() is self._thread:
            return  # Storage diagnostics must not recursively store themselves.
        try:
            with self._capture_lock:
                text = event["text"]
                width = max(64, min(32768, self.settings.ram_bytes // 512))
                parts = max(1, (len(text) + width - 1) // width)
                event_id = str(uuid4())
                metadata = {key: value for key, value in event.items() if key not in {"text", "seq"}}
                for part in range(parts):
                    data = {"event_id": event_id, "part": part, "parts": parts,
                            "event": {**metadata, "text": text[part * width:(part + 1) * width]}}
                    with self._cv:
                        op = Operation.create(self.next_seq + 1, data, self.last_hash)
                        if op.size > self.settings.ram_bytes // 4:
                            raise DebuggerPersistenceError("Log metadata exceeds RAM capacity; capture is incomplete")
                        if self.ram_bytes + op.size > self.settings.ram_bytes // 2:
                            raise DebuggerPersistenceError("Log capture capacity exhausted; new work is blocked")
                        if self.fatal or self._closing or self.closed:
                            raise DebuggerPersistenceError("Log capture is closed or conflicted; evidence is incomplete")
                        self.queue.append(op)
                        self.next_seq, self.last_hash = op.seq, op.chain_hash
                        self.ram_bytes += op.size
                        self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes + self.replay_bytes + op.size)
                        self.oldest_at = self.oldest_at or time.monotonic()
                        self._cv.notify_all()
        except Exception as exc:
            self.capture_failure(str(exc) if isinstance(exc, DebuggerPersistenceError) else "Log encoding failed: " + type(exc).__name__)

    def capture_failure(self, reason: str) -> None:
        """Latch a producer failure that logging's error handling must not hide."""
        with self._cv:
            self.capture_failed = True
            self.rejected_fragments += 1
            self.state, self.reason = "blocked", reason
            self._cv.notify_all()

    def admission_error(self) -> str | None:
        """A safe gate for /generate and provider dispatch, never inside emit."""
        with self._cv:
            if self.fatal or self.capture_failed or self.closed or self._closing or self.state == "blocked":
                return "Operational log capture is blocked; inspect console storage status"
            if self.settings.strict and not self._connected:
                return "Strict log persistence requires PostgreSQL before new work"
            if self.ram_bytes >= self.settings.ram_bytes * .375:
                return "Log RAM reserve is protecting calls already in flight"
            if self.spool and self.spool.pending_bytes >= self.settings.spool_bytes * .8:
                return "Log journal reserve is protecting calls already in flight"
            if not self.settings.spill and self.oldest_at and time.monotonic() - self.oldest_at >= self.settings.ram_seconds:
                return "Log RAM time limit reached and local spill is disabled"
            return None

    def status(self) -> dict:
        """No credentials, paths or payloads in operational status."""
        with self._cv:
            pending = self.next_seq - self.ack_seq
            return {"backend": "postgresql", "state": self.state, "reason": self.reason,
                    "instance_id": self.settings.instance_id, "origin": self.origin,
                    "accepted_sequence": self.next_seq, "persisted_sequence": self.ack_seq,
                    "pending_operations": pending, "ram_bytes": self.ram_bytes + self.replay_bytes,
                    "peak_ram_bytes": self.peak_ram_bytes, "local_bytes": self.spool.pending_bytes if self.spool else 0,
                    "oldest_pending_seconds": round(time.monotonic() - self.oldest_at, 3) if self.oldest_at else 0,
                    "capture_failed": self.capture_failed, "rejected_fragments": self.rejected_fragments,
                    "loss_unknown": self.loss_unknown, "spill_enabled": self.settings.spill,
                    "strict": self.settings.strict, "admission_blocked": self.admission_error() is not None,
                    "complete": pending == 0 and self._connected and not
                    (self.fatal or self.capture_failed or self.loss_unknown)}

    def _report(self) -> None:
        status = self.status()
        signature = (status["state"], status["admission_blocked"], status["capture_failed"], status["reason"])
        if signature == self._last_notice:
            return
        previous, self._last_notice = self._last_notice, signature
        # Healthy startup is routine. Recovery after an incident is not.
        if previous is None and self.state == "available" and not status["admission_blocked"]:
            return
        if self._notice:
            self._notice(f"{self.state}; pending={status['pending_operations']}; "
                         f"new_work={'paused' if status['admission_blocked'] else 'allowed'}; "
                         f"{self.reason or 'connection and capture recovered'}")

    def _failed(self, exc: BaseException) -> None:
        with self._cv:
            self.fatal = self.fatal or isinstance(exc, (DebuggerConflict, asyncpg.DataError, asyncpg.IntegrityConstraintViolationError,
                                                        asyncpg.InsufficientPrivilegeError, asyncpg.InvalidPasswordError))
            self.state = "blocked" if self.fatal or self.capture_failed else ("local" if self.spool else "buffering")
            # Driver exception strings may contain addresses, SQL or payloads.
            self.reason = str(exc) if isinstance(exc, DebuggerPersistenceError) else "PostgreSQL log write failed: " + type(exc).__name__
            self._retry_at = time.monotonic() + (1, 2, 5, 10, 30)[min(self._retry_count, 4)]
            self._retry_count += 1
            self._cv.notify_all()

    async def _disconnect(self) -> None:
        self._connected = False
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                await connection.close(timeout=2)
            except Exception:
                connection.terminate()

    async def _connect(self) -> None:
        connection = await asyncpg.connect(self.settings.database_url, timeout=3, command_timeout=30,
                                          server_settings={"application_name": "gransabio-logs:" + self.settings.instance_id})
        self.connection = connection
        if await connection.fetchval("SHOW fsync") != "on":
            raise DebuggerConflict("PostgreSQL fsync must be enabled for log durability")
        await connection.execute("SET synchronous_commit TO on")
        if not await connection.fetchval("SELECT pg_try_advisory_lock(hashtextextended(current_database() || $1,0))", self.schema):
            raise DebuggerConflict("Another writer owns LOG_INSTANCE_ID; use one identity per installation")
        async with connection.transaction():
            await connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            await connection.execute(f'SET search_path TO "{self.schema}"')
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL,
                    store_id TEXT NOT NULL, owner_token TEXT, clean_shutdown BOOLEAN NOT NULL DEFAULT TRUE,
                    loss_unknown BOOLEAN NOT NULL DEFAULT FALSE);
                CREATE TABLE IF NOT EXISTS sources (origin TEXT PRIMARY KEY, seq BIGINT NOT NULL, chain_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS records (
                    origin TEXT NOT NULL, seq BIGINT NOT NULL, event_id TEXT NOT NULL,
                    part INTEGER NOT NULL, parts INTEGER NOT NULL, observed_at TIMESTAMPTZ NOT NULL,
                    level TEXT, logger TEXT, session_id TEXT, project_id TEXT, phase TEXT,
                    payload BYTEA NOT NULL, chain_hash TEXT NOT NULL,
                    PRIMARY KEY(origin,seq), UNIQUE(origin,event_id,part));
                CREATE INDEX IF NOT EXISTS records_observed_at ON records(observed_at);
                CREATE INDEX IF NOT EXISTS records_session ON records(session_id,observed_at);
            """)
            await connection.execute("INSERT INTO metadata(id,version,store_id) VALUES(1,1,$1) ON CONFLICT DO NOTHING", str(uuid4()))
            row = await connection.fetchrow("SELECT * FROM metadata WHERE id=1 FOR UPDATE")
            if row["version"] != 1:
                raise DebuggerConflict("Unsupported PostgreSQL log schema version")
            if self.store_id and self.store_id != row["store_id"]:
                raise DebuggerConflict("PostgreSQL log store identity changed; journal preserved")
            remote = await connection.fetchrow("SELECT seq,chain_hash FROM sources WHERE origin=$1", self.origin)
            confirmed = remote and tuple(remote) == (self.next_seq, self.last_hash)
            allowed = (self.expected_owner, self.owner_token, self.spool.meta.get("previous_owner") if self.spool else None)
            if self.store_id and self.expected_owner and row["owner_token"] not in allowed and not confirmed:
                raise DebuggerConflict("Another log owner took this installation while disconnected")
            self.store_id = row["store_id"]
            clean_journal = self._recovered_clean and row["owner_token"] == self.expected_owner
            self.loss_unknown = self.loss_unknown or row["loss_unknown"] or bool(
                row["owner_token"] and row["owner_token"] != self.owner_token and
                not row["clean_shutdown"] and not clean_journal)
            if self.spool:
                await self.spool.set_meta(store_id=self.store_id, owner_token=self.owner_token, previous_owner=row["owner_token"])
            await connection.execute("UPDATE metadata SET owner_token=$1,clean_shutdown=FALSE,loss_unknown=$2 WHERE id=1",
                                     self.owner_token, self.loss_unknown)
            await connection.execute("INSERT INTO sources VALUES($1,0,$2) ON CONFLICT DO NOTHING", self.origin, ZERO_HASH)
        self.expected_owner = self.owner_token
        await self._reconcile(*(tuple(remote) if remote else (0, ZERO_HASH)))
        self._connected = True
        if not self.capture_failed:
            self.reason = None
            self.state = "recovering" if self.next_seq > self.ack_seq else "available"

    async def _reconcile(self, seq: int, digest: str) -> None:
        with self._cv:
            if seq == self.ack_seq:
                found = self.ack_hash
            elif self.ack_seq < seq <= self.next_seq:
                found = next((op.chain_hash for op in self.queue if op.seq == seq), None)
            else:
                raise DebuggerConflict("Remote log prefix moved outside the retained capture range")
        if found is None and self.spool:
            async with self.spool.db.execute("SELECT chain_hash FROM operations WHERE seq=?", (seq,)) as cursor:
                row = await cursor.fetchone()
                found = row[0] if row else None
        if found != digest:
            raise DebuggerConflict("Remote log prefix differs from retained content")
        await self._acknowledge(seq, digest)

    async def _acknowledge(self, seq: int, digest: str) -> None:
        if self.spool and self.spool.meta["ack_seq"] < seq:
            await self.spool.acknowledge(seq, digest)
        if self.spool and not self.spool.pending_count:
            await self.spool.close()
            self.path.unlink()
            self.spool = None
        with self._cv:
            if seq > self.ack_seq:
                self._retry_count = 0
            while self.queue and self.queue[0].seq <= seq:
                self.ram_bytes -= self.queue.popleft().size
            self.ack_seq, self.ack_hash = seq, digest
            if self.ack_seq == self.next_seq:
                self.oldest_at = None
            self._cv.notify_all()

    async def _spill(self) -> None:
        with self._cv:
            batch = list(self.queue)
        if not batch or not self.settings.spill:
            return
        if self.spool is None:
            spool = DebuggerSpool(self.path, self.settings.spool_bytes, self.settings.min_free_bytes,
                                  application_id=LOG_JOURNAL_ID)
            await spool.open({"instance_id": self.settings.instance_id, "target": self.target,
                                   "origin": self.origin, "store_id": self.store_id, "owner_token": self.owner_token,
                                   "ack_seq": self.ack_seq, "ack_hash": self.ack_hash, "clean_shutdown": False})
            self.spool = spool
        await self.spool.append(batch)
        with self._cv:
            for op in batch:
                assert self.queue.popleft() is op
                self.ram_bytes -= op.size
            self._cv.notify_all()
        self.state = "blocked" if self.capture_failed or self.fatal else "local"

    async def _apply(self, batch: list[Operation]) -> None:
        seq, digest = self.ack_seq, self.ack_hash
        async with self.connection.transaction():
            receipt = await self.connection.fetchval(
                "UPDATE sources AS source SET seq=$4,chain_hash=$5 FROM metadata AS owner "
                "WHERE source.origin=$1 AND source.seq=$2 AND source.chain_hash=$3 "
                "AND owner.id=1 AND owner.owner_token=$6 RETURNING source.seq",
                self.origin, seq, digest, batch[-1].seq, batch[-1].chain_hash, self.owner_token)
            if receipt is None:
                raise DebuggerConflict("Log writer ownership or confirmed prefix changed")
            rows = []
            for op in batch:
                op.verify()
                if op.seq != seq + 1 or op.previous_hash != digest:
                    raise DebuggerConflict("Log operations contain a sequence gap")
                data = json.loads(op.payload)
                event, context = data["event"], data["event"].get("context") or {}
                rows.append((self.origin, op.seq, data["event_id"], data["part"], data["parts"],
                             datetime.fromisoformat(event["created_at"]), event.get("level"), event.get("logger"),
                             context.get("session_id"), context.get("project_id"), context.get("phase"), op.payload, op.chain_hash))
                seq, digest = op.seq, op.chain_hash
            await self.connection.executemany("INSERT INTO records VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)", rows)
        await self._acknowledge(seq, digest)

    async def _retention(self) -> None:
        if not self.settings.retention_days or time.monotonic() < self._next_retention or self.next_seq != self.ack_seq:
            return
        # Receipt rows survive retention. Never expire unconfirmed or replaying capture.
        result = await self.connection.execute(
            "WITH expired AS (SELECT origin,event_id FROM records "
            "WHERE observed_at < clock_timestamp() - $1 * interval '1 day' "
            "ORDER BY observed_at LIMIT 1000) "
            "DELETE FROM records AS record USING expired "
            "WHERE record.origin=expired.origin AND record.event_id=expired.event_id",
            self.settings.retention_days)
        self._next_retention = time.monotonic() + (0.05 if result != "DELETE 0" else 3600)

    async def _run(self) -> None:
        try:
            self._lease.acquire()
            self._path_lease.acquire()
            if self.path.exists():
                self.spool = DebuggerSpool(self.path, self.settings.spool_bytes, self.settings.min_free_bytes,
                                           application_id=LOG_JOURNAL_ID)
                await self.spool.open()
                meta = self.spool.meta
                if meta["target"] != self.target or meta["instance_id"] != self.settings.instance_id:
                    raise DebuggerConflict("Log journal belongs to another target; original preserved")
                self.origin, self.store_id, self.expected_owner = meta["origin"], meta["store_id"], meta["owner_token"]
                self.ack_seq, self.ack_hash = meta["ack_seq"], meta["ack_hash"]
                self.next_seq, self.last_hash = await self.spool.last()
                self._recovered_clean = meta.get("clean_shutdown", False)
                self.loss_unknown = not self._recovered_clean
                await self.spool.set_meta(clean_shutdown=False)
                if self.next_seq > self.ack_seq:
                    self.oldest_at = time.monotonic()
            self._ready.set()
            while not self._closing or (self.next_seq != self.ack_seq and time.monotonic() < self._close_deadline):
                try:
                    if self.connection is None and not self.fatal and time.monotonic() >= self._retry_at:
                        await self._connect()
                    if self.connection is not None and not self.fatal:
                        if self.spool and self.spool.pending_count:
                            batch = await self.spool.batch(max_bytes=self.settings.ram_bytes // 8)
                            self.replay_bytes = sum(op.size for op in batch)
                        else:
                            with self._cv:
                                batch, size = [], 0
                                for op in self.queue:
                                    batch.append(op)
                                    size += op.size
                                    if len(batch) >= 100 or size >= self.settings.ram_bytes // 8:
                                        break
                        self.peak_ram_bytes = max(self.peak_ram_bytes, self.ram_bytes + self.replay_bytes)
                        if batch:
                            await self._apply(batch)
                        batch = []
                        self.replay_bytes = 0
                        if not self.capture_failed:
                            self.state = "recovering" if self.spool else "available"
                            self.reason = None
                        await self._retention()
                except Exception as exc:
                    self.replay_bytes = 0
                    await self._disconnect()
                    self._failed(exc)
                if (self.queue and self.settings.spill and self.connection is None and not self.fatal
                        and time.monotonic() >= self._spill_retry_at and
                        (self.ram_bytes >= self.settings.ram_bytes * .3 or self._closing or
                         self.oldest_at and time.monotonic() - self.oldest_at >= self.settings.ram_seconds)):
                    try:
                        await self._spill()
                    except Exception as exc:
                        self._failed(exc)
                        self.state = "blocked"
                        self._spill_retry_at = time.monotonic() + 5
                self._report()
                # A healthy backlog already yields at each DB operation; do not
                # impose another fixed delay between acknowledged batches.
                await asyncio.sleep(0 if self.queue and self.connection is not None else .05)
            if self.queue and self.settings.spill:
                await self._spill()
            if self.spool:
                await self.spool.set_meta(clean_shutdown=not self.queue and not self.capture_failed)
            if self.connection is not None and self.next_seq == self.ack_seq and not self.capture_failed:
                await self.connection.execute("UPDATE metadata SET clean_shutdown=TRUE WHERE id=1 AND owner_token=$1", self.owner_token)
            if self.queue:
                self.loss_unknown = True
                self.reason = "Shutdown leaves volatile log fragments unpersisted"
                self.state = "blocked"
                self._report()
        finally:
            if self.spool:
                await self.spool.close()
            await self._disconnect()
            self._path_lease.close()
            self._lease.close()
            with self._cv:
                self.closed = True
                self._cv.notify_all()

    def flush(self, timeout: float = 5) -> None:
        """Wait for a stable captured prefix, never report partial logs as complete."""
        deadline = time.monotonic() + timeout
        with self._cv:
            cut = self.next_seq
            while self.ack_seq < cut:
                if self.fatal or self.closed or time.monotonic() >= deadline:
                    raise DebuggerPersistenceError("Operational logs have not reached PostgreSQL")
                self._cv.wait(min(.1, max(0, deadline - time.monotonic())))
            if self.capture_failed:
                raise DebuggerPersistenceError("Operational log capture is incomplete")

    def close(self, timeout: float = 5) -> None:
        """Bounded graceful drain, then journal pending capture if enabled."""
        if self.closed or self._thread is None:
            return
        with self._cv:
            self._closing = True
            self._close_deadline = time.monotonic() + timeout
            self._cv.notify_all()
        self._thread.join(timeout + 12)
        if self._thread.is_alive():
            raise RuntimeError("Log writer has not closed; volatile capture may remain")
