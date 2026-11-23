"""Ordered PostgreSQL capture: bounded RAM normally, local journal only on outage.

Acceptance transfers ownership to this queue; it is not a durability receipt.
One task applies both live and recovered operations. No provider is ever invoked.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg

import json_utils as json
from debugger_lease import InstanceLease
from debugger_payload import CaptureField, apply_fields, estimate_size
from debugger_spool import ZERO_HASH, DebuggerConflict, DebuggerPersistenceError, DebuggerSpool, Operation, validate_spool_path

logger = logging.getLogger(__name__)
applying_operation: ContextVar[bool] = ContextVar("debugger_applying", default=False)


@dataclass(frozen=True)
class PersistencePolicy:
    """Capture bounds; age triggers spill/backpressure and never deletes data."""

    spill: bool = True
    strict: bool = False
    ram_bytes: int = 64 * 1024 * 1024
    ram_seconds: float = 30.0
    spool_bytes: int = 1024 * 1024 * 1024
    min_free_bytes: int = 256 * 1024 * 1024
    spool_path: str = ""
    read_timeout: float = 2.0

    def __post_init__(self):
        if self.ram_bytes < 4096 or self.spool_bytes < 4096 or self.ram_seconds <= 0 or self.read_timeout <= 0:
            raise ValueError("Debugger persistence limits must be positive (RAM/disk at least 4096 bytes)")
        if self.min_free_bytes < 0:
            raise ValueError("Debugger free-space reserve cannot be negative")


@dataclass(frozen=True, slots=True)
class TraceAcceptance:
    """Logical trace offset accepted by RAM; never claim remote persistence here."""

    chars: int
    sequence: int
    durability: str = "volatile"


class PostgresCapture:
    """One owner of admission, sequence identity, spill, replay and PG connection."""

    def __init__(self, owner: Any, policy: PersistencePolicy):
        from debugger_postgres import debugger_schema

        debugger_schema(owner.instance_id)
        self.owner = owner
        self.policy = policy
        self.path = Path(policy.spool_path) if policy.spool_path else owner.db_path.parent / "pending" / owner.instance_id / "capture.sqlite"
        validate_spool_path(self.path, owner.db_path)
        address = urlsplit(owner.database_url)
        self.target = hashlib.sha256(json.dumps([address.hostname, address.port or 5432, address.path, owner.instance_id]).encode()).hexdigest()
        self.lease = InstanceLease(self.target)
        self.origin = str(uuid4())
        self.owner_token = str(uuid4())
        self.expected_owner: str | None = None
        self.store_id: str | None = None
        self.next_seq = 0
        self.last_hash = ZERO_HASH
        self.ack_seq = 0
        self.ack_hash = ZERO_HASH
        self.queue: deque[Operation] = deque()
        self.ram_bytes = 0
        self.external_bytes = 0
        self.external_pending_bytes = 0
        self._encoding_bytes = 0
        self._replay_bytes = 0
        self.oldest_at: float | None = None
        self.spool: DebuggerSpool | None = None
        self.state = "buffering"
        self.reason: str | None = None
        self.fatal = False
        self.loss_unknown = False
        self.closed = False
        self._closing = False
        self._wake = asyncio.Event()
        self._progress = asyncio.Condition()
        self._admission_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._attempt = 0
        self.next_retry_at = 0.0
        self._last_warning = 0.0
        self._peak_ram = 0
        self._connected_at = 0.0
        self.flush_external = None

    async def start(self) -> None:
        """Recover an existing journal before accepting new operations."""
        try:
            self.lease.acquire()
        except DebuggerConflict as exc:
            self._failed(exc, fatal=True)
            return
        if self.path.exists():
            self.spool = DebuggerSpool(self.path, self.policy.spool_bytes, self.policy.min_free_bytes)
            try:
                await self.spool.open()
                meta = self.spool.meta
                if meta["target"] != self.target or meta["instance_id"] != self.owner.instance_id:
                    raise DebuggerConflict("Journal belongs to another target/installation; original preserved")
                self.origin = meta["origin"]
                self.store_id = meta["store_id"]
                self.expected_owner = meta["owner_token"]
                self.ack_seq, self.ack_hash = meta["ack_seq"], meta["ack_hash"]
                self.next_seq, self.last_hash = await self.spool.last()
                self.oldest_at = time.monotonic() - max(0, time.time()-meta.get("pending_since", time.time()))
                # A recovered prefix cannot prove there was no later RAM tail,
                # including a boot that never managed to contact PostgreSQL.
                self.loss_unknown = True
            except Exception as exc:
                self._failed(exc, fatal=True)
                return
        try:
            await self._connect()
        except Exception as exc:
            self._failed(exc)
        self._task = asyncio.create_task(self._run(), name=f"debugger-capture:{self.owner.instance_id}")

    def _failed(self, exc: BaseException, *, fatal: bool = False) -> None:
        from debugger_postgres import DebuggerSetupError, postgres_failure_reason
        self.reason = postgres_failure_reason(exc)
        self.fatal = fatal or isinstance(exc, (DebuggerConflict, DebuggerSetupError, asyncpg.DataError, asyncpg.IntegrityConstraintViolationError))
        self.state = "blocked" if self.fatal else ("local" if self.spool else "buffering")
        steps = (1, 2, 5, 10, 30, 60)
        delay = steps[min(self._attempt, len(steps)-1)]
        self._attempt += 1
        self.next_retry_at = time.monotonic() + delay * random.uniform(.8, 1.2)
        logger.warning("Debugger %s: %s; pending capture is not complete remote evidence", self.state, self.reason)

    async def _disconnect(self) -> None:
        pool, self.owner._pool = self.owner._pool, None
        if pool is not None:
            try:
                await pool.close()
            except Exception:
                pool.connection.terminate()

    async def _connect(self) -> None:
        try:
            await self.owner._connect_postgres()
            pool = self.owner._pool
            async with pool.operation():
                row = await pool.connection.fetchrow("SELECT store_id,owner_token,clean_shutdown,loss_unknown FROM debugger_metadata WHERE instance_id=$1 FOR UPDATE", self.owner.instance_id)
                store_id, remote_owner, clean_shutdown, previous_loss = row
                if self.store_id and self.store_id != store_id:
                    raise DebuggerConflict("PostgreSQL store identity changed; journal preserved")
                remote = await pool.connection.fetchrow("SELECT seq,chain_hash FROM debugger_sources WHERE origin=$1", self.origin)
                already_confirmed = self.spool and remote and remote[0] == self.next_seq and remote[1] == self.last_hash
                accepted_owners = (self.expected_owner, self.owner_token, self.spool.meta.get("previous_owner") if self.spool else None)
                # An offline first start has a proposed local owner but has not
                # observed any PG store/owner yet. Bind it exactly as a fresh
                # start; enforce the fence once a remote store has been observed.
                if self.store_id and self.expected_owner and remote_owner not in accepted_owners and not already_confirmed:
                    raise DebuggerConflict("Another owner took this installation while disconnected")
                self.store_id = store_id
                self.loss_unknown = self.loss_unknown or previous_loss or bool(remote_owner and remote_owner != self.owner_token and not clean_shutdown)
                # Persist a proposed handoff before changing the remote fence.
                if self.spool:
                    await self.spool.set_meta(store_id=store_id, owner_token=self.owner_token, previous_owner=remote_owner)
                await pool.connection.execute("UPDATE debugger_metadata SET owner_token=$1,clean_shutdown=FALSE,loss_unknown=$3 WHERE instance_id=$2", self.owner_token, self.owner.instance_id, self.loss_unknown)
                await pool.connection.execute("INSERT INTO debugger_sources(origin,seq,chain_hash) VALUES($1,0,$2) ON CONFLICT DO NOTHING", self.origin, ZERO_HASH)
                remote = remote or (0, ZERO_HASH)
                if remote[0] < self.ack_seq:
                    raise DebuggerConflict("PostgreSQL confirmed prefix moved backwards; no automatic repair")
            self.expected_owner = self.owner_token
            await self._reconcile(remote[0], remote[1])
            await self._release_empty_spool()
            self.reason = None
            self._connected_at = time.monotonic()
            self.state = "recovering" if self.pending_count else "available"
        except BaseException:
            await self._disconnect()
            raise

    async def _reconcile(self, seq: int, chain_hash: str) -> None:
        """A lost COMMIT reply may put PG ahead; prove the retained prefix first."""
        if seq == self.ack_seq:
            if chain_hash != self.ack_hash:
                raise DebuggerConflict("Remote prefix checksum differs")
        else:
            if not self.ack_seq < seq <= self.next_seq:
                raise DebuggerConflict("Remote prefix is outside the retained capture range")
            found = next((op.chain_hash for op in self.queue if op.seq == seq), None)
            if found is None and self.spool:
                async with self.spool.db.execute("SELECT chain_hash FROM operations WHERE seq=?", (seq,)) as cursor:
                    row = await cursor.fetchone()
                    found = row[0] if row else None
            if found != chain_hash:
                raise DebuggerConflict("Remote prefix does not match retained content")
        if self.spool and self.spool.meta["ack_seq"] < seq:
            await self.spool.acknowledge(seq, chain_hash)
        while self.queue and self.queue[0].seq <= seq:
            self.ram_bytes -= self.queue.popleft().size
        self.ack_seq, self.ack_hash = seq, chain_hash
        async with self._progress:
            self._progress.notify_all()

    @property
    def pending_count(self) -> int:
        return len(self.queue) + (self.spool.pending_count if self.spool else 0)

    async def _release_empty_spool(self) -> None:
        if self.spool and not self.spool.pending_count:
            await self.spool.close()
            self.spool = None
            self.path.unlink()

    async def reserve_external(self, amount: int, *, pending_bytes: int | None = None) -> None:
        """Account for coalescer/bus buffers before retaining another stream chunk."""
        deadline = time.monotonic() + self.policy.ram_seconds
        while self.external_bytes + self.ram_bytes + self._encoding_bytes + amount > self._queue_limit():
            if self.fatal or time.monotonic() >= deadline:
                self.state = "blocked"
                raise DebuggerPersistenceError("Debugger stream buffer exhausted; new dispatch paused")
            self._wake.set()
            async with self._progress:
                try:
                    await asyncio.wait_for(self._progress.wait(), .1)
                except TimeoutError:
                    pass
        self.external_bytes += amount
        pending = amount if pending_bytes is None else pending_bytes
        self.external_pending_bytes += pending
        if pending:
            self.oldest_at = self.oldest_at or time.monotonic()
        self._peak_ram = max(self._peak_ram, self.external_bytes + self.ram_bytes + self._encoding_bytes + self._replay_bytes)

    async def release_external(self, amount: int, *, pending_bytes: int | None = None) -> None:
        self.external_bytes -= amount
        self.external_pending_bytes -= amount if pending_bytes is None else pending_bytes
        assert self.external_bytes >= 0
        assert self.external_pending_bytes >= 0
        if not self.pending_count and not self.external_pending_bytes:
            self.oldest_at = None
        async with self._progress:
            self._progress.notify_all()

    def status(self) -> dict[str, Any]:
        """Safe operational status, without paths, connection data or payloads."""
        age = max(0.0, time.monotonic()-self.oldest_at) if self.oldest_at else 0.0
        return {"state": self.state, "pending_operations": self.pending_count,
                "ram_bytes": self.ram_bytes + self.external_bytes + self._encoding_bytes + self._replay_bytes, "peak_ram_bytes": self._peak_ram,
                "local_bytes": self.spool.pending_bytes if self.spool else 0,
                "buffered_stream_bytes": self.external_pending_bytes,
                "oldest_pending_seconds": round(age, 3), "accepted_sequence": self.next_seq,
                "persisted_sequence": self.ack_seq, "reason": self.reason,
                "next_retry_seconds": round(max(0.0, self.next_retry_at-time.monotonic()), 2),
                "spill_enabled": self.policy.spill, "strict": self.policy.strict,
                "loss_unknown": self.loss_unknown, "complete": self.pending_count == 0 and self.external_pending_bytes == 0 and self._encoding_bytes == 0 and self.owner._pool is not None and not self.fatal}

    def require_admission(self) -> None:
        """Stop new dispatch before capture capacity or the explicit strict gate is lost."""
        status = self.status()
        used = self.ram_bytes + self.external_bytes + self._encoding_bytes
        if self.closed or self._closing or self.fatal or self.state == "blocked":
            raise DebuggerPersistenceError("Debugger capture blocked; new provider calls are paused")
        if self.policy.strict and (self.owner._pool is None or self.reason):
            raise DebuggerPersistenceError("Strict debugger mode requires PostgreSQL before new work")
        if used >= self.policy.ram_bytes * .75:
            raise DebuggerPersistenceError("Debugger RAM reserve is protecting calls already in flight")
        if not self.policy.spill and status["oldest_pending_seconds"] >= self.policy.ram_seconds:
            raise DebuggerPersistenceError("Debugger RAM time limit reached; disk fallback is disabled")
        if self.spool and self.spool.pending_bytes >= self.policy.spool_bytes * .8:
            raise DebuggerPersistenceError("Debugger journal reserve is protecting calls already in flight")

    async def wait_admission(self) -> None:
        """Let a healthy writer drain a burst before evaluating the dispatch gate."""
        deadline = time.monotonic() + self.policy.ram_seconds
        while (self.owner._pool is not None and not self.fatal and not self.closed
               and self.ram_bytes + self.external_bytes + self._encoding_bytes >= self.policy.ram_bytes * .75):
            if time.monotonic() >= deadline:
                raise DebuggerPersistenceError("Debugger capacity did not drain before the admission deadline")
            self._wake.set()
            async with self._progress:
                try:
                    await asyncio.wait_for(self._progress.wait(), .1)
                except TimeoutError:
                    pass
        self.require_admission()

    async def maintenance(self, operation):
        """Delete only after a confirmed cut, excluding new capture during deletion."""
        deadline = time.monotonic() + self.policy.read_timeout
        while True:
            await self.flush(max(.01, deadline-time.monotonic()))
            async with self._admission_lock:
                if not self.pending_count:
                    try:
                        async with self.owner._pool.operation():
                            return await operation()
                    except DebuggerPersistenceError:
                        raise
                    except Exception as exc:
                        self._failed(exc)
                        await self._disconnect()
                        self._wake.set()
                        raise DebuggerPersistenceError("Debugger maintenance failed; pending capture preserved") from exc
            if time.monotonic() >= deadline:
                raise DebuggerPersistenceError("Debugger maintenance requires a drained capture queue")

    @asynccontextmanager
    async def _capture_slot(self, estimate: int):
        """Wait outside the encoder lock so a large caller cannot stall small traces."""
        deadline = time.monotonic() + self.policy.ram_seconds
        while True:
            async with self._admission_lock:
                if self.closed or self._closing or self.fatal:
                    raise DebuggerPersistenceError("Debugger cannot accept capture; evidence is incomplete")
                required = min(estimate, self._queue_limit() // 2)
                if self.ram_bytes + self.external_bytes + required <= self._queue_limit():
                    self._encoding_bytes = required
                    self._peak_ram = max(self._peak_ram, self.ram_bytes+self.external_bytes+required+self._replay_bytes)
                    try:
                        # Yield before acceptance, with only one encoder reserved.
                        await asyncio.sleep(0)
                        yield
                    finally:
                        self._encoding_bytes = 0
                    return
            self._wake.set()
            if self.owner._pool is None and time.monotonic() >= deadline:
                raise DebuggerPersistenceError("Debugger capture capacity is exhausted")
            async with self._progress:
                try:
                    await asyncio.wait_for(self._progress.wait(), .1)
                except TimeoutError:
                    pass

    def _queue_limit(self) -> int:
        # During replay reserve scratch for the oldest local record, so new
        # producers cannot fill all RAM and prevent that prefix from draining.
        reserve = self.policy.ram_bytes // 2 if self.spool and self.spool.pending_count else 0
        return self.policy.ram_bytes - max(reserve, self._replay_bytes)

    async def submit(self, kind: str, arguments: dict[str, Any]) -> int:
        """Transfer one immutable operation to bounded RAM, preserving its identity."""
        estimate = estimate_size(arguments) + 4096
        source_arguments = arguments
        async with self._capture_slot(estimate):
            data = {"version": 1, "kind": kind, "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"), "arguments": arguments}
            fields = {}
            oversized = estimate > self._queue_limit() // 2
            arguments = dict(arguments)
            candidates = ("request_payload", "preflight_payload", "payload", "final_payload", "usage_payload", "content_delta", "segments")
            if kind == "record_session_start":
                original = arguments.get("request_payload")
                arguments["_request_name"] = original.get("request_name") if isinstance(original, dict) else None
            encoder = CaptureField.freeze if oversized else CaptureField.encode
            try:
                for name in candidates:
                    if arguments.get(name) is not None and not (name == "segments" and not arguments[name]):
                        fields[name] = encoder(arguments.pop(name), raw=name == "content_delta", chunk_chars=max(64, min(32768, self.policy.ram_bytes // 128)))
            except Exception as exc:
                self.state = "blocked"
                self.reason = "Debugger payload cannot be captured without loss"
                raise DebuggerPersistenceError(self.reason) from exc
            data["arguments"] = arguments
            op = Operation.create(self.next_seq+1, data, self.last_hash, fields or None)
            if not oversized and op.size > self._queue_limit() // 2:
                # Escaping may expand unusual payloads much more than their
                # input size. Use the same bounded oversized path, never reject.
                fields = {name: CaptureField.freeze(source_arguments[name], raw=name == "content_delta", chunk_chars=field.chunk_chars)
                          for name, field in fields.items()}
                op = Operation.create(self.next_seq+1, data, self.last_hash, fields or None)
                oversized = True
            if op.size > self.policy.ram_bytes:
                raise DebuggerPersistenceError("Debugger metadata exceeds the configured RAM capacity")
            deadline = time.monotonic() + self.policy.ram_seconds
            while self.ram_bytes + self.external_bytes + op.size > self._queue_limit():
                self._wake.set()
                if self.fatal or (self.owner._pool is None and time.monotonic() >= deadline):
                    self.state = "blocked"
                    raise DebuggerPersistenceError("Debugger capacity exhausted; capture was not accepted")
                async with self._progress:
                    try:
                        await asyncio.wait_for(self._progress.wait(), .1)
                    except TimeoutError:
                        pass
            self.queue.append(op)
            self.next_seq, self.last_hash = op.seq, op.chain_hash
            self.ram_bytes += op.size
            self._encoding_bytes = 0
            self._peak_ram = max(self._peak_ram, self.ram_bytes + self.external_bytes + self._replay_bytes)
            self.oldest_at = self.oldest_at or time.monotonic()
            self._wake.set()
        if oversized:
            # Borrow the awaiting producer's input until PG or local confirmation.
            while self.ack_seq < op.seq and not (self.spool and self.spool.pending_count and not any(item.seq == op.seq for item in self.queue)):
                if self.fatal:
                    raise DebuggerPersistenceError("Oversized capture is blocked; operation retained")
                async with self._progress:
                    try:
                        await asyncio.wait_for(self._progress.wait(), .1)
                    except TimeoutError:
                        pass
        return op.seq

    async def _spill(self) -> None:
        if not self.queue or not self.policy.spill:
            return
        if self.spool is None:
            spool = DebuggerSpool(self.path, self.policy.spool_bytes, self.policy.min_free_bytes)
            await spool.open({"instance_id": self.owner.instance_id, "target": self.target, "origin": self.origin,
                              "store_id": self.store_id, "owner_token": self.owner_token,
                              "pending_since": time.time() - (time.monotonic()-self.oldest_at if self.oldest_at else 0),
                              "ack_seq": self.ack_seq, "ack_hash": self.ack_hash})
            self.spool = spool
        batch = list(self.queue)
        await self.spool.append(batch)
        for op in batch:
            assert self.queue.popleft() is op
            self.ram_bytes -= op.size
        if not self.fatal:
            self.state = "local"
        async with self._progress:
            self._progress.notify_all()

    async def _apply(self, batch: list[Operation]) -> None:
        pool = self.owner._pool
        remote_seq, remote_hash = self.ack_seq, self.ack_hash
        async with pool.operation():
            # The fence and receipt move together with the content below. A
            # rollback undoes all of them; a lost reply is reconciled on connect.
            updated = await pool.connection.fetchval(
                "UPDATE debugger_sources AS source SET seq=$4,chain_hash=$5 "
                "FROM debugger_metadata AS owner WHERE source.origin=$1 "
                "AND source.seq=$2 AND source.chain_hash=$3 "
                "AND owner.instance_id=$6 AND owner.owner_token=$7 RETURNING source.seq",
                self.origin, remote_seq, remote_hash, batch[-1].seq, batch[-1].chain_hash,
                self.owner.instance_id, self.owner_token,
            )
            if updated is None:
                raise DebuggerConflict("Writer ownership or confirmed prefix changed; replay preserved")
            for op in batch:
                if op.seq != remote_seq + 1 or op.previous_hash != remote_hash:
                    raise DebuggerConflict("Operation sequence contains a gap")
                op.verify()
                data = json.loads(op.payload)
                if data.get("version") != 1 or data.get("kind") not in {"record_session_start", "update_session_status", "record_usage_summary", "record_event", "record_content_trace"}:
                    raise DebuggerConflict("Unsupported operation version")
                marker = applying_operation.set(True)
                try:
                    pool.observed_at = data["observed_at"]
                    if op.fields:
                        await apply_fields(self.owner, data["kind"], data["arguments"], op.fields)
                    else:
                        await getattr(self.owner, data["kind"])(**data["arguments"])
                finally:
                    pool.observed_at = None
                    applying_operation.reset(marker)
                remote_seq, remote_hash = op.seq, op.chain_hash
        # This is remote commit proof. A lost response never reaches this point.
        self.ack_seq, self.ack_hash = remote_seq, remote_hash

    async def _run(self) -> None:
        while not self.closed:
            self._wake.clear()
            try:
                age = time.monotonic()-self.oldest_at if self.oldest_at else 0
                if self.owner._pool is None and not self.fatal and time.monotonic() >= self.next_retry_at:
                    try:
                        await self._connect()
                    except Exception as exc:
                        self._failed(exc)
                if (self.queue and self.owner._pool is None and self.policy.spill
                    and (age >= self.policy.ram_seconds or self.ram_bytes >= self.policy.ram_bytes*.6)
                    and time.monotonic() >= getattr(self, "_next_spill_at", 0)):
                    try:
                        await self._spill()
                    except Exception as exc:
                        from debugger_postgres import postgres_failure_reason
                        self.reason = postgres_failure_reason(exc)
                        self.state = "blocked"
                        self.fatal = self.fatal or isinstance(exc, DebuggerConflict)
                        self._next_spill_at = time.monotonic() + 5
                if self.owner._pool is not None and not self.fatal and self.pending_count:
                    if self.spool and self.spool.pending_count:
                        batch = await self.spool.batch(max_bytes=self.policy.ram_bytes//4)
                        self._replay_bytes = sum(op.size for op in batch)
                        self._peak_ram = max(self._peak_ram, self.ram_bytes+self.external_bytes+self._encoding_bytes+self._replay_bytes)
                    else:
                        batch, size = [], 0
                        for op in self.queue:
                            batch.append(op)
                            size += op.size
                            if len(batch) >= 100 or size >= self.policy.ram_bytes//4:
                                break
                    await self._apply(batch)
                    if time.monotonic() - self._connected_at >= 60:
                        self._attempt = 0
                    if self.spool and self.spool.pending_count:
                        await self.spool.acknowledge(self.ack_seq, self.ack_hash)
                    while self.queue and self.queue[0].seq <= self.ack_seq:
                        self.ram_bytes -= self.queue.popleft().size
                    batch.clear()
                    op = None
                    self._replay_bytes = 0
                    # Retire the confirmed journal before applying new volatile
                    # RAM. Otherwise its old watermark survives a crash and
                    # incorrectly describes the intentionally unjournaled tail.
                    await self._release_empty_spool()
                    if not self.pending_count:
                        if not self.external_pending_bytes:
                            self.oldest_at = None
                        self.state, self.reason = "available", None
                    else:
                        self.state = "recovering" if self.spool else "available"
                    async with self._progress:
                        self._progress.notify_all()
                    await asyncio.sleep(0)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._replay_bytes = 0
                await self._disconnect()
                self._failed(exc)
                if isinstance(exc, DebuggerPersistenceError) and not isinstance(exc, DebuggerConflict):
                    self.state = "blocked"
            if self.pending_count and time.monotonic()-self._last_warning >= 30 and self.reason:
                logger.warning("Debugger %s: %d pending operations; %s", self.state, self.pending_count, self.reason)
                self._last_warning = time.monotonic()
            try:
                await asyncio.wait_for(self._wake.wait(), .1 if self.pending_count else 1.0)
            except TimeoutError:
                pass

    async def flush(self, timeout: float | None = None) -> None:
        """Wait for the captured prefix, or report unavailable instead of partial history."""
        if self.fatal or self.closed:
            raise DebuggerPersistenceError("Debugger evidence is unavailable")
        self._wake.set()
        async def wait():
            if self.flush_external is not None:
                await self.flush_external()
            async with self._admission_lock:
                cut = self.next_seq
            while self.ack_seq < cut:
                if self.fatal:
                    raise DebuggerPersistenceError("Debugger replay is blocked")
                async with self._progress:
                    await self._progress.wait()
        try:
            await asyncio.wait_for(wait(), timeout if timeout is not None else self.policy.read_timeout)
        except TimeoutError as exc:
            raise DebuggerPersistenceError("Debugger evidence has not reached PostgreSQL yet") from exc
        if self.owner._pool is None:
            raise DebuggerPersistenceError("Debugger PostgreSQL is unavailable")

    def retry(self) -> None:
        """Request one serialized retry; never replace or discard captured operations."""
        if self.fatal:
            raise DebuggerConflict("A consistency conflict requires verified manual recovery")
        self.next_retry_at = 0
        if self.state == "blocked":
            self.state = "available" if self.owner._pool is not None and not self.pending_count else "recovering"
            if self.state == "available":
                self.reason = None
        self._wake.set()

    async def close(self) -> None:
        """Drain normally; on an outage preserve accepted operations if spill is enabled."""
        if self.closed:
            return
        drained = False
        try:
            await self.flush(timeout=2)
            drained = True
        except DebuggerPersistenceError:
            pass
        self._closing = True
        # The owner task finishes its current transaction before yielding here.
        self.closed = True
        self._wake.set()
        if self._task:
            await self._task
        try:
            if self.queue and self.policy.spill:
                await self._spill()
            elif self.queue:
                self.loss_unknown = True
                logger.error("Debugger shutdown leaves %d volatile operations unpersisted", len(self.queue))
            if drained and not self.pending_count and self.owner._pool is not None:
                try:
                    async with self.owner._pool.operation():
                        await self.owner._pool.connection.execute(
                            "UPDATE debugger_metadata SET clean_shutdown=TRUE WHERE instance_id=$1 AND owner_token=$2",
                            self.owner.instance_id, self.owner_token,
                        )
                except Exception:
                    logger.warning("Debugger clean-shutdown receipt unavailable; next startup will report possible volatile loss")
        finally:
            try:
                if self.spool:
                    await self.spool.close()
                await self._disconnect()
            finally:
                self.lease.close()
