"""Private, bounded transport journal; never the historical debugger database."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

import json_utils as json
from debugger_payload import CaptureField
from provider_drain import ProviderDispatchClosed


class DebuggerPersistenceError(ProviderDispatchClosed):
    """An actionable persistence failure with no payloads or credentials."""


class DebuggerConflict(DebuggerPersistenceError):
    """Continuity cannot be proved; preserve the journal for operator review."""


@dataclass(frozen=True, slots=True)
class Operation:
    """Immutable operation identity and exact payload, including original time."""

    seq: int
    payload: bytes
    previous_hash: str
    chain_hash: str
    fields: dict[str, CaptureField] | None = None

    @classmethod
    def create(cls, seq: int, data: dict[str, Any], previous_hash: str, fields=None) -> Operation:
        if fields:
            data = {**data, "fields": {name: field.descriptor() for name, field in fields.items()}}
        payload = json.dumps_bytes(data, ensure_ascii=False)
        digest = hashlib.sha256(bytes.fromhex(previous_hash) + seq.to_bytes(8, "big") + payload).hexdigest()
        return cls(seq, payload, previous_hash, digest, fields)

    @property
    def size(self) -> int:
        # Payload, decoded call, driver binding and bookkeeping may coexist.
        return len(self.payload) * 4 + 1024 + sum(sys.getsizeof(field.cached) + field.byte_count * (6 if isinstance(field.cached, bytes) else 3) if field.cached is not None else field.chunk_chars * 16 for field in (self.fields or {}).values())

    @property
    def disk_size(self) -> int:
        return len(self.payload) + sum(field.byte_count for field in (self.fields or {}).values())

    def verify(self) -> None:
        digest = hashlib.sha256(bytes.fromhex(self.previous_hash) + self.seq.to_bytes(8, "big") + self.payload).hexdigest()
        if digest != self.chain_hash:
            raise DebuggerConflict("Journal operation checksum mismatch; original preserved")


ZERO_HASH = "0" * 64
APPLICATION_ID = 0x4753444A  # GSDJ, checked before an existing file is opened writable.


def private_directory(path: Path) -> None:
    """Protect a new, dedicated directory before SQLite creates any sidecars."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if any(path.iterdir()):
        raise DebuggerConflict("A new journal requires its own empty private directory")
    if os.name != "nt":
        path.chmod(0o700)
        return
    import csv
    import ctypes
    import subprocess
    from ctypes import wintypes
    result = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    sid = next(csv.reader(result.stdout.strip().splitlines()))[1]
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    security.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    descriptor = ctypes.c_void_p()
    # Protected DACL: current user and SYSTEM; inherited by the rollback journal.
    sddl = f"D:P(A;OICI;FA;;;{sid})(A;OICI;FA;;;SY)"
    if not security.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, ctypes.byref(descriptor), None):
        raise PermissionError("Cannot construct private debugger journal permissions")
    try:
        if not security.SetFileSecurityW(str(path), 0x80000004, descriptor):
            raise PermissionError("Cannot protect the debugger journal directory")
    finally:
        kernel.LocalFree(descriptor)


def validate_spool_path(path: Path, history_path: Path) -> None:
    """Reject history aliases and remote volumes before opening any file."""
    resolved = path.expanduser().resolve()
    if resolved == history_path.expanduser().resolve():
        raise DebuggerConflict("Journal must not be the historical SQLite database")
    if resolved.exists() and history_path.exists() and resolved.samefile(history_path):
        raise DebuggerConflict("Journal aliases the historical SQLite database")
    if str(resolved).startswith(("\\\\", "//")):
        raise DebuggerConflict("Journal must be on a private local disk")
    if os.name == "nt":
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(resolved.anchor)) == 4:
            raise DebuggerConflict("Journal must not be on a mapped network drive")


class DebuggerSpool:
    """One SQLite connection, exclusively owned; confirmed prefixes only are removed."""

    def __init__(self, path: Path, max_bytes: int, min_free_bytes: int, *, application_id: int = APPLICATION_ID):
        self.path = path
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        # Operational logs reuse the journal mechanics with a distinct brand;
        # neither reader can open the other's file for writing.
        self.application_id = application_id
        self.db: aiosqlite.Connection | None = None
        self.meta: dict[str, Any] = {}
        self.pending_bytes = 0
        self.pending_count = 0

    async def open(self, initial: dict[str, Any] | None = None) -> None:
        """Open or create only an explicitly supplied, private transport journal."""
        exists = self.path.exists()
        if not exists and initial is None:
            raise DebuggerConflict("Journal does not exist")
        if exists:
            with self.path.open("rb") as source:
                header = source.read(100)
            if len(header) < 100 or int.from_bytes(header[68:72], "big") != self.application_id:
                raise DebuggerConflict("Not a GranSabio transport journal; original untouched")
            if int.from_bytes(header[60:64], "big") != 1:
                raise DebuggerConflict("Unsupported journal version; original untouched")
            # Only our branded format reaches a writable connection. SQLite may
            # need to roll back a hot journal after a killed writer; mode=ro
            # would prevent that ordinary native recovery.
        if not exists:
            private_directory(self.path.parent)
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self.db = await aiosqlite.connect(self.path, timeout=.2)
        try:
            await self.db.execute("PRAGMA locking_mode=EXCLUSIVE")
            await self.db.execute("PRAGMA journal_mode=DELETE")
            await self.db.execute("PRAGMA synchronous=EXTRA")
            await self.db.execute("BEGIN EXCLUSIVE")
            if not exists:
                await self.db.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)")
                await self.db.execute("CREATE TABLE operations (seq INTEGER PRIMARY KEY, payload BLOB NOT NULL, previous_hash TEXT NOT NULL, chain_hash TEXT NOT NULL)")
                await self.db.execute("CREATE TABLE fields (seq INTEGER NOT NULL, name TEXT NOT NULL, part INTEGER NOT NULL, content BLOB NOT NULL, PRIMARY KEY(seq,name,part))")
                self.meta = {"version": 1, **initial}
                await self.db.execute("INSERT INTO metadata VALUES (1, ?)", (json.dumps(self.meta),))
                await self.db.execute("PRAGMA user_version=1")
                await self.db.execute(f"PRAGMA application_id={self.application_id}")
            else:
                async with self.db.execute("PRAGMA user_version") as cursor:
                    if (await cursor.fetchone())[0] != 1:
                        raise DebuggerConflict("Unsupported journal version; original preserved")
                async with self.db.execute("SELECT value FROM metadata WHERE id=1") as cursor:
                    row = await cursor.fetchone()
                self.meta = json.loads(row[0])
                if self.meta.get("version") != 1:
                    raise DebuggerConflict("Unsupported journal metadata; original preserved")
            await self.db.commit()
            async with self.db.execute("PRAGMA quick_check") as cursor:
                if (await cursor.fetchone())[0] != "ok":
                    raise DebuggerConflict("Journal integrity check failed; original preserved")
            await self._counts()
            await self.verify()
        except BaseException:
            await self.close()
            raise

    async def _counts(self) -> None:
        async with self.db.execute("SELECT COUNT(*), COALESCE(SUM(length(payload)),0) + (SELECT COALESCE(SUM(length(content)),0) FROM fields) FROM operations") as cursor:
            self.pending_count, self.pending_bytes = await cursor.fetchone()

    async def set_meta(self, **updates: Any) -> None:
        """Commit metadata before relying on it across a process restart."""
        replacement = {**self.meta, **updates}
        await self.db.execute("UPDATE metadata SET value=? WHERE id=1", (json.dumps(replacement),))
        await self.db.commit()
        self.meta = replacement

    async def append(self, operations: list[Operation]) -> None:
        """Atomically append immutable operations, reconciling an uncertain prior commit."""
        if not operations:
            return
        added_bytes, added_count = 0, 0
        try:
            await self.db.execute("BEGIN IMMEDIATE")
            for op in operations:
                op.verify()
                async with self.db.execute("SELECT chain_hash FROM operations WHERE seq=?", (op.seq,)) as cursor:
                    existing = await cursor.fetchone()
                if existing:
                    if existing[0] != op.chain_hash:
                        raise DebuggerConflict("Journal identity reused with different content")
                    continue
                amount = op.disk_size
                if self.pending_bytes + added_bytes + amount > self.max_bytes:
                    raise DebuggerPersistenceError("Journal byte quota reached; pending data preserved")
                if shutil.disk_usage(self.path.parent).free < self.min_free_bytes + amount * 2 + 65536:
                    raise DebuggerPersistenceError("Journal free-space reserve reached; pending data preserved")
                await self.db.execute("INSERT INTO operations VALUES (?,?,?,?)", (op.seq, op.payload, op.previous_hash, op.chain_hash))
                for name, field in (op.fields or {}).items():
                    index = 0
                    async for piece in field.chunks():
                        await self.db.execute("INSERT INTO fields VALUES (?,?,?,?)", (op.seq, name, index, piece.encode("utf-8")))
                        index += 1
                added_bytes += amount
                added_count += 1
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            await self._counts()  # Reconcile only an uncertain local transaction.
            raise
        self.pending_bytes += added_bytes
        self.pending_count += added_count

    async def batch(self, limit: int = 100, max_bytes: int = 1024 * 1024) -> list[Operation]:
        """Read a bounded prefix without constructing a second full history."""
        result, size = [], 0
        async with self.db.execute("SELECT seq,payload,previous_hash,chain_hash FROM operations ORDER BY seq LIMIT ?", (limit,)) as cursor:
            while row := await cursor.fetchone():
                op = self._restore(row)
                op.verify()
                result.append(op)
                size += op.size
                if size >= max_bytes:
                    break
        return result

    def _restore(self, row) -> Operation:
        data = json.loads(row[1])
        fields = {name: CaptureField(None, spec["raw"], spec["chunk_chars"], spec["sha256"], spec["bytes"], spec["chars"], self, row[0], name)
                  for name, spec in data.get("fields", {}).items()}
        return Operation(*row, fields or None)

    async def verify(self) -> None:
        """Verify sequence continuity and every hash without loading the whole journal."""
        previous_seq = self.meta["ack_seq"]
        previous_hash = self.meta["ack_hash"]
        async with self.db.execute("SELECT seq,payload,previous_hash,chain_hash FROM operations ORDER BY seq") as cursor:
            async for row in cursor:
                op = self._restore(row)
                op.verify()
                for field in (op.fields or {}).values():
                    async for _piece in field.chunks():
                        pass
                if op.seq != previous_seq + 1 or op.previous_hash != previous_hash:
                    raise DebuggerConflict("Journal contains a sequence gap or incompatible prefix")
                previous_seq, previous_hash = op.seq, op.chain_hash

    async def last(self) -> tuple[int, str]:
        async with self.db.execute("SELECT seq,chain_hash FROM operations ORDER BY seq DESC LIMIT 1") as cursor:
            row = await cursor.fetchone()
        return tuple(row) if row else (self.meta["ack_seq"], self.meta["ack_hash"])

    async def acknowledge(self, seq: int, chain_hash: str) -> None:
        """Delete only a prefix whose remote transaction has been proved committed."""
        replacement = {**self.meta, "ack_seq": seq, "ack_hash": chain_hash}
        try:
            await self.db.execute("BEGIN IMMEDIATE")
            async with self.db.execute("SELECT COUNT(*),COALESCE(SUM(length(payload)),0) + (SELECT COALESCE(SUM(length(content)),0) FROM fields WHERE seq<=?) FROM operations WHERE seq<=?", (seq, seq)) as cursor:
                removed_count, removed_bytes = await cursor.fetchone()
            await self.db.execute("DELETE FROM operations WHERE seq<=?", (seq,))
            await self.db.execute("DELETE FROM fields WHERE seq<=?", (seq,))
            await self.db.execute("UPDATE metadata SET value=? WHERE id=1", (json.dumps(replacement),))
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            async with self.db.execute("SELECT value FROM metadata WHERE id=1") as cursor:
                self.meta = json.loads((await cursor.fetchone())[0])
            await self._counts()
            raise
        self.meta = replacement
        self.pending_count -= removed_count
        self.pending_bytes -= removed_bytes

    async def close(self) -> None:
        if self.db is not None:
            db, self.db = self.db, None
            await db.close()
