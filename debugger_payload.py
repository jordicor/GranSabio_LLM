"""Bounded JSON/text encoding for a record larger than the capture RAM budget."""

from __future__ import annotations

import codecs
import hashlib
import sys
from dataclasses import dataclass
from typing import Any, Iterator

import json_utils as json


@dataclass(slots=True)
class SerializedJSON:
    """Already captured JSON; binding it must not serialize the payload again."""

    text: str | bytes
    request_name: str | None = None


def estimate_size(value: Any) -> int:
    """Conservatively size input without first allocating its serialized copy."""
    if isinstance(value, str):
        return sys.getsizeof(value) * 2 + len(value) * 6
    if isinstance(value, dict):
        return 64 + sum(estimate_size(k) + estimate_size(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return 64 + sum(estimate_size(v) for v in value)
    return 64


def json_pieces(value: Any, chars: int) -> Iterator[str]:
    """Produce the same compact JSON, escaping only bounded string slices."""
    if isinstance(value, str):
        yield '"'
        for offset in range(0, len(value), chars):
            yield json.dumps(value[offset:offset+chars], ensure_ascii=False)[1:-1]
        yield '"'
    elif isinstance(value, dict):
        yield "{"
        for index, (key, item) in enumerate(value.items()):
            if index:
                yield ","
            yield from json_pieces(str(key), chars)
            yield ":"
            yield from json_pieces(item, chars)
        yield "}"
    elif isinstance(value, (tuple, list)):
        yield "["
        for index, item in enumerate(value):
            if index:
                yield ","
            yield from json_pieces(item, chars)
        yield "]"
    else:
        yield json.dumps(value, ensure_ascii=False)


@dataclass
class CaptureField:
    """A stable value encoded a piece at a time, or a field already in the journal."""

    value: Any
    raw: bool
    chunk_chars: int
    sha256: str = ""
    byte_count: int = 0
    char_count: int = 0
    spool: Any = None
    sequence: int = 0
    name: str = ""
    cached: str | bytes | None = None

    @classmethod
    def encode(cls, value: Any, *, raw: bool, chunk_chars: int) -> CaptureField:
        """Capture the normal path once, without a second JSON decode/encode."""
        captured = value if raw else json.dumps_bytes(value, ensure_ascii=False)
        encoded = captured.encode("utf-8") if raw else captured
        return cls(None, raw, chunk_chars, hashlib.sha256(encoded).hexdigest(), len(encoded), len(value) if raw else 0, cached=captured)

    @classmethod
    def freeze(cls, value: Any, *, raw: bool, chunk_chars: int) -> CaptureField:
        # The oversized producer stays awaiting confirmation, so borrow its
        # input rather than copying an unbounded container tree. The second
        # checksum during application detects any concurrent mutation.
        field = cls(value, raw, chunk_chars)
        digest = hashlib.sha256()
        for piece in field.pieces():
            encoded = piece.encode("utf-8")
            digest.update(encoded)
            field.byte_count += len(encoded)
            field.char_count += len(piece)
        field.sha256 = digest.hexdigest()
        return field

    def pieces(self) -> Iterator[str]:
        if isinstance(self.cached, bytes):
            decoder = codecs.getincrementaldecoder("utf-8")()
            for offset in range(0, len(self.cached), self.chunk_chars):
                yield decoder.decode(self.cached[offset:offset+self.chunk_chars])
            final = decoder.decode(b"", final=True)
            if final:
                yield final
            return
        if self.cached is not None or self.raw:
            value = self.cached if self.cached is not None else self.value
            for offset in range(0, len(value), self.chunk_chars):
                yield value[offset:offset+self.chunk_chars]
        else:
            yield from json_pieces(self.value, self.chunk_chars)

    async def chunks(self):
        """Read at most a bounded encoded slice, including during journal replay."""
        digest, size = hashlib.sha256(), 0
        if self.spool:
            async with self.spool.db.execute("SELECT content FROM fields WHERE seq=? AND name=? ORDER BY part", (self.sequence, self.name)) as cursor:
                async for row in cursor:
                    digest.update(row[0])
                    size += len(row[0])
                    yield row[0].decode("utf-8")
        else:
            for piece in self.pieces():
                encoded = piece.encode("utf-8")
                digest.update(encoded)
                size += len(encoded)
                yield piece
        if digest.hexdigest() != self.sha256 or size != self.byte_count:
            from debugger_spool import DebuggerConflict
            raise DebuggerConflict("Journal field checksum mismatch; pending content preserved")

    def descriptor(self) -> dict:
        return {"raw": self.raw, "bytes": self.byte_count, "chars": self.char_count, "sha256": self.sha256, "chunk_chars": self.chunk_chars}


async def apply_fields(owner: Any, kind: str, arguments: dict, fields: dict[str, CaptureField]) -> None:
    """Apply an oversized record inside the SAME PG transaction and receipt."""
    arguments = dict(arguments)
    request_name = arguments.pop("_request_name", None)
    if all(field.cached is not None for field in fields.values()):
        captured = {}
        for name, field in fields.items():
            value = field.cached if field.raw else SerializedJSON(field.cached)
            if name == "request_payload":
                value.request_name = request_name
            captured[name] = value
        await getattr(owner, kind)(**{**arguments, **captured})
        return
    if kind == "update_session_status" and arguments.get("status"):
        from debug_logger import TERMINAL_DEBUGGER_STATUSES
        old_status = await owner._pool.connection.fetchval("SELECT status FROM sessions WHERE session_id=$1", arguments["session_id"])
        if old_status in TERMINAL_DEBUGGER_STATUSES and old_status != arguments["status"]:
            return
    if kind == "record_content_trace":
        await _apply_trace(owner, arguments, fields)
        return
    placeholders = {name: {} for name in fields}
    await getattr(owner, kind)(**{**arguments, **placeholders})
    sid = arguments["session_id"]
    mapping = {"request_payload": "request_json", "preflight_payload": "preflight_json",
               "usage_payload": "usage_json", "final_payload": "final_json", "payload": "payload_json"}
    if kind == "record_event":
        event_id = await owner._pool.connection.fetchval("SELECT MAX(id) FROM session_events WHERE session_id=$1", sid)
        table, where, keys = "session_events", "id=$2", [event_id]
    else:
        table, where, keys = "sessions", "session_id=$2", [sid]
    for name, field in fields.items():
        await _write_field(owner._pool.connection, table, mapping[name], where, keys, field)
    if kind == "record_session_start" and "request_payload" in fields:
        await owner._pool.connection.execute("UPDATE sessions SET request_name=$1 WHERE session_id=$2", request_name, sid)


async def _write_field(connection, table: str, column: str, where: str, keys: list, field: CaptureField):
    # Identifiers above are fixed internal mappings; no journal-supplied SQL.
    # A transaction-local large object avoids repeatedly rewriting growing TEXT
    # prefixes. It is unlinked before commit (or removed by rollback on failure).
    object_id = await connection.fetchval("SELECT lo_create(0)")
    offset = 0
    async for piece in field.chunks():
        encoded = piece.encode("utf-8")
        await connection.execute("SELECT lo_put($1,$2,$3)", object_id, offset, encoded)
        offset += len(encoded)
    await connection.execute(f"UPDATE {table} SET {column}=convert_from(lo_get($1),'UTF8') WHERE {where}", object_id, *keys)
    await connection.execute("SELECT lo_unlink($1)", object_id)


class TraceReader:
    """Read an existing BYTEA trace in bounded UTF-8 pieces, including NUL."""

    def __init__(self, connection, keys, chunk_bytes=32768):
        self.connection, self.keys, self.chunk_bytes = connection, keys, chunk_bytes
        self.byte_offset = 0
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()

    async def read(self, chars: int) -> str:
        result = []
        while chars:
            if not self.buffer:
                data = await self.connection.fetchval(
                    "SELECT substring(content FROM $6 FOR $7) FROM content_traces "
                    "WHERE session_id=$1 AND phase=$2 AND attempt=$3 AND model=$4 AND layer=$5",
                    *self.keys, self.byte_offset+1, self.chunk_bytes,
                )
                if not data:
                    from debugger_spool import DebuggerConflict
                    raise DebuggerConflict("Trace prefix is shorter than its recorded character count")
                self.byte_offset += len(data)
                self.buffer = self.decoder.decode(data)
            take = min(chars, len(self.buffer))
            result.append(self.buffer[:take])
            self.buffer = self.buffer[take:]
            chars -= take
        return "".join(result)

    async def skip(self, chars: int) -> None:
        while chars:
            take = min(chars, self.chunk_bytes)
            await self.read(take)
            chars -= take


async def _apply_trace(owner, arguments: dict, fields: dict[str, CaptureField]) -> None:
    """Append one oversized trace once, without repeatedly rewriting its prefix."""
    from debugger_spool import DebuggerConflict
    connection = owner._pool.connection
    keys = [arguments[k] for k in ("session_id", "phase", "attempt", "model", "layer")]
    old_chars = await connection.fetchval(
        "SELECT content_chars FROM content_traces WHERE session_id=$1 AND phase=$2 AND attempt=$3 AND model=$4 AND layer=$5", *keys,
    )
    expected = arguments["expected_offset"]
    if old_chars is None:
        await owner._require_session(arguments["session_id"])
        old_chars = 0
    if old_chars < expected:
        raise DebuggerConflict("Trace prefix is missing; pending content preserved")
    content = fields["content_delta"]
    overlap = min(old_chars-expected, content.char_count)
    reader = TraceReader(connection, keys, content.chunk_chars)
    if overlap:
        await reader.skip(expected)
    object_id = await connection.fetchval("SELECT lo_create(0)")
    byte_offset = 0
    async for piece in content.chunks():
        shared = min(overlap, len(piece))
        if shared and await reader.read(shared) != piece[:shared]:
            raise DebuggerConflict("Trace overlap differs from captured content")
        overlap -= shared
        encoded = piece[shared:].encode("utf-8")
        if encoded:
            await connection.execute("SELECT lo_put($1,$2,$3)", object_id, byte_offset, encoded)
            byte_offset += len(encoded)
    if byte_offset:
        parameters = {**arguments, "object_id": object_id, "total": expected+content.char_count}
        await owner._pool.execute(
            "INSERT INTO content_traces (session_id,phase,attempt,model,layer,project_id,subphase,iteration,content_status,content,content_chars,created_at,updated_at) "
            "VALUES (:session_id,:phase,:attempt,:model,:layer,:project_id,:subphase,:iteration,:content_status,lo_get(:object_id),:total,datetime('now'),datetime('now')) "
            "ON CONFLICT(session_id,phase,attempt,model,layer) DO UPDATE SET "
            "content=content_traces.content || excluded.content,content_chars=excluded.content_chars,"
            "project_id=COALESCE(excluded.project_id,content_traces.project_id),subphase=excluded.subphase,"
            "iteration=COALESCE(excluded.iteration,content_traces.iteration),content_status=excluded.content_status,"
            "segments=NULL,updated_at=excluded.updated_at", parameters,
        )
    if "segments" in fields:
        await _write_field(connection, "content_traces", "segments",
                           "session_id=$2 AND phase=$3 AND attempt=$4 AND model=$5 AND layer=$6", keys, fields["segments"])
    await connection.execute("SELECT lo_unlink($1)", object_id)
