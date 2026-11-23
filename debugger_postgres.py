"""PostgreSQL transport for the debugger's fixed SQL statements.

This is deliberately scoped to DebugLogger, not a general SQL compatibility
layer. Each installation owns a schema and one writer connection. Operations
are transactional and serialized; PostgreSQL never becomes execution authority.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping, Sequence
from uuid import uuid4

import asyncpg


class DebuggerSetupError(ValueError):
    """A safe, actionable setup error containing no connection or payload data."""


def debugger_schema(instance_id: str) -> str:
    """Validate the explicit, stable installation identity used as a schema."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", instance_id):
        raise DebuggerSetupError("DEBUGGER_INSTANCE_ID must be 1-40 lowercase letters, digits or underscores, starting with a letter")
    return "gransabio_debugger_" + instance_id


def _statement(query: str, parameters=(), observed_at=None):
    """Bind the debugger's SQLite placeholders using PostgreSQL parameters."""
    query = query.replace(
        "datetime('now')",
        "__debugger_observed_at__" if observed_at else "to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')",
    )
    if isinstance(parameters, dict):
        names = []

        def bind(match):
            name = match.group(1)
            if name not in parameters:
                return match.group(0)
            if name not in names:
                names.append(name)
            return f"${names.index(name) + 1}"

        query = re.sub(r":([a-zA-Z_][a-zA-Z0-9_]*)", bind, query)
        values = [parameters[name] for name in names]
    else:
        values = list(parameters)
        index = iter(range(1, len(values) + 1))
        query = re.sub(r"\?", lambda _: f"${next(index)}", query)
    if "__debugger_observed_at__" in query:
        values.append(observed_at)
        query = query.replace("__debugger_observed_at__", f"${len(values)}")
    return query, values


class _Query:
    """Small cursor bridge for DebugLogger's existing await/context-manager API."""

    def __init__(self, connection, query, parameters, observed_at=None):
        self.connection = connection
        self.query, self.parameters = _statement(query, parameters, observed_at)
        self.rows = []

    async def _run(self):
        self.rows = await self.connection.fetch(self.query, *self.parameters)
        return self

    def __await__(self):
        return self._run().__await__()

    async def __aenter__(self):
        return await self._run()

    async def __aexit__(self, *args):
        return False

    async def fetchone(self):
        return self.rows[0] if self.rows else None

    async def fetchall(self):
        return self.rows


class PostgresDebuggerConnection:
    """A single debugger writer with a transaction per public operation."""

    def __init__(self, connection, schema):
        self.connection = connection
        self.schema = schema
        self.lock = asyncio.Lock()
        self.observed_at = None

    @classmethod
    async def connect(cls, database_url: str, instance_id: str) -> PostgresDebuggerConnection:
        """Connect with the configured TLS policy and fence duplicate writers."""
        schema = debugger_schema(instance_id)
        if not database_url.startswith(("postgresql://", "postgres://")):
            raise DebuggerSetupError("DEBUGGER_DATABASE_URL must be a PostgreSQL connection URI")
        connection = await asyncpg.connect(
            database_url, timeout=3, command_timeout=30,
            server_settings={"application_name": f"gransabio-debugger:{instance_id}"},
        )
        try:
            if await connection.fetchval("SHOW fsync") != "on":
                raise DebuggerSetupError("PostgreSQL fsync must be enabled for durable debugger acknowledgements")
            await connection.execute("SET synchronous_commit TO on")
            locked = await connection.fetchval(
                "SELECT pg_try_advisory_lock(hashtextextended(current_database() || $1, 0))",
                schema,
            )
            if not locked:
                raise DebuggerSetupError("Another debugger writer owns DEBUGGER_INSTANCE_ID; use a distinct identity per installation")
            return cls(connection, schema)
        except BaseException:
            await connection.close()
            raise

    async def initialize(self, schema_sql: str, instance_id: str) -> None:
        """Create only this debugger's schema; refuse unrelated existing data."""
        async with self.operation():
            existing = await self.connection.fetchval(
                "SELECT to_regnamespace($1)::oid", self.schema,
            )
            if existing is not None:
                marker = await self.connection.fetchval(
                    "SELECT to_regclass($1)::oid", f"{self.schema}.debugger_metadata",
                )
                if marker is None:
                    raise DebuggerSetupError("Debugger schema already exists without its ownership marker")
            await self.connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            await self.connection.execute(f'SET search_path TO "{self.schema}"')
            await self.connection.execute(
                "CREATE TABLE IF NOT EXISTS debugger_metadata (instance_id TEXT PRIMARY KEY, version INTEGER NOT NULL)"
            )
            if existing is not None:
                rows = await self.connection.fetch("SELECT instance_id, version FROM debugger_metadata")
                if len(rows) != 1 or rows[0][0] != instance_id or rows[0][1] not in (1, 2):
                    raise DebuggerSetupError("Unsupported debugger schema identity or version")
            else:
                await self.connection.execute("INSERT INTO debugger_metadata VALUES ($1, 1)", instance_id)
            await self.connection.execute("ALTER TABLE debugger_metadata ADD COLUMN IF NOT EXISTS store_id TEXT; ALTER TABLE debugger_metadata ADD COLUMN IF NOT EXISTS owner_token TEXT")
            await self.connection.execute("ALTER TABLE debugger_metadata ADD COLUMN IF NOT EXISTS clean_shutdown BOOLEAN NOT NULL DEFAULT TRUE; ALTER TABLE debugger_metadata ADD COLUMN IF NOT EXISTS loss_unknown BOOLEAN NOT NULL DEFAULT FALSE")
            await self.connection.execute("UPDATE debugger_metadata SET version=2, store_id=COALESCE(store_id,$1)", str(uuid4()))
            await self.connection.execute("CREATE TABLE IF NOT EXISTS debugger_sources (origin TEXT PRIMARY KEY, seq BIGINT NOT NULL, chain_hash TEXT NOT NULL)")
            schema_sql, _ = _statement(schema_sql)
            schema_sql = schema_sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY")
            schema_sql = schema_sql.replace(" TEXT", ' TEXT COLLATE "C"')
            # Raw traces can contain NUL. BYTEA preserves every UTF-8 byte;
            # content_chars and append offsets still count Python characters.
            schema_sql = schema_sql.replace('content TEXT COLLATE "C" NOT NULL DEFAULT \'\'', "content BYTEA NOT NULL DEFAULT ''::bytea")
            await self.connection.execute(schema_sql)

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        """Commit an entire operation, or roll it back on error/cancellation."""
        async with self.lock:
            async with self.connection.transaction():
                yield

    def execute(self, query: str, parameters: Sequence[Any] | Mapping[str, Any] = ()) -> _Query:
        """Execute one of DebugLogger's fixed, parameterized SQL statements."""
        return _Query(self.connection, query, parameters, self.observed_at)

    async def commit(self) -> None:
        """Let the surrounding operation own its atomic commit."""
        # The operation context owns the commit, including multi-statement
        # append/retention operations. SQLite still uses its explicit commits.
        pass

    async def close(self) -> None:
        """Close the connection, also releasing its instance ownership lock."""
        await self.connection.close(timeout=5)


def postgres_failure_reason(exc: BaseException) -> str:
    """Expose failure type without DSNs, credentials, queries or payload values."""
    if isinstance(exc, DebuggerSetupError):
        return f"PostgreSQL setup: {exc}"
    from debugger_spool import DebuggerPersistenceError
    if isinstance(exc, DebuggerPersistenceError):
        return str(exc)
    code = getattr(exc, "sqlstate", None)
    return f"PostgreSQL {type(exc).__name__}" + (f" (SQLSTATE {code})" if code else "")


def is_postgres_storage_error(exc: BaseException) -> bool:
    """Connection/storage failures disable capture until the next startup."""
    return isinstance(exc, (
        asyncpg.PostgresConnectionError, asyncpg.InterfaceError,
        asyncpg.InsufficientPrivilegeError, asyncpg.DiskFullError,
        asyncpg.ReadOnlySQLTransactionError, OSError, TimeoutError,
    ))
