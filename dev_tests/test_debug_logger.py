"""Targeted tests for debugger persistence hardening."""

import os
import sqlite3
from pathlib import Path

import pytest

import debug_logger as debug_logger_module
from debug_logger import (
    DEFAULT_DEBUGGER_DB_NAME,
    DebugLogger,
    migrate_debugger_db_if_needed,
    resolve_debugger_db_path,
)


class _RaisingConnection:
    """Minimal async connection stub that raises a configured SQLite error."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.execute_calls = 0
        self.closed = False

    async def execute(self, *args, **kwargs):
        self.execute_calls += 1
        raise self._exc

    async def commit(self):
        return None

    async def close(self):
        self.closed = True


def test_resolve_debugger_db_path_uses_local_state_for_implicit_default(monkeypatch, tmp_path):
    """The implicit default should move out of the repo into user-local state."""
    monkeypatch.delenv("DEBUGGER_DB_PATH", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)

    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        expected = tmp_path / "localappdata" / "GranSabio_LLM" / "debugger" / DEFAULT_DEBUGGER_DB_NAME
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        expected = tmp_path / "state" / "gransabio" / "debugger" / DEFAULT_DEBUGGER_DB_NAME

    resolved = resolve_debugger_db_path(DEFAULT_DEBUGGER_DB_NAME)

    assert resolved == expected


def test_resolve_debugger_db_path_respects_explicit_override(monkeypatch):
    """An explicit override keeps the legacy relative path unchanged."""
    monkeypatch.setenv("DEBUGGER_DB_PATH", DEFAULT_DEBUGGER_DB_NAME)

    resolved = resolve_debugger_db_path(DEFAULT_DEBUGGER_DB_NAME)

    assert resolved == Path(DEFAULT_DEBUGGER_DB_NAME)


def test_migrate_debugger_db_if_needed_copies_legacy_history_from_cwd(monkeypatch, tmp_path):
    """Legacy history should be copied once from the old default location."""
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.chdir(legacy_root)
    monkeypatch.delenv("DEBUGGER_DB_PATH", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    legacy_db.write_text("legacy history", encoding="utf-8")
    target_db = resolve_debugger_db_path(DEFAULT_DEBUGGER_DB_NAME)

    migrated_from = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert migrated_from == legacy_db
    assert target_db.read_text(encoding="utf-8") == "legacy history"


@pytest.mark.asyncio
async def test_record_event_disables_storage_after_terminal_io_error():
    """Disk I/O errors should disable the debugger and stop future write attempts."""
    logger_instance = DebugLogger(enabled=True, db_path=DEFAULT_DEBUGGER_DB_NAME)
    logger_instance._pool = _RaisingConnection(sqlite3.OperationalError("disk I/O error"))
    logger_instance._event_counters["session-1"] = 0

    await logger_instance.record_event("session-1", event_type="evt", payload={"ok": True})

    assert logger_instance.storage_disabled_reason is not None
    assert logger_instance._pool is None


@pytest.mark.asyncio
async def test_record_event_keeps_storage_enabled_on_transient_operational_error():
    """Transient SQLite errors should propagate instead of disabling persistence."""
    logger_instance = DebugLogger(enabled=True, db_path=DEFAULT_DEBUGGER_DB_NAME)
    fake_connection = _RaisingConnection(sqlite3.OperationalError("database is locked"))
    logger_instance._pool = fake_connection
    logger_instance._event_counters["session-1"] = 0

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await logger_instance.record_event("session-1", event_type="evt", payload={"ok": True})

    assert logger_instance.storage_disabled_reason is None
    assert logger_instance._pool is fake_connection


@pytest.mark.asyncio
async def test_initialize_does_not_retry_after_terminal_storage_failure(monkeypatch, tmp_path):
    """A terminal init failure should disable the shared logger and prevent reopen loops."""
    await debug_logger_module.shutdown_debug_logger()
    monkeypatch.delenv("DEBUGGER_DB_PATH", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    calls = 0

    async def fake_connect(path: str):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(debug_logger_module.aiosqlite, "connect", fake_connect)

    try:
        logger_instance = await debug_logger_module.get_debug_logger(
            enabled=True,
            db_path=DEFAULT_DEBUGGER_DB_NAME,
        )
        await logger_instance.initialize()

        same_instance = await debug_logger_module.get_debug_logger(
            enabled=True,
            db_path=DEFAULT_DEBUGGER_DB_NAME,
        )
        await same_instance.initialize()

        assert same_instance is logger_instance
        assert logger_instance.storage_disabled_reason is not None
        assert calls == 1
    finally:
        await debug_logger_module.shutdown_debug_logger()
