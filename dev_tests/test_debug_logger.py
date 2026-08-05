"""Targeted tests for debugger persistence and legacy-database adoption.

Contract since C13 (verbatim capture, no TTL by default): a legacy database is
adopted EXACTLY as it is. Nothing is redacted on write or on startup, nothing
expires unless a retention window is configured, and a database is only wiped
when it is not a usable debugger database at all. What is still verified, and
matters more now that the store grows without a ceiling, is SQLite integrity.
"""

import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import debug_logger as debug_logger_module
import json_utils as json
from debug_logger import (
    DEFAULT_DEBUGGER_DB_NAME,
    DebugLogger,
    migrate_debugger_db_if_needed,
    resolve_debugger_db_path,
)

LEGACY_REQUEST = {
    "prompt": "PRIVATE LEGACY PROMPT",
    "source_text": "PRIVATE LEGACY SOURCE",
    "request_name": "legacy request",
}
LEGACY_EVENT_PAYLOAD = {"system_prompt": "PRIVATE EVENT PROMPT"}


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


def _create_legacy_debugger_db(path: Path) -> None:
    """Create the pre-hardening debugger schema with sensitive telemetry."""
    with closing(sqlite3.connect(str(path))) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                status TEXT,
                request_json TEXT,
                preflight_json TEXT,
                final_json TEXT,
                usage_json TEXT,
                notes TEXT
            );
            CREATE TABLE session_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_order INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO sessions (
                session_id,
                created_at,
                updated_at,
                status,
                request_json
            )
            VALUES (?, datetime('now'), ?, ?, ?)
            """,
            (
                (
                    "fresh-session",
                    "2099-01-01 00:00:00",
                    "completed",
                    json.dumps(LEGACY_REQUEST),
                ),
                (
                    "ancient-session",
                    "2000-01-01 00:00:00",
                    "generating",
                    json.dumps({"prompt": "ANCIENT PRIVATE PROMPT"}),
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO session_events (
                session_id,
                event_order,
                event_type,
                payload_json,
                created_at
            )
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (
                "fresh-session",
                1,
                "legacy-event",
                json.dumps(LEGACY_EVENT_PAYLOAD),
            ),
        )
        connection.commit()


def _read_legacy_request(path: Path, session_id: str) -> dict:
    with closing(sqlite3.connect(str(path))) as connection:
        row = connection.execute(
            "SELECT request_json FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert row is not None
    return json.loads(row[0])


def _assert_legacy_db_kept_verbatim(path: Path) -> None:
    """Assert nothing was redacted or expired, without the async logger."""
    assert _read_legacy_request(path, "fresh-session") == LEGACY_REQUEST
    with closing(sqlite3.connect(str(path))) as connection:
        ancient = connection.execute(
            """
            SELECT COUNT(*)
            FROM sessions
            WHERE session_id = 'ancient-session'
            """
        ).fetchone()[0]
        event_payload = connection.execute(
            """
            SELECT payload_json
            FROM session_events
            WHERE session_id = 'fresh-session'
            """
        ).fetchone()[0]
    assert ancient == 1, "rows older than any former TTL must survive"
    assert json.loads(event_payload) == LEGACY_EVENT_PAYLOAD


def _configure_implicit_debugger_paths(monkeypatch, tmp_path) -> tuple[Path, Path]:
    """Point implicit legacy and managed paths at an isolated test directory."""
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.chdir(legacy_root)
    for variable in (
        "DEBUGGER_DB_PATH",
        "XDG_STATE_HOME",
        "LOCALAPPDATA",
        "APPDATA",
    ):
        monkeypatch.delenv(variable, raising=False)
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return legacy_root, resolve_debugger_db_path(DEFAULT_DEBUGGER_DB_NAME)


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


def test_migrate_debugger_db_if_needed_moves_verified_history_from_cwd(
    monkeypatch,
    tmp_path,
):
    """A verified SQLite snapshot is installed before the legacy copy is removed."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)

    outcome = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert outcome.adopted == legacy_db
    assert outcome.blocked_candidate is None
    assert outcome.cleanup_ok
    assert not legacy_db.exists()
    _assert_legacy_db_kept_verbatim(target_db)


@pytest.mark.asyncio
async def test_initialize_migrates_legacy_history_without_scrub_or_expiry(
    monkeypatch,
    tmp_path,
):
    """Startup adopts the legacy database as is: no scrub, no TTL, no discard."""
    legacy_root, _target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    logger_instance = DebugLogger(
        enabled=True,
        db_path=DEFAULT_DEBUGGER_DB_NAME,
    )

    await logger_instance.initialize()
    try:
        assert not legacy_db.exists()
        fresh = await logger_instance.get_session_details("fresh-session")
        assert fresh is not None
        assert json.loads(fresh["request_json"]) == LEGACY_REQUEST
        ancient = await logger_instance.get_session_details("ancient-session")
        assert ancient is not None, "startup must not expire anything by default"
        events = await logger_instance.get_session_events("fresh-session")
        assert json.loads(events[0]["payload_json"]) == LEGACY_EVENT_PAYLOAD
    finally:
        await logger_instance.close()


def test_migration_keeps_source_verbatim_when_removal_fails(monkeypatch, tmp_path):
    """A locked source is retained untouched once the target is verified."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_unlink = Path.unlink

    def _locked_source_unlink(path, *args, **kwargs):
        if path == legacy_db:
            raise PermissionError("legacy database is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _locked_source_unlink)

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    assert outcome.adopted == legacy_db
    assert outcome.blocked_candidate is None
    assert target_db.exists()
    assert legacy_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)
    _assert_legacy_db_kept_verbatim(target_db)


def test_existing_target_preserves_legacy_history_untouched(
    monkeypatch,
    tmp_path,
):
    """An unprovable prior copy is neither deleted nor rewritten."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    target_db.parent.mkdir(parents=True)
    _create_legacy_debugger_db(target_db)
    with closing(sqlite3.connect(str(target_db))) as connection:
        connection.execute("CREATE TABLE target_marker (value TEXT)")
        connection.execute("INSERT INTO target_marker VALUES ('do not overwrite')")
        connection.commit()

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    # Preserved beside a healthy target is NOT an adoption: reporting it as one
    # is what used to hide a failed install on the following boot.
    assert outcome.preserved == legacy_db
    assert outcome.adopted is None
    assert outcome.blocked_candidate is None
    assert legacy_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)
    with closing(sqlite3.connect(str(target_db))) as connection:
        marker = connection.execute("SELECT value FROM target_marker").fetchone()[0]
    assert marker == "do not overwrite"


def test_existing_target_removes_orphaned_legacy_sqlite_sidecars(
    monkeypatch,
    tmp_path,
):
    """A prior interrupted cleanup must not leave journal payloads behind."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    legacy_wal = Path(f"{legacy_db}-wal")
    legacy_wal.write_bytes(b"PRIVATE ORPHANED SQLITE JOURNAL CONTENT")
    target_db.parent.mkdir(parents=True)
    with closing(sqlite3.connect(str(target_db))) as connection:
        connection.execute("CREATE TABLE target_marker (value TEXT)")
        connection.commit()

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    assert outcome.adopted is None
    assert outcome.preserved is None
    assert outcome.blocked_candidate is None
    assert not legacy_wal.exists()
    assert target_db.exists()


def test_missing_target_still_removes_orphaned_legacy_sidecars(
    monkeypatch,
    tmp_path,
):
    """Orphan journals are retired even when no main or target DB exists."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    legacy_journal = Path(f"{legacy_db}-journal")
    legacy_journal.write_bytes(b"PRIVATE ORPHANED SQLITE JOURNAL CONTENT")

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    assert outcome.adopted is None
    assert outcome.blocked_candidate is None
    assert outcome.cleanup_ok
    assert not legacy_journal.exists()
    assert not target_db.exists()


def test_failed_atomic_install_keeps_source_for_safe_retry(
    monkeypatch,
    tmp_path,
):
    """A failed install leaves the source intact and the migration retryable."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_link = debug_logger_module.os.link

    def _failed_link(*_args, **_kwargs):
        raise OSError("atomic install unavailable")

    monkeypatch.setattr(debug_logger_module.os, "link", _failed_link)

    blocked = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert blocked.adopted is None
    assert blocked.blocked_candidate == legacy_db
    assert "atomic install unavailable" in blocked.blocked_error
    assert legacy_db.exists()
    assert not target_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)

    monkeypatch.setattr(debug_logger_module.os, "link", original_link)

    retried = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert retried.adopted == legacy_db
    assert retried.blocked_candidate is None
    assert target_db.exists()
    assert not legacy_db.exists()
    assert debug_logger_module._verify_debugger_database(target_db)


def test_post_install_verification_failure_preserves_source(
    monkeypatch,
    tmp_path,
):
    """Once the target is proven, a later failure still leaves the source intact."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )

    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_snapshot = debug_logger_module._create_debugger_db_snapshot
    calls = 0

    def _fail_second_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sqlite3.OperationalError("source verification unavailable")
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(
        debug_logger_module,
        "_create_debugger_db_snapshot",
        _fail_second_snapshot,
    )

    outcome = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert outcome.adopted == legacy_db
    assert outcome.blocked_candidate is None
    assert target_db.exists()
    assert legacy_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_locked_sidecar_survives_only_as_empty_file(
    suffix,
    monkeypatch,
    tmp_path,
):
    """Every SQLite sidecar is wiped before a failed unlink can leave it behind."""
    database_path = tmp_path / "debugger.db"
    sidecar = Path(f"{database_path}{suffix}")
    sidecar.write_bytes(b"PRIVATE LEGACY JOURNAL PAYLOAD")
    original_unlink = Path.unlink

    def _locked_unlink(path, *args, **kwargs):
        if path == sidecar:
            raise PermissionError("sidecar is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _locked_unlink)

    assert debug_logger_module._cleanup_sqlite_sidecars(database_path) is True
    assert sidecar.exists()
    assert sidecar.read_bytes() == b""


def test_retirement_orders_sidecars_before_main_database(
    monkeypatch,
    tmp_path,
):
    """Retirement cannot remove the main DB while raw sidecars remain."""
    database_path = tmp_path / "debugger.db"
    _create_legacy_debugger_db(database_path)
    assert debug_logger_module._verify_debugger_database(database_path)
    order = []
    original_unlink = Path.unlink

    def _cleanup_sidecars(path):
        assert path == database_path
        order.append("sidecars")
        return True

    def _record_main_unlink(path, *args, **kwargs):
        if path == database_path:
            order.append("main")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        debug_logger_module,
        "_cleanup_sqlite_sidecars",
        _cleanup_sidecars,
    )
    monkeypatch.setattr(Path, "unlink", _record_main_unlink)

    assert debug_logger_module._retire_migrated_legacy_db(database_path)
    assert order == ["sidecars", "main"]


@pytest.mark.parametrize("failure_surface", ["integrity", "hash"])
def test_invalid_new_target_is_removed_and_retryable(
    failure_surface,
    monkeypatch,
    tmp_path,
):
    """A post-link target verification failure cannot stick or lose the source."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_verify = debug_logger_module._verify_debugger_database
    original_hash = debug_logger_module._sha256_file

    if failure_surface == "integrity":
        def _reject_installed_target(path):
            if path == target_db:
                return False
            return original_verify(path)

        monkeypatch.setattr(
            debug_logger_module,
            "_verify_debugger_database",
            _reject_installed_target,
        )
    else:
        def _mismatch_installed_target(path):
            if path == target_db:
                return "mismatched-target-digest"
            return original_hash(path)

        monkeypatch.setattr(
            debug_logger_module,
            "_sha256_file",
            _mismatch_installed_target,
        )

    blocked = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert blocked.adopted is None
    assert blocked.blocked_candidate == legacy_db
    assert not target_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)

    monkeypatch.setattr(
        debug_logger_module,
        "_verify_debugger_database",
        original_verify,
    )
    monkeypatch.setattr(debug_logger_module, "_sha256_file", original_hash)

    retried = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert retried.adopted == legacy_db
    assert target_db.exists()
    assert debug_logger_module._verify_debugger_database(target_db)


def test_locked_invalid_target_is_zeroed_then_unblocks_next_retry(
    monkeypatch,
    tmp_path,
):
    """If unlink is temporarily blocked, an invalid target survives only empty."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_verify = debug_logger_module._verify_debugger_database
    original_unlink = Path.unlink

    def _reject_installed_target(path):
        if path == target_db:
            return False
        return original_verify(path)

    def _lock_target(path, *args, **kwargs):
        if path == target_db:
            raise PermissionError("target is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        debug_logger_module,
        "_verify_debugger_database",
        _reject_installed_target,
    )
    monkeypatch.setattr(Path, "unlink", _lock_target)

    blocked = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert blocked.adopted is None
    assert blocked.blocked_candidate == legacy_db
    assert not blocked.cleanup_ok
    assert target_db.exists()
    assert target_db.read_bytes() == b""
    _assert_legacy_db_kept_verbatim(legacy_db)

    monkeypatch.setattr(
        debug_logger_module,
        "_verify_debugger_database",
        original_verify,
    )
    monkeypatch.setattr(Path, "unlink", original_unlink)

    retried = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert retried.adopted == legacy_db
    assert debug_logger_module._verify_debugger_database(target_db)


def test_failed_temp_cleanup_leaves_only_empty_snapshot(
    monkeypatch,
    tmp_path,
):
    """A temp verification and unlink failure leaves no sensitive bytes on disk."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    assert debug_logger_module._verify_debugger_database(legacy_db)
    target_db.parent.mkdir(parents=True, exist_ok=True)
    original_verify = debug_logger_module._verify_debugger_database
    original_unlink = Path.unlink

    def _reject_temp(path):
        if path.name.startswith(f".{DEFAULT_DEBUGGER_DB_NAME}.migration-"):
            return False
        return original_verify(path)

    def _lock_temp(path, *args, **kwargs):
        if path.name.startswith(f".{DEFAULT_DEBUGGER_DB_NAME}.migration-"):
            raise PermissionError("temp is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        debug_logger_module,
        "_verify_debugger_database",
        _reject_temp,
    )
    monkeypatch.setattr(Path, "unlink", _lock_temp)

    with pytest.raises(
        sqlite3.DatabaseError,
        match="snapshot failed verification",
    ):
        debug_logger_module._create_debugger_db_snapshot(
            legacy_db,
            target_db.parent,
        )

    snapshots = list(
        target_db.parent.glob(
            f".{DEFAULT_DEBUGGER_DB_NAME}.migration-*.tmp"
        )
    )
    assert snapshots
    assert all(snapshot.read_bytes() == b"" for snapshot in snapshots)


def test_successful_migration_temp_unlink_failure_leaves_verified_snapshots(
    monkeypatch,
    tmp_path,
):
    """Successful temp files may survive a lock, but only as verified databases."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_unlink = Path.unlink

    def _lock_migration_temp(path, *args, **kwargs):
        if path.name.startswith(f".{DEFAULT_DEBUGGER_DB_NAME}.migration-"):
            raise PermissionError("migration temp is locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _lock_migration_temp)

    assert (
        migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME).adopted
        == legacy_db
    )
    snapshots = list(
        target_db.parent.glob(
            f".{DEFAULT_DEBUGGER_DB_NAME}.migration-*.tmp"
        )
    )
    assert snapshots
    assert all(
        debug_logger_module._verify_debugger_database(snapshot)
        for snapshot in snapshots
    )
    assert debug_logger_module._verify_debugger_database(target_db)

    monkeypatch.setattr(Path, "unlink", original_unlink)
    second_boot = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)
    assert second_boot.adopted is None
    assert second_boot.blocked_candidate is None
    assert not list(
        target_db.parent.glob(
            f".{DEFAULT_DEBUGGER_DB_NAME}.migration-*.tmp"
        )
    )


def test_migration_processes_all_distinct_legacy_candidates(
    monkeypatch,
    tmp_path,
):
    """Installing the first candidate must not leave a second candidate raw."""
    _legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    first_candidate = tmp_path / "candidate-one.db"
    second_candidate = tmp_path / "candidate-two.db"
    _create_legacy_debugger_db(first_candidate)
    _create_legacy_debugger_db(second_candidate)
    monkeypatch.setattr(
        debug_logger_module,
        "get_legacy_debugger_db_candidates",
        lambda _configured_path: [first_candidate, second_candidate],
    )

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    assert outcome.adopted == first_candidate
    assert outcome.preserved == second_candidate
    assert outcome.blocked_candidate is None
    assert not first_candidate.exists()
    assert second_candidate.exists()
    _assert_legacy_db_kept_verbatim(second_candidate)
    assert debug_logger_module._verify_debugger_database(target_db)


def test_blocked_candidate_outranks_every_later_candidate(
    monkeypatch,
    tmp_path,
    caplog,
):
    """A sound candidate that failed to install stops the adoption of the rest.

    Adopting a later candidate would create the target, and the next boot would
    then find a healthy target beside the blocked history: that history would
    fall into the "left it untouched" branch and stay there forever, which is the
    exact silent loss the blocked signal exists to prevent.
    """
    _legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    first_candidate = tmp_path / "candidate-one.db"
    second_candidate = tmp_path / "candidate-two.db"
    _create_legacy_debugger_db(first_candidate)
    _create_legacy_debugger_db(second_candidate)
    monkeypatch.setattr(
        debug_logger_module,
        "get_legacy_debugger_db_candidates",
        lambda _configured_path: [first_candidate, second_candidate],
    )
    original_link = debug_logger_module.os.link
    link_calls = []

    def _first_install_fails(*args, **kwargs):
        link_calls.append(args)
        if len(link_calls) == 1:
            raise OSError("sharing violation")
        return original_link(*args, **kwargs)

    monkeypatch.setattr(debug_logger_module.os, "link", _first_install_fails)
    caplog.set_level(logging.WARNING, logger="debug_logger")

    blocked = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert blocked.blocked_candidate == first_candidate
    assert blocked.adopted is None, "no later candidate may be adopted over a block"
    assert len(link_calls) == 1, "the second candidate was not even attempted"
    assert not target_db.exists(), "the target must stay absent for the retry"
    assert first_candidate.exists()
    assert second_candidate.exists()
    _assert_legacy_db_kept_verbatim(first_candidate)
    _assert_legacy_db_kept_verbatim(second_candidate)
    assert any(
        "remains blocked" in record.getMessage() for record in caplog.records
    )

    # The retry keeps the candidate order: the blocked history is adopted first
    # and the second candidate is preserved, never merged.
    monkeypatch.setattr(debug_logger_module.os, "link", original_link)

    retried = migrate_debugger_db_if_needed(target_db, DEFAULT_DEBUGGER_DB_NAME)

    assert retried.blocked_candidate is None
    assert retried.adopted == first_candidate
    assert retried.preserved == second_candidate
    assert target_db.exists()
    assert not first_candidate.exists()
    assert second_candidate.exists()
    assert debug_logger_module._verify_debugger_database(target_db)


def test_retirement_failure_is_reported_as_cleanup_debt_not_a_lost_adoption(
    monkeypatch,
    tmp_path,
    caplog,
):
    """A failed retire is hygiene debt: the history WAS adopted (F13).

    The old single return value conflated the two, so a successful install with
    a locked source looked exactly like an install that never happened.
    """
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    monkeypatch.setattr(
        debug_logger_module,
        "_retire_migrated_legacy_db",
        lambda _path: False,
    )
    caplog.set_level(logging.INFO, logger="debug_logger")

    outcome = migrate_debugger_db_if_needed(
        target_db,
        DEFAULT_DEBUGGER_DB_NAME,
    )

    assert outcome.adopted == legacy_db
    assert not outcome.cleanup_ok
    assert outcome.blocked_candidate is None
    assert target_db.exists()
    assert legacy_db.exists()
    _assert_legacy_db_kept_verbatim(legacy_db)
    messages = [record.getMessage() for record in caplog.records]
    assert any("requires a later retry" in message for message in messages)
    assert not any(
        message.startswith("Migrated debugger history")
        for message in messages
    )


@pytest.mark.asyncio
async def test_initialize_refuses_a_fresh_db_when_adoption_fails(
    monkeypatch,
    tmp_path,
):
    """F13: a blocked adoption disables storage instead of opening an empty DB.

    Creating the target here is what used to strand the history: on the next
    boot the fresh file verified OK, so the surviving candidate fell into the
    "left it untouched" branch and stayed there forever.
    """
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)

    def _failed_link(*_args, **_kwargs):
        raise OSError("sharing violation")

    monkeypatch.setattr(debug_logger_module.os, "link", _failed_link)
    logger_instance = DebugLogger(enabled=True, db_path=DEFAULT_DEBUGGER_DB_NAME)

    await logger_instance.initialize()
    try:
        reason = logger_instance.storage_disabled_reason
        assert reason is not None
        assert str(legacy_db) in reason
        assert "sharing violation" in reason
        assert logger_instance.storage_available is False
        assert logger_instance._pool is None
        assert not target_db.exists(), "no fresh database may shadow the candidate"
        _assert_legacy_db_kept_verbatim(legacy_db)
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_initialize_adopts_the_candidate_once_the_failure_clears(
    monkeypatch,
    tmp_path,
):
    """F13: the next boot retries the adoption and the history is there."""
    legacy_root, target_db = _configure_implicit_debugger_paths(
        monkeypatch,
        tmp_path,
    )
    legacy_db = legacy_root / DEFAULT_DEBUGGER_DB_NAME
    _create_legacy_debugger_db(legacy_db)
    original_link = debug_logger_module.os.link

    def _failed_link(*_args, **_kwargs):
        raise OSError("sharing violation")

    monkeypatch.setattr(debug_logger_module.os, "link", _failed_link)
    blocked_boot = DebugLogger(enabled=True, db_path=DEFAULT_DEBUGGER_DB_NAME)
    await blocked_boot.initialize()
    await blocked_boot.close()
    assert blocked_boot.storage_disabled_reason is not None

    monkeypatch.setattr(debug_logger_module.os, "link", original_link)
    next_boot = DebugLogger(enabled=True, db_path=DEFAULT_DEBUGGER_DB_NAME)

    await next_boot.initialize()
    try:
        assert next_boot.storage_disabled_reason is None
        assert next_boot.storage_available is True
        assert target_db.exists()
        assert not legacy_db.exists()
        fresh = await next_boot.get_session_details("fresh-session")
        assert fresh is not None
        assert json.loads(fresh["request_json"]) == LEGACY_REQUEST
        ancient = await next_boot.get_session_details("ancient-session")
        assert ancient is not None
    finally:
        await next_boot.close()


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


@pytest.mark.asyncio
async def test_update_session_status_persists_nonterminal_transition(tmp_path):
    """Normal running sessions should not remain stuck in initializing."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "session-1",
            request_payload={"request_name": "debug lifecycle"},
            status="initializing",
        )

        await logger_instance.update_session_status("session-1", status="generating")

        details = await logger_instance.get_session_details("session-1")
        assert details is not None
        assert details["status"] == "generating"
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_update_session_status_does_not_overwrite_terminal_with_late_nonterminal(tmp_path):
    """Late queued lifecycle writes must not move terminal sessions backwards."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "session-1",
            request_payload={"request_name": "debug lifecycle"},
            status="initializing",
        )
        await logger_instance.update_session_status(
            "session-1",
            status="completed",
            final_payload={"status": "completed"},
        )

        await logger_instance.update_session_status("session-1", status="qa_evaluation")

        details = await logger_instance.get_session_details("session-1")
        assert details is not None
        assert details["status"] == "completed"
        assert details["final_json"] is not None
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_update_session_status_does_not_overwrite_terminal_with_different_terminal(tmp_path):
    """Terminal writes should be idempotent once a final state is stored."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "session-1",
            request_payload={"request_name": "debug lifecycle"},
            status="initializing",
        )
        await logger_instance.update_session_status(
            "session-1",
            status="completed",
            final_payload={"status": "completed"},
        )

        await logger_instance.update_session_status(
            "session-1",
            status="failed",
            final_payload={"status": "failed"},
        )

        details = await logger_instance.get_session_details("session-1")
        assert details is not None
        assert details["status"] == "completed"
        assert json.loads(details["final_json"]) == {"status": "completed"}
    finally:
        await logger_instance.close()


# ---------------------------------------------------------------------------
# C13.1 - no deletion unless retention is explicitly configured
# ---------------------------------------------------------------------------


async def _insert_ancient_session(
    logger_instance: DebugLogger,
    session_id: str,
) -> None:
    """Record a session and backdate it far beyond any former TTL."""
    await logger_instance.record_session_start(
        session_id,
        request_payload={"request_name": "ancient"},
        status="generating",
        project_id="ancient-project",
    )
    await logger_instance.record_event(
        session_id,
        event_type="ancient-event",
        payload={"state": "observed"},
    )
    await logger_instance.record_content_trace(
        session_id=session_id,
        phase="generation",
        attempt=1,
        content_delta="ancient draft",
        expected_offset=0,
    )
    await logger_instance._pool.execute(
        """
        UPDATE sessions
        SET created_at = datetime('now', '-3650 days'),
            updated_at = datetime('now', '-3650 days')
        WHERE session_id = ?
        """,
        (session_id,),
    )
    await logger_instance._pool.commit()


@pytest.mark.parametrize("retention_days", [None, 0])
def test_unset_retention_normalizes_to_keep_forever(retention_days):
    """Unset and 0 are the same instruction: never delete anything."""
    logger_instance = DebugLogger(
        enabled=True,
        db_path=DEFAULT_DEBUGGER_DB_NAME,
        retention_days=retention_days,
    )

    assert logger_instance.retention_days is None


@pytest.mark.asyncio
async def test_without_retention_nothing_is_ever_deleted(tmp_path):
    """None of the six former TTL entry points may delete a backdated row.

    The entry points were: the CLEANUP_RETENTION_DAYS constant and its two uses
    in the migration chain, the cleanup inside initialize(), the piggyback
    cleanup, the default of cleanup_old_sessions, and the hardcoded startup call
    in core/app_state.py. This exercises every one that survives as code.
    """
    db_path = str(tmp_path / "debugger.db")
    first_logger = DebugLogger(enabled=True, db_path=db_path)
    await first_logger.initialize()
    try:
        await _insert_ancient_session(first_logger, "ancient-session")
    finally:
        await first_logger.close()

    restarted = DebugLogger(enabled=True, db_path=db_path)
    await restarted.initialize()
    try:
        # initialize() must not have run any cleanup at all.
        assert await restarted.get_session_details("ancient-session") is not None

        # The piggyback path must not even create a task; a write is the trigger.
        assert restarted._spawn_cleanup_task() is None
        assert not restarted._bg_tasks
        await restarted.record_event(
            "ancient-session",
            event_type="write-trigger",
            payload={"still": "here"},
        )
        assert not restarted._bg_tasks

        # The parameter default must be inert as well.
        result = await restarted.cleanup_old_sessions()
        assert result["status"] == "retention_disabled"
        assert result["deleted_sessions"] == 0

        assert await restarted.get_session_details("ancient-session") is not None
        assert await restarted.get_session_events("ancient-session")
        assert await restarted.get_content_traces("ancient-session")
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_configured_retention_deletes_sessions_events_and_traces(tmp_path):
    """A positive window restores cleanup, content traces included."""
    logger_instance = DebugLogger(
        enabled=True,
        db_path=str(tmp_path / "debugger.db"),
        retention_days=7,
    )
    await logger_instance.initialize()
    try:
        await _insert_ancient_session(logger_instance, "ancient-session")
        await logger_instance.record_session_start(
            "recent-session",
            request_payload={"request_name": "recent"},
            status="generating",
        )
        await logger_instance.record_content_trace(
            session_id="recent-session",
            phase="generation",
            attempt=1,
            content_delta="recent draft",
            expected_offset=0,
        )

        result = await logger_instance.cleanup_old_sessions()

        assert result["status"] == "ok"
        assert result["deleted_sessions"] == 1
        assert result["deleted_events"] == 1
        assert result["retention_days"] == 7
        assert await logger_instance.get_session_details("ancient-session") is None
        assert await logger_instance.get_content_traces("ancient-session") == []
        assert await logger_instance.get_session_details("recent-session") is not None
        assert await logger_instance.get_content_traces("recent-session")
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_configured_retention_cleans_up_on_initialize(tmp_path):
    """With a window configured, startup cleanup still runs."""
    db_path = str(tmp_path / "debugger.db")
    first_logger = DebugLogger(enabled=True, db_path=db_path, retention_days=7)
    await first_logger.initialize()
    try:
        await _insert_ancient_session(first_logger, "ancient-session")
    finally:
        await first_logger.close()

    restarted = DebugLogger(enabled=True, db_path=db_path, retention_days=7)
    await restarted.initialize()
    try:
        assert await restarted.get_session_details("ancient-session") is None
    finally:
        await restarted.close()


# ---------------------------------------------------------------------------
# C13.2 - verbatim capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_and_sources_are_persisted_verbatim(tmp_path):
    """Nothing is redacted or dropped on write."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    try:
        payload = {
            "request_name": "verbatim session",
            "prompt": "write something",
            "system_prompt": "private system instruction",
            "nested": {"revision_prompt": "private revision instruction"},
            "source_text": "FULL SOURCE CONTENT",
            "cumulative_text": "FULL CUMULATIVE CONTENT",
        }

        await logger_instance.record_session_start(
            "verbatim-session",
            request_payload=payload,
        )
        await logger_instance.record_event(
            "verbatim-session",
            event_type="generator_output",
            payload={"user_prompt": "event prompt", "content": "generated text"},
        )

        details = await logger_instance.get_session_details("verbatim-session")
        assert details is not None
        assert json.loads(details["request_json"]) == payload
        events = await logger_instance.get_session_events("verbatim-session")
        assert json.loads(events[0]["payload_json"]) == {
            "user_prompt": "event prompt",
            "content": "generated text",
        }
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_startup_does_not_scrub_a_preexisting_database(tmp_path):
    """A restart leaves stored payloads byte-identical (no startup scrub)."""
    db_path = str(tmp_path / "debugger.db")
    payload = {"prompt": "STORED PROMPT", "source_text": "STORED SOURCE"}
    first_logger = DebugLogger(enabled=True, db_path=db_path)
    await first_logger.initialize()
    try:
        await first_logger.record_session_start(
            "scrub-target",
            request_payload=payload,
        )
        await first_logger.record_event(
            "scrub-target",
            event_type="legacy-event",
            payload={"nested": {"system_prompt": "STORED SYSTEM PROMPT"}},
        )
    finally:
        await first_logger.close()

    restarted = DebugLogger(enabled=True, db_path=db_path)
    await restarted.initialize()
    try:
        details = await restarted.get_session_details("scrub-target")
        assert details is not None
        assert json.loads(details["request_json"]) == payload
        events = await restarted.get_session_events("scrub-target")
        assert json.loads(events[0]["payload_json"]) == {
            "nested": {"system_prompt": "STORED SYSTEM PROMPT"},
        }
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_malformed_stored_payload_is_neither_rewritten_nor_fatal(tmp_path):
    """Tolerant deserialization survives; the stored bytes are left alone."""
    db_path = str(tmp_path / "debugger.db")
    first_logger = DebugLogger(enabled=True, db_path=db_path)
    await first_logger.initialize()
    try:
        await first_logger.record_session_start(
            "malformed-session",
            request_payload={"request_name": "malformed"},
        )
        await first_logger._pool.execute(
            """
            UPDATE sessions
            SET preflight_json = 'NOT VALID JSON'
            WHERE session_id = 'malformed-session'
            """
        )
        await first_logger._pool.commit()
    finally:
        await first_logger.close()

    restarted = DebugLogger(enabled=True, db_path=db_path)
    await restarted.initialize()
    try:
        details = await restarted.get_session_details("malformed-session")
        assert details is not None
        assert details["preflight_json"] == "NOT VALID JSON"
    finally:
        await restarted.close()


# ---------------------------------------------------------------------------
# C13.3 - the verify chain keeps integrity checks and nothing else
# ---------------------------------------------------------------------------


def test_verify_rejects_a_corrupt_database(tmp_path):
    """Corruption detection matters more in a store that now lives forever."""
    database_path = tmp_path / "corrupt.db"
    _create_legacy_debugger_db(database_path)
    payload = bytearray(database_path.read_bytes())
    # Scribble over the page area, leaving the SQLite header intact so the file
    # is still opened as a database and fails on quick_check, not on open.
    for offset in range(2048, min(len(payload), 6144)):
        payload[offset] = 0x5A
    database_path.write_bytes(bytes(payload))

    assert debug_logger_module._verify_debugger_database(database_path) is False


def test_verify_rejects_a_sqlite_file_without_debugger_tables(tmp_path):
    """A sound but unrelated SQLite file is not adopted as debugger history."""
    database_path = tmp_path / "unrelated.db"
    with closing(sqlite3.connect(str(database_path))) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.commit()

    assert debug_logger_module._verify_debugger_database(database_path) is False
    assert debug_logger_module._verify_sqlite_database(database_path) is True


def test_verify_accepts_a_wal_database_with_a_live_sidecar(tmp_path):
    """A WAL sidecar from an unclean shutdown is replayed, not called corrupt."""
    database_path = tmp_path / "wal.db"
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO sessions VALUES ('kept')")
        connection.commit()
        assert Path(f"{database_path}-wal").exists()

        assert debug_logger_module._verify_debugger_database(database_path) is True
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# C13.5 - phase denormalized onto the session row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_phase_is_stored_and_listed(tmp_path):
    """The session list exposes the phase without replaying the event timeline."""
    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    try:
        await logger_instance.record_session_start(
            "phase-session",
            request_payload={"request_name": "phase"},
            status="initializing",
            project_id="phase-project",
        )

        await logger_instance.update_session_status(
            "phase-session",
            status="generating",
            phase="generation",
        )
        await logger_instance.update_session_status(
            "phase-session",
            phase="qa_evaluation",
        )

        listed = await logger_instance.list_sessions(project_id="phase-project")
        assert [entry["phase"] for entry in listed] == ["qa_evaluation"]
        assert listed[0]["status"] == "generating"
        details = await logger_instance.get_session_details("phase-session")
        assert details is not None
        assert details["phase"] == "qa_evaluation"
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_phase_column_is_added_to_a_preexisting_database(tmp_path):
    """The phase migration follows the project_id pattern on old databases."""
    db_path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(str(db_path))) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                status TEXT,
                request_json TEXT,
                preflight_json TEXT,
                final_json TEXT,
                usage_json TEXT,
                notes TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO sessions (session_id, status) VALUES ('old-row', 'completed')"
        )
        connection.commit()

    logger_instance = DebugLogger(enabled=True, db_path=str(db_path))
    await logger_instance.initialize()
    try:
        await logger_instance.update_session_status("old-row", phase="gran_sabio_review")

        details = await logger_instance.get_session_details("old-row")
        assert details is not None
        assert details["phase"] == "gran_sabio_review"
    finally:
        await logger_instance.close()


@pytest.mark.asyncio
async def test_engine_phase_updates_reach_the_session_row(tmp_path, monkeypatch):
    """A phase-only lifecycle transition is persisted, not just its event.

    ``update_session_phase`` reports a phase with no status change; before
    C13.5 the status-update path was skipped entirely in that case, so the
    column would have stayed empty for most of a run.
    """
    from core import app_state as app_state_module

    logger_instance = DebugLogger(enabled=True, db_path=str(tmp_path / "debugger.db"))
    await logger_instance.initialize()
    monkeypatch.setattr(app_state_module, "debug_logger", logger_instance)
    try:
        await logger_instance.record_session_start(
            "engine-session",
            request_payload={"request_name": "engine"},
            status="initializing",
        )

        await app_state_module._debug_record_session_transition(
            "engine-session",
            status="generating",
            phase="generation",
        )
        await app_state_module._debug_record_session_transition(
            "engine-session",
            phase="qa_evaluation",
        )

        details = await logger_instance.get_session_details("engine-session")
        assert details is not None
        assert details["status"] == "generating"
        assert details["phase"] == "qa_evaluation"
    finally:
        await logger_instance.close()
