"""
Debug logger for Gran Sabio LLM Engine.

Persists a verbatim request/response timeline to SQLite so developers can
inspect each session in chronological order via the /debugger interface.

Persistence notes (what is stored, where, and for how long):
- Storage is an unencrypted SQLite database. The default path lives under
  per-user local app storage; DEBUGGER_DB_PATH overrides it.
- Capture is VERBATIM (C13.2): prompts, source material, generated content, QA
  payloads and usage are stored exactly as the engine saw them. Absolute
  traceability is the point of this database; unbounded growth is an accepted
  trade-off.
- Nothing is deleted unless retention is explicitly configured (C13.1).
  ``DEBUGGER_RETENTION_DAYS`` unset or ``0`` means keep forever, and no cleanup
  task is even created. A positive value restores time-based cleanup for
  sessions, their events and their content traces.
- All rows, including non-terminal sessions, are telemetry. This database is
  never an authority for live sessions, projects, cancellation, or restart
  recovery.
- ``content_traces`` stores the verbatim trace of streamed content, batched as
  one row per (session, phase, attempt) so looking back at what a run produced
  does not depend on somebody having been subscribed to the stream.
- Treat the database as sensitive: do not enable the debugger on shared
  machines when processing private data. Access to /debugger/* stays behind the
  internal-IP filter and the DEBUGGER_EXPOSE_ENABLED kill switch.
"""

import asyncio
import hashlib
import logging
import os
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Set, TypeVar

import aiosqlite

import json_utils as json

logger = logging.getLogger(__name__)

# Piggyback cleanup configuration. There is no default retention: cleanup only
# runs when the deployment configures DEBUGGER_RETENTION_DAYS to a positive
# value (C13.1), and this interval only bounds how often it may then run.
CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour between opportunistic cleanups
DEFAULT_DEBUGGER_DB_NAME = "debugger_history.db"
TERMINAL_DEBUGGER_STATUSES = frozenset(
    {"completed", "rejected", "failed", "cancelled", "interrupted"}
)

TERMINAL_SQLITE_ERROR_MARKERS = (
    "disk i/o error",
    "database or disk is full",
    "database disk image is malformed",
    "file is not a database",
    "readonly database",
    "unable to open database file",
    "permission denied",
    "access is denied",
)

T = TypeVar("T")


def is_explicit_debugger_db_path(db_path: str) -> bool:
    """Return whether the debugger path was explicitly configured."""
    return os.getenv("DEBUGGER_DB_PATH") is not None or db_path != DEFAULT_DEBUGGER_DB_NAME


def resolve_debugger_db_path(db_path: str) -> Path:
    """Resolve the debugger DB path, relocating the implicit default to local app storage."""
    if is_explicit_debugger_db_path(db_path):
        return Path(db_path).expanduser()

    if os.name == "nt":
        base_dir = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        if base_dir:
            return Path(base_dir) / "GranSabio_LLM" / "debugger" / DEFAULT_DEBUGGER_DB_NAME
        return Path.home() / "AppData" / "Local" / "GranSabio_LLM" / "debugger" / DEFAULT_DEBUGGER_DB_NAME

    xdg_state_home = os.getenv("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / "gransabio" / "debugger" / DEFAULT_DEBUGGER_DB_NAME

    return Path.home() / ".local" / "state" / "gransabio" / "debugger" / DEFAULT_DEBUGGER_DB_NAME


def get_legacy_debugger_db_candidates(db_path: str) -> List[Path]:
    """Return legacy default-path candidates that may contain existing debugger history."""
    if is_explicit_debugger_db_path(db_path):
        return []

    candidates: List[Path] = []
    seen: set[str] = set()
    for candidate in (
        Path.cwd() / DEFAULT_DEBUGGER_DB_NAME,
        Path(__file__).resolve().parent / DEFAULT_DEBUGGER_DB_NAME,
    ):
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def _sqlite_readonly_uri(path: Path) -> str:
    """Return a SQLite URI that cannot create or mutate the source database."""
    return f"{path.resolve().as_uri()}?mode=ro"


def _quick_check_ok(connection: sqlite3.Connection) -> bool:
    """Return whether SQLite reports the open database as structurally sound."""
    rows = connection.execute("PRAGMA quick_check").fetchall()
    return bool(rows) and all(str(row[0]).casefold() == "ok" for row in rows)


def _verify_sqlite_database(path: Path) -> bool:
    """Return whether an existing file is a structurally sound SQLite database.

    The connection is opened read-write on purpose: the managed debugger
    database runs in WAL mode, and a read-only connection cannot replay a
    ``-wal`` sidecar left behind by an unclean shutdown. Reporting that as
    corruption would push a perfectly recoverable database down the discard
    path.
    """
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(str(path), timeout=5)) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            return _quick_check_ok(connection)
    except (OSError, sqlite3.Error):
        return False


def _sha256_file(path: Path) -> str:
    """Hash a database snapshot without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_sqlite_sidecars(path: Path) -> tuple[Path, ...]:
    """Return SQLite files that may contain journaled copies of old payloads."""
    return tuple(Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))


def _verify_debugger_connection(connection: sqlite3.Connection) -> bool:
    """Verify SQLite integrity and debugger schema shape on an open database.

    This is the whole contract a debugger database has to satisfy (C13.3).
    Capture is verbatim, so there is nothing to check about payload content;
    what still matters — more than before, since the database now lives forever
    and grows without a ceiling — is that SQLite reports it as structurally
    sound and that it really is a debugger database rather than an unrelated
    file that happens to sit at the same path.
    """
    if not _quick_check_ok(connection):
        return False

    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    return "sessions" in table_names


def _verify_debugger_database(path: Path) -> bool:
    """Verify that a file is a structurally sound debugger database.

    Opened read-write for the same reason as :func:`_verify_sqlite_database`: a
    WAL sidecar from an unclean shutdown must be replayed, not mistaken for
    corruption.
    """
    if not path.exists():
        return False
    try:
        with closing(sqlite3.connect(str(path), timeout=5)) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            return _verify_debugger_connection(connection)
    except (OSError, sqlite3.Error):
        return False


def _wipe_file(path: Path) -> bool:
    """Overwrite a non-database artifact so any surviving file is harmless."""
    if not path.exists():
        return True
    try:
        remaining = path.stat().st_size
        zero_block = b"\0" * (1024 * 1024)
        with path.open("r+b", buffering=0) as stream:
            while remaining > 0:
                chunk_size = min(remaining, len(zero_block))
                stream.write(zero_block[:chunk_size])
                remaining -= chunk_size
            stream.truncate(0)
            stream.flush()
            os.fsync(stream.fileno())
        return True
    except OSError as exc:
        logger.error("Failed to wipe sensitive debugger artifact %s: %s", path, exc)
        return False


def _wipe_and_unlink(path: Path) -> bool:
    """Wipe an artifact before unlinking; a survivor contains only zeros."""
    if not path.exists():
        return True
    if not _wipe_file(path):
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "Sanitized debugger artifact %s could not be removed: %s",
            path,
            exc,
        )
    return True


def _cleanup_sqlite_sidecars(path: Path) -> bool:
    """Sanitize every SQLite sidecar before the main database is retired."""
    results = [
        _wipe_and_unlink(sidecar)
        for sidecar in _legacy_sqlite_sidecars(path)
    ]
    return all(results)


def _create_debugger_db_snapshot(
    source_path: Path,
    target_dir: Path,
) -> tuple[Path, str]:
    """Back up a verified debugger source into a temporary migration file."""
    if not _verify_debugger_database(source_path):
        raise sqlite3.DatabaseError(
            "Debugger source failed pre-snapshot integrity verification"
        )

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=target_dir,
        prefix=f".{DEFAULT_DEBUGGER_DB_NAME}.migration-",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        with closing(
            sqlite3.connect(
                _sqlite_readonly_uri(source_path),
                uri=True,
                timeout=5,
            )
        ) as source_connection:
            source_connection.execute("PRAGMA query_only=ON")
            with closing(
                sqlite3.connect(str(temporary_path), timeout=5)
            ) as target_connection:
                source_connection.backup(target_connection)
                target_connection.commit()

        if not _verify_debugger_database(temporary_path):
            raise sqlite3.DatabaseError(
                "Debugger migration snapshot failed verification"
            )
        return temporary_path, _sha256_file(temporary_path)
    except Exception:
        _wipe_and_unlink(temporary_path)
        raise


def _cleanup_migration_snapshot(
    snapshot_path: Path,
    target_path: Path,
) -> bool:
    """Drop a migration temp link without corrupting a linked target."""
    if not snapshot_path.exists():
        return True
    snapshot_safe = (
        _is_zero_filled(snapshot_path)
        or _verify_debugger_database(snapshot_path)
    )
    try:
        linked_to_target = (
            target_path.exists()
            and os.path.samefile(snapshot_path, target_path)
        )
    except OSError:
        if target_path.exists():
            logger.error(
                "Could not prove whether migration snapshot %s aliases target %s",
                snapshot_path,
                target_path,
            )
            return False
        linked_to_target = False

    if linked_to_target:
        if not snapshot_safe:
            logger.error(
                "Refusing to wipe migration snapshot %s because it aliases "
                "the target and is not verifiably intact",
                snapshot_path,
            )
            return False
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Migration link %s could not be removed: %s",
                snapshot_path,
                exc,
            )
        return True
    if snapshot_safe:
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Migration snapshot %s could not be removed: %s",
                snapshot_path,
                exc,
            )
        return True
    return _wipe_and_unlink(snapshot_path)


def _cleanup_orphan_migration_snapshots(target_path: Path) -> bool:
    """Retire stale migration temporaries left by interrupted startups."""
    snapshots = target_path.parent.glob(
        f".{DEFAULT_DEBUGGER_DB_NAME}.migration-*.tmp"
    )
    results = [
        _cleanup_migration_snapshot(snapshot, target_path)
        for snapshot in snapshots
    ]
    return all(results)


def _retire_migrated_legacy_db(path: Path) -> bool:
    """Retire sidecars first, then remove or retain a verified legacy DB."""
    if not _verify_debugger_database(path):
        return False
    if not _cleanup_sqlite_sidecars(path):
        return False
    try:
        path.unlink(missing_ok=True)
        logger.info("Removed migrated legacy debugger history at %s", path)
        return True
    except OSError as exc:
        if _verify_debugger_database(path):
            logger.warning(
                "Migrated legacy debugger history at %s could not be removed: %s",
                path,
                exc,
            )
            return True
        return False


def _is_zero_filled(path: Path) -> bool:
    """Return whether a file contains no data other than zero bytes."""
    if not path.exists():
        return False
    try:
        with path.open("rb") as stream:
            return all(
                not any(chunk)
                for chunk in iter(lambda: stream.read(1024 * 1024), b"")
            )
    except OSError:
        return False


def _discard_invalid_new_target(path: Path) -> bool:
    """Wipe and remove a newly linked target that failed verification."""
    if not path.exists():
        return True
    if not _cleanup_sqlite_sidecars(path):
        return False
    if not _wipe_file(path):
        return False
    try:
        path.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.error(
            "Wiped invalid debugger migration target %s could not be removed: %s",
            path,
            exc,
        )
        return False


class _CandidateInstall(NamedTuple):
    """What happened to ONE legacy candidate, with nothing conflated.

    ``source_verified`` says the file WAS sound debugger history, which is what
    makes a failed ``installed`` a blocking condition instead of a skipped junk
    file. ``cleanup_ok`` is hygiene of the leftovers (source, snapshots,
    sidecars) and never invalidates a successful install.
    """

    installed: bool
    cleanup_ok: bool
    source_verified: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class DebuggerMigrationOutcome:
    """What the startup migration achieved, reported honestly.

    ``adopted`` is the candidate whose history now lives at the managed target.
    ``preserved`` is a sound candidate deliberately left alone because a healthy
    target already existed (two histories, no merge). ``blocked_candidate`` is a
    sound candidate that could NOT be installed: the target was left absent so a
    later boot can retry, and the caller must NOT open a fresh database over it.
    ``cleanup_ok`` reports leftover hygiene only.
    """

    adopted: Optional[Path] = None
    preserved: Optional[Path] = None
    blocked_candidate: Optional[Path] = None
    blocked_error: Optional[str] = None
    cleanup_ok: bool = True


def _migrate_legacy_candidate(
    source_path: Path,
    target_path: Path,
) -> _CandidateInstall:
    """Install one verified candidate and explicitly report what is left.

    The candidate is adopted AS IS (C13.3): its payloads are never rewritten and
    its rows are never expired. The only precondition is that it verifies as a
    structurally sound debugger database.
    """
    migration_snapshot: Optional[Path] = None
    verification_snapshot: Optional[Path] = None
    installed = False
    source_verified = False
    install_error: Optional[str] = None
    target_linked = False
    cleanup_ok = True

    try:
        if not _verify_debugger_database(source_path):
            raise sqlite3.DatabaseError(
                "Legacy debugger source failed integrity verification"
            )
        source_verified = True
        migration_snapshot, snapshot_digest = _create_debugger_db_snapshot(
            source_path,
            target_path.parent,
        )
        os.link(migration_snapshot, target_path)
        target_linked = True

        if (
            not _verify_debugger_database(target_path)
            or _sha256_file(target_path) != snapshot_digest
        ):
            cleanup_ok = _discard_invalid_new_target(target_path)
            raise sqlite3.DatabaseError(
                "Installed debugger migration failed integrity verification"
            )
        installed = True

        current_digest: Optional[str] = None
        try:
            verification_snapshot, current_digest = (
                _create_debugger_db_snapshot(
                    source_path,
                    target_path.parent,
                )
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning(
                "Could not re-snapshot legacy debugger history at %s: %s",
                source_path,
                exc,
            )

        if current_digest == snapshot_digest:
            retired = _retire_migrated_legacy_db(source_path)
            cleanup_ok = cleanup_ok and retired
            if not retired:
                logger.error(
                    "Installed debugger target at %s, but retirement "
                    "of source %s remains incomplete",
                    target_path,
                    source_path,
                )
        else:
            logger.warning(
                "Legacy debugger history at %s changed or could not be "
                "re-snapshotted; retained the source untouched",
                source_path,
            )
    except (OSError, sqlite3.Error) as exc:
        install_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Failed to migrate legacy debugger history from %s to %s: %s",
            source_path,
            target_path,
            exc,
        )
        if not installed and target_linked and target_path.exists():
            cleanup_ok = _discard_invalid_new_target(target_path) and cleanup_ok
    finally:
        if verification_snapshot is not None:
            cleanup_ok = (
                _cleanup_migration_snapshot(
                    verification_snapshot,
                    target_path,
                )
                and cleanup_ok
            )
        if migration_snapshot is not None:
            cleanup_ok = (
                _cleanup_migration_snapshot(
                    migration_snapshot,
                    target_path,
                )
                and cleanup_ok
            )

    if installed and cleanup_ok:
        logger.info(
            "Migrated debugger history from %s to %s; all surviving files "
            "are verified intact",
            source_path,
            target_path,
        )
    elif installed:
        logger.error(
            "Debugger history was installed at %s, but cleanup for %s "
            "requires a later retry",
            target_path,
            source_path,
        )
    return _CandidateInstall(
        installed=installed,
        cleanup_ok=cleanup_ok,
        source_verified=source_verified,
        error=install_error,
    )


def migrate_debugger_db_if_needed(
    target_path: Path,
    configured_db_path: str,
) -> DebuggerMigrationOutcome:
    """Move legacy history to the managed path without ever rewriting it.

    Databases are adopted verbatim (C13.3). A file is only wiped when it is not
    a usable debugger database at all and is therefore blocking the managed
    path; never because of what it contains or how old its rows are.

    A sound candidate that could not be installed is reported as
    ``blocked_candidate``: the caller must keep the target absent so the next
    boot retries, because a fresh database at the target would verify OK and
    strand that history in the "left it untouched" branch forever. That block
    takes precedence over every candidate the deterministic candidate order
    places after it: none of them is adopted while it stands, because creating
    the target would strand the blocked history exactly as a fresh database
    would.
    """
    if is_explicit_debugger_db_path(configured_db_path):
        return DebuggerMigrationOutcome()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_resolved = target_path.resolve()
    except Exception:
        target_resolved = target_path

    all_cleanup_ok = True
    adopted_candidate: Optional[Path] = None
    preserved_candidate: Optional[Path] = None
    blocked_candidate: Optional[Path] = None
    blocked_error: Optional[str] = None
    seen_candidates: set[str] = set()

    if target_path.exists():
        if _is_zero_filled(target_path):
            if not _cleanup_sqlite_sidecars(target_path):
                all_cleanup_ok = False
            else:
                try:
                    target_path.unlink()
                except OSError as exc:
                    logger.error(
                        "Wiped debugger migration target %s still blocks "
                        "retry: %s",
                        target_path,
                        exc,
                    )
                    all_cleanup_ok = False
        elif not _verify_debugger_database(target_path):
            if not _verify_sqlite_database(target_path):
                sidecars_safe = _cleanup_sqlite_sidecars(target_path)
                removed_or_wiped = (
                    _wipe_and_unlink(target_path)
                    if sidecars_safe
                    else False
                )
                target_unblocked = not target_path.exists()
                if target_unblocked:
                    logger.warning(
                        "Existing unusable debugger target %s was retired",
                        target_path,
                    )
                else:
                    logger.error(
                        "Existing unusable debugger target %s was wiped but "
                        "still blocks migration",
                        target_path,
                    )
                all_cleanup_ok = (
                    removed_or_wiped
                    and target_unblocked
                    and all_cleanup_ok
                )
            else:
                logger.error(
                    "Existing debugger target %s is a sound SQLite file "
                    "without debugger tables; legacy candidates will be "
                    "preserved rather than merged",
                    target_path,
                )
                all_cleanup_ok = False

    all_cleanup_ok = (
        _cleanup_orphan_migration_snapshots(target_path)
        and all_cleanup_ok
    )

    for candidate in get_legacy_debugger_db_candidates(configured_db_path):
        try:
            candidate_resolved = candidate.resolve()
        except Exception:
            candidate_resolved = candidate
        candidate_key = os.path.normcase(str(candidate_resolved))
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)

        if not candidate.exists():
            all_cleanup_ok = (
                _cleanup_sqlite_sidecars(candidate)
                and all_cleanup_ok
            )
            continue
        if candidate_resolved == target_resolved:
            continue

        if blocked_candidate is not None:
            # An earlier sound candidate failed to install, so the target must
            # stay absent for the next boot to retry THAT history first.
            # Adopting this one would create the target and strand the blocked
            # candidate in the "left it untouched" branch forever, which is the
            # exact silent loss the blocked signal exists to prevent.
            logger.warning(
                "Not attempting adoption of legacy debugger history at %s while "
                "the sound candidate %s remains blocked; left it untouched",
                candidate,
                blocked_candidate,
            )
            continue

        if target_path.exists():
            if _verify_debugger_database(candidate):
                if preserved_candidate is None:
                    preserved_candidate = candidate
                logger.warning(
                    "Found legacy debugger history at %s beside an existing "
                    "target; left it untouched rather than assuming "
                    "equivalence or discarding history",
                    candidate,
                )
            else:
                all_cleanup_ok = False
            continue

        install = _migrate_legacy_candidate(candidate, target_path)
        if install.installed:
            if adopted_candidate is None:
                adopted_candidate = candidate
        elif install.source_verified and blocked_candidate is None:
            # Sound history that did not make it across. The target must stay
            # absent so the next boot can retry the adoption.
            blocked_candidate = candidate
            blocked_error = install.error
        all_cleanup_ok = install.cleanup_ok and all_cleanup_ok

    all_cleanup_ok = (
        _cleanup_orphan_migration_snapshots(target_path)
        and all_cleanup_ok
    )
    return DebuggerMigrationOutcome(
        adopted=adopted_candidate,
        preserved=preserved_candidate,
        blocked_candidate=blocked_candidate,
        blocked_error=blocked_error,
        cleanup_ok=all_cleanup_ok,
    )


def is_terminal_debugger_storage_error(exc: BaseException) -> bool:
    """Return whether the error indicates unusable debugger storage."""
    if isinstance(exc, OSError):
        return True

    if not isinstance(exc, sqlite3.Error):
        return False

    message = str(exc).lower()
    return any(marker in message for marker in TERMINAL_SQLITE_ERROR_MARKERS)


def _normalize_retention_days(retention_days: Optional[int]) -> Optional[int]:
    """Return a positive retention window, or None when nothing expires.

    ``None`` and ``0`` are the same instruction (C13.1): keep the telemetry
    forever. Negative values would mean "delete rows from the future" and are
    treated as unconfigured rather than silently applied.
    """
    if retention_days is None:
        return None
    window = int(retention_days)
    return window if window > 0 else None


def _decode_trace_segments(segments_json: Optional[str]) -> List[Dict[str, Any]]:
    """Decode stored trace label boundaries, tolerating malformed rows."""
    if not segments_json:
        return []
    try:
        decoded = json.loads(segments_json)
    except Exception:
        logger.warning("Discarding malformed content trace segments")
        return []
    return decoded if isinstance(decoded, list) else []


class DebugLogger:
    """Centralized, non-authoritative persistence for debugging sessions."""

    def __init__(
        self,
        *,
        enabled: bool,
        db_path: str,
        retention_days: Optional[int] = None,
    ):
        self.enabled = enabled
        self.configured_db_path = db_path
        self.db_path = resolve_debugger_db_path(db_path)
        # Single retention setting, read once (C13.1). None means "keep
        # forever"; no cleanup task is created at all in that case.
        self.retention_days = _normalize_retention_days(retention_days)
        self._pool: Optional[aiosqlite.Connection] = None
        self._init_lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._event_counters: Dict[str, int] = {}
        self._last_cleanup_ts: float = 0.0
        self._cleanup_running: bool = False
        self._storage_disabled_reason: Optional[str] = None
        self._bg_tasks: Set[asyncio.Task] = set()

    @property
    def storage_disabled_reason(self) -> Optional[str]:
        """Explain why debugger persistence was disabled, if it was."""
        return self._storage_disabled_reason

    @property
    def storage_available(self) -> bool:
        """Return whether debugger persistence can still be used."""
        return self.enabled and self._storage_disabled_reason is None

    async def _disable_storage(self, exc: BaseException) -> None:
        """Disable debugger persistence after a terminal storage error."""
        await self._disable_storage_with_reason(f"{type(exc).__name__}: {exc}")

    async def _disable_storage_with_reason(self, reason: str) -> None:
        """Disable debugger persistence and record why, logging once."""
        first_disable = self._storage_disabled_reason is None
        self._storage_disabled_reason = reason
        await self.close()
        if first_disable:
            logger.warning(
                "Debugger persistence disabled for database %s: %s",
                self.db_path,
                reason,
            )

    async def _run_guarded(self, operation: Callable[[], Awaitable[T]], *, default: T) -> T:
        """Run a storage operation and disable persistence on terminal I/O failures."""
        if not self.storage_available or self._pool is None:
            return default

        try:
            return await operation()
        except Exception as exc:
            if is_terminal_debugger_storage_error(exc):
                await self._disable_storage(exc)
                return default
            raise

    async def initialize(self) -> None:
        """Open SQLite connection and create schema if needed."""
        if not self.storage_available:
            return

        async with self._init_lock:
            if not self.storage_available or self._pool is not None:
                return

            try:
                migration = migrate_debugger_db_if_needed(
                    self.db_path,
                    self.configured_db_path,
                )
                if migration.blocked_candidate is not None:
                    # Fail fast, loudly, and WITHOUT touching the target: opening
                    # a fresh database here would verify OK on the next boot, so
                    # the surviving history would land in the "left it untouched"
                    # branch forever. The engine keeps serving; only the debugger
                    # store stays off until a boot manages to adopt it.
                    await self._disable_storage_with_reason(
                        f"legacy debugger history at {migration.blocked_candidate} "
                        "could not be adopted: "
                        f"{migration.blocked_error or 'installation failed'}"
                    )
                    return
                if not migration.cleanup_ok:
                    # History is already in place; what is left over is disk
                    # hygiene, so it must be visible but must not stop startup.
                    logger.warning(
                        "Debugger migration left files behind that need a later "
                        "retry (adopted=%s, preserved=%s)",
                        migration.adopted,
                        migration.preserved,
                    )
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._pool = await aiosqlite.connect(self.db_path.as_posix())
                await self._set_journal_mode_wal()
                await self._pool.execute("PRAGMA synchronous=NORMAL")
                await self._pool.execute("PRAGMA foreign_keys=ON")
                await self._create_schema()
                await self._pool.commit()
                if self.retention_days is not None:
                    await self.cleanup_old_sessions()
                    self._last_cleanup_ts = time.monotonic()
                    if not self.storage_available:
                        return
                if not is_explicit_debugger_db_path(self.configured_db_path):
                    logger.warning(
                        "Debugger DB default path relocated from %s to %s. "
                        "Set DEBUGGER_DB_PATH to override this expected location.",
                        self.configured_db_path,
                        self.db_path,
                    )
                logger.info("Debug logger initialized with database at %s", self.db_path)
            except Exception as exc:
                if is_terminal_debugger_storage_error(exc):
                    await self._disable_storage(exc)
                    return
                await self.close()
                raise

    async def _set_journal_mode_wal(self) -> None:
        """Switch the debugger connection to WAL and report the result.

        The draft trace (C0.8) turns the debugger into a periodic writer: one
        batched UPDATE per second and per active (session, phase, attempt) row,
        on top of the existing event inserts. WAL removes the rollback-journal
        rewrite from every commit and lets the read endpoints work while a run
        is streaming. ``synchronous=NORMAL`` under WAL fsyncs at checkpoints
        instead of at every commit, which is the right durability trade for
        telemetry that is never a control plane.

        SQLite reports the mode it actually adopted: filesystems that cannot
        support shared memory keep the previous journal mode, and that is
        logged as a warning rather than hidden.
        """
        if self._pool is None:
            return

        async with self._pool.execute("PRAGMA journal_mode=WAL") as cursor:
            row = await cursor.fetchone()
        mode = str(row[0]).lower() if row else "unknown"
        if mode != "wal":
            logger.warning(
                "Debugger database at %s kept journal mode %s; WAL was refused "
                "by the filesystem",
                self.db_path,
                mode,
            )

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
                phase TEXT,
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
        await self._ensure_sessions_columns()
        await self._ensure_content_traces_table()

    async def _ensure_sessions_columns(self) -> None:
        """Add the denormalized sessions columns to pre-existing databases.

        Idempotent migration: ``project_id`` and ``request_name`` predate this
        change, ``phase`` arrives with C13.5 so the session list can show what a
        run was doing without opening its event timeline.
        """
        if self._pool is None:
            return

        async with self._pool.execute("PRAGMA table_info(sessions)") as cursor:
            columns = await cursor.fetchall()

        col_names = {col[1] for col in columns}

        if "project_id" not in col_names:
            await self._pool.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")

        if "request_name" not in col_names:
            await self._pool.execute("ALTER TABLE sessions ADD COLUMN request_name TEXT")

        if "phase" not in col_names:
            await self._pool.execute("ALTER TABLE sessions ADD COLUMN phase TEXT")

        await self._pool.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id)"
        )
        await self._pool.commit()

    _CONTENT_TRACES_COLUMNS = (
        "id",
        "session_id",
        "phase",
        "attempt",
        "model",
        "layer",
        "project_id",
        "subphase",
        "iteration",
        "content_status",
        "content",
        "content_chars",
        "segments",
        "created_at",
        "updated_at",
    )
    _CONTENT_TRACES_UNIQUE_KEY = ("session_id", "phase", "attempt", "model", "layer")
    _CONTENT_TRACES_TABLE_DDL = """
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                layer TEXT NOT NULL DEFAULT '',
                project_id TEXT,
                subphase TEXT,
                iteration INTEGER,
                content_status TEXT,
                content TEXT NOT NULL DEFAULT '',
                content_chars INTEGER NOT NULL DEFAULT 0,
                segments TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, phase, attempt, model, layer)
            )"""
    _CONTENT_TRACES_INDEX_DDL = """
            CREATE INDEX IF NOT EXISTS idx_content_traces_session
                ON content_traces(session_id, phase, attempt);

            CREATE INDEX IF NOT EXISTS idx_content_traces_updated
                ON content_traces(updated_at DESC);"""

    async def _ensure_content_traces_table(self) -> None:
        """Create or rebuild the draft-trace table to the current contract.

        Idempotent migration (same pattern as ``_ensure_sessions_columns``),
        run on every initialization. A database created before the visibility
        restoration simply gains the table on the next start; a database whose
        ``content_traces`` predates ``model``/``layer`` joining the row
        identity is rebuilt in place with every existing row copied verbatim
        (columns the old shape lacked take their defaults). The rebuild path
        exists because the row identity lives in a table UNIQUE constraint and
        SQLite cannot alter constraints: the old narrower key would reject
        concurrent per-model rows instead of upserting them.

        Granularity is one row per ``(session_id, phase, attempt, model,
        layer)``, appended in batches: never one write per token, never one
        write per bus event. Model and layer belong to the row identity because
        one phase can stream from several producers at once (concurrent QA
        evaluators) and their verbatim text must never interleave. The
        ``segments`` column indexes the label boundaries (subphase, iteration,
        content_status, is_thinking) inside the verbatim ``content`` so a
        reader can tell thinking from draft from accepted text without markers
        being injected into the text itself.

        There is intentionally no foreign key to ``sessions``: a trace must
        never fail to persist because its session row has not been written yet.
        Deletion is handled explicitly by ``purge_session`` and
        ``cleanup_old_sessions``.
        """
        if self._pool is None:
            return

        await self._pool.executescript(
            self._CONTENT_TRACES_TABLE_DDL.format(table="content_traces")
            + ";\n"
            + self._CONTENT_TRACES_INDEX_DDL
        )
        if not await self._content_traces_matches_contract():
            await self._rebuild_content_traces_table()
        await self._pool.commit()

    async def _content_traces_matches_contract(self) -> bool:
        """True when content_traces has the contract columns and unique key."""
        async with self._pool.execute("PRAGMA table_info(content_traces)") as cursor:
            columns = await cursor.fetchall()
        if {col[1] for col in columns} != set(self._CONTENT_TRACES_COLUMNS):
            return False

        async with self._pool.execute("PRAGMA index_list(content_traces)") as cursor:
            indexes = await cursor.fetchall()
        for index in indexes:
            # index_list rows are (seq, name, unique, origin, partial);
            # origin "u" marks indexes created by a UNIQUE constraint.
            if index[3] != "u":
                continue
            async with self._pool.execute(
                f'PRAGMA index_info("{index[1]}")'
            ) as cursor:
                key_columns = await cursor.fetchall()
            if tuple(col[2] for col in key_columns) == self._CONTENT_TRACES_UNIQUE_KEY:
                return True
        return False

    async def _rebuild_content_traces_table(self) -> None:
        """Rebuild content_traces to the current contract, copying every row.

        The trace is verbatim history: it is never scrubbed or discarded, so
        the old table's rows are copied into the new shape (columns the old
        shape lacked take their defaults, e.g. empty model/layer). Runs as one
        transaction: either the rebuilt table with all rows replaces the old
        one, or nothing changes and initialization fails loudly.
        """
        async with self._pool.execute("PRAGMA table_info(content_traces)") as cursor:
            old_columns = [col[1] for col in await cursor.fetchall()]
        contract = set(self._CONTENT_TRACES_COLUMNS)
        shared_csv = ", ".join(col for col in old_columns if col in contract)
        copy_sql = ""
        if shared_csv:
            copy_sql = (
                f"INSERT INTO content_traces_rebuild ({shared_csv})\n"
                f"    SELECT {shared_csv} FROM content_traces;\n"
            )
        logger.warning(
            "Rebuilding debugger content_traces to the current row identity "
            "(session_id, phase, attempt, model, layer); copying existing rows verbatim"
        )
        await self._pool.executescript(
            "BEGIN IMMEDIATE;\n"
            "DROP TABLE IF EXISTS content_traces_rebuild;\n"
            + self._CONTENT_TRACES_TABLE_DDL.format(table="content_traces_rebuild")
            + ";\n"
            + copy_sql
            + "DROP TABLE content_traces;\n"
            "ALTER TABLE content_traces_rebuild RENAME TO content_traces;\n"
            + self._CONTENT_TRACES_INDEX_DDL
            + "\nCOMMIT;"
        )

    def _serialize(self, payload: Any) -> str:
        """Serialize a telemetry payload to JSON verbatim (C13.2)."""
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
            result = await self.cleanup_old_sessions()
            deleted = result.get("deleted_sessions", 0)
            if deleted > 0:
                logger.info("Piggyback cleanup: removed %d old debug sessions", deleted)
        except Exception:
            logger.exception("Piggyback cleanup failed")
        finally:
            self._cleanup_running = False

    def _spawn_cleanup_task(self) -> None:
        """Launch the piggyback cleanup as a strongly-referenced background task.

        With no retention configured nothing can ever be deleted, so the task is
        not created at all (C13.1) instead of being created to no-op.
        """
        if self.retention_days is None:
            return
        task = asyncio.create_task(self._maybe_piggyback_cleanup())
        self._bg_tasks.add(task)
        task.add_done_callback(self._on_cleanup_task_done)

    def _on_cleanup_task_done(self, task: "asyncio.Task[None]") -> None:
        """Release the task reference, re-arm the cleanup guard and observe failures."""
        self._bg_tasks.discard(task)
        self._cleanup_running = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Piggyback cleanup task failed: %s", exc)

    async def record_session_start(
        self,
        session_id: str,
        *,
        request_payload: Any,
        preflight_payload: Optional[Any] = None,
        status: str = "initializing",
        project_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Persist initial session data verbatim, prompts and sources included."""
        async def _operation() -> None:
            logger.debug(
                "Recording session %s... with project_id: %s",
                session_id[:8],
                project_id if project_id else "NULL",
            )

            # Extract request_name for denormalized column
            req_name = None
            if isinstance(request_payload, dict):
                req_name = request_payload.get("request_name")
            elif hasattr(request_payload, "request_name"):
                req_name = request_payload.request_name

            await self._pool.execute(
                """
                INSERT INTO sessions
                (session_id, created_at, updated_at, status, phase, request_json, request_name, preflight_json, project_id)
                VALUES
                (?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = datetime('now'),
                    status = excluded.status,
                    phase = COALESCE(excluded.phase, sessions.phase),
                    request_json = excluded.request_json,
                    request_name = excluded.request_name,
                    preflight_json = excluded.preflight_json,
                    project_id = COALESCE(excluded.project_id, sessions.project_id)
                """,
                (
                    session_id,
                    status,
                    phase,
                    self._serialize(request_payload),
                    req_name,
                    self._serialize(preflight_payload) if preflight_payload is not None else None,
                    project_id,
                ),
            )
            await self._pool.commit()
            self._spawn_cleanup_task()

        await self._run_guarded(_operation, default=None)

    async def update_session_status(
        self,
        session_id: str,
        *,
        status: Optional[str] = None,
        final_payload: Optional[Any] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Update stored session status, phase and optional final payload."""
        async def _operation() -> None:
            where_clause = "WHERE session_id = ?"
            params: List[Any] = [
                status,
                phase,
                self._serialize(final_payload) if final_payload is not None else None,
                session_id,
            ]
            if status is not None and status not in TERMINAL_DEBUGGER_STATUSES:
                placeholders = ", ".join("?" for _ in TERMINAL_DEBUGGER_STATUSES)
                where_clause += f" AND (status IS NULL OR status NOT IN ({placeholders}))"
                params.extend(sorted(TERMINAL_DEBUGGER_STATUSES))
            elif status is not None:
                placeholders = ", ".join("?" for _ in TERMINAL_DEBUGGER_STATUSES)
                where_clause += f" AND (status IS NULL OR status NOT IN ({placeholders}) OR status = ?)"
                params.extend(sorted(TERMINAL_DEBUGGER_STATUSES))
                params.append(status)

            await self._pool.execute(
                f"""
                UPDATE sessions
                SET
                    status = COALESCE(?, status),
                    phase = COALESCE(?, phase),
                    final_json = COALESCE(?, final_json),
                    updated_at = datetime('now')
                {where_clause}
                """,
                tuple(params),
            )
            await self._pool.commit()

        await self._run_guarded(_operation, default=None)

    async def record_usage_summary(self, session_id: str, usage_payload: Any) -> None:
        """Store aggregated usage metrics for a session."""
        async def _operation() -> None:
            await self._pool.execute(
                """
                UPDATE sessions
                SET usage_json = ?, updated_at = datetime('now')
                WHERE session_id = ?
                """,
                (self._serialize(usage_payload), session_id),
            )
            await self._pool.commit()

        await self._run_guarded(_operation, default=None)

    async def record_event(
        self,
        session_id: str,
        *,
        event_type: str,
        payload: Any,
    ) -> None:
        """Insert chronological event for the session."""
        async def _operation() -> None:
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
            await self._pool.execute(
                """
                UPDATE sessions
                SET updated_at = datetime('now')
                WHERE session_id = ?
                """,
                (session_id,),
            )
            await self._pool.commit()
            self._spawn_cleanup_task()

        await self._run_guarded(_operation, default=None)

    async def record_content_trace(
        self,
        *,
        session_id: str,
        phase: str,
        attempt: int,
        content_delta: str,
        expected_offset: int,
        model: str = "",
        layer: str = "",
        project_id: Optional[str] = None,
        subphase: Optional[str] = None,
        iteration: Optional[int] = None,
        content_status: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """Append a batch of verbatim content to the trace of one attempt.

        Called only by ``content_stream.ContentStreamCoalescer`` (single
        writer, at least one second between writes per row). ``content_delta``
        is appended to the existing text so a long stream never rewrites the
        whole blob, and the label columns keep the last observed state of the
        attempt.

        The append is a RESUMABLE, idempotent write positioned at
        ``expected_offset``, the length the caller believes the row already has:
        the row only grows to ``expected_offset + len(content_delta)``, and only
        the part of the delta the row does not already hold is concatenated.
        Retries are therefore free of duplication, and a retry is unavoidable:
        the aiosqlite worker thread executes a queued statement even when the
        future awaiting it was already cancelled (``aiosqlite/core.py``:
        ``_connection_worker_thread`` never inspects the future), so a
        cancellation between this statement and its COMMIT can leave the delta
        applied — committed later by the next writer on the shared connection —
        while the caller still holds it as unwritten. The retry may also arrive
        BATCHED with newer text, which is why the overlap is skipped by length
        instead of the whole append being refused.

        Returns the row's ``content_chars`` AFTER the operation, so the caller
        can resynchronize its own accounting instead of adding lengths blindly,
        or ``None`` when persistence is unavailable.
        """
        if not content_delta:
            return None

        expected_total = expected_offset + len(content_delta)

        async def _operation() -> Optional[int]:
            await self._pool.execute(
                """
                INSERT INTO content_traces (
                    session_id, phase, attempt, model, layer, project_id,
                    subphase, iteration, content_status, content, content_chars,
                    segments, created_at, updated_at
                )
                VALUES (
                    :session_id, :phase, :attempt, :model, :layer, :project_id,
                    :subphase, :iteration, :content_status, :content,
                    :content_chars, :segments, datetime('now'), datetime('now')
                )
                ON CONFLICT(session_id, phase, attempt, model, layer) DO UPDATE SET
                    content = content_traces.content || substr(
                        excluded.content,
                        content_traces.content_chars - :expected_offset + 1
                    ),
                    content_chars = :expected_offset + excluded.content_chars,
                    project_id = COALESCE(excluded.project_id, content_traces.project_id),
                    subphase = excluded.subphase,
                    iteration = COALESCE(excluded.iteration, content_traces.iteration),
                    content_status = excluded.content_status,
                    segments = excluded.segments,
                    updated_at = datetime('now')
                WHERE content_traces.content_chars >= :expected_offset
                  AND content_traces.content_chars
                      < :expected_offset + excluded.content_chars
                """,
                {
                    "session_id": session_id,
                    "phase": phase,
                    "attempt": attempt,
                    "model": model,
                    "layer": layer,
                    "project_id": project_id,
                    "subphase": subphase,
                    "iteration": iteration,
                    "content_status": content_status,
                    "content": content_delta,
                    "content_chars": len(content_delta),
                    "segments": (
                        json.dumps(segments, ensure_ascii=True) if segments else None
                    ),
                    "expected_offset": expected_offset,
                },
            )
            async with self._pool.execute(
                """
                SELECT content_chars
                FROM content_traces
                WHERE session_id = ?
                  AND phase = ?
                  AND attempt = ?
                  AND model = ?
                  AND layer = ?
                """,
                (session_id, phase, attempt, model, layer),
            ) as cursor:
                row = await cursor.fetchone()
            await self._pool.commit()

            if row is None:
                return None
            stored_chars = int(row[0])
            if stored_chars < expected_total:
                # The row shrank or was replaced under the single writer, which
                # only a purge or a retention sweep does. The append was refused
                # (or landed on a recreated row), so this delta is what the trace
                # loses; say it instead of leaving a silent hole.
                logger.warning(
                    "Content trace row for session %s phase %s attempt %s is at "
                    "%d chars instead of the expected %d: the row changed under "
                    "the writer and this %d-char delta may be missing",
                    session_id,
                    phase,
                    attempt,
                    stored_chars,
                    expected_total,
                    len(content_delta),
                )
            return stored_chars

        return await self._run_guarded(_operation, default=None)

    async def get_content_traces(
        self,
        session_id: str,
        *,
        phase: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return the stored draft traces of a session, oldest attempt first.

        ``limit``/``offset`` page the result like the session and event readers
        do. The ORDER BY is total over the row identity, so a page stays stable
        across calls even while a run keeps appending to an existing row.
        """
        async def _operation() -> List[Dict[str, Any]]:
            where_clause = "WHERE session_id = ?"
            params: List[Any] = [session_id]
            if phase:
                where_clause += " AND phase = ?"
                params.append(phase)

            # SQLite has no bare OFFSET. A limit of -1 means "no ceiling", which
            # is what an offset without a limit asks for.
            page_clause = ""
            if limit is not None or offset:
                page_clause = "LIMIT ? OFFSET ?"
                params.extend([-1 if limit is None else limit, offset])

            async with self._pool.execute(
                f"""
                SELECT session_id, phase, attempt, model, layer, project_id,
                       subphase, iteration, content_status, content,
                       content_chars, segments, created_at, updated_at
                FROM content_traces
                {where_clause}
                ORDER BY phase ASC, attempt ASC, model ASC, layer ASC
                {page_clause}
                """,
                params,
            ) as cursor:
                rows = await cursor.fetchall()

            traces: List[Dict[str, Any]] = []
            for row in rows:
                traces.append(
                    {
                        "session_id": row[0],
                        "phase": row[1],
                        "attempt": row[2],
                        "model": row[3],
                        "layer": row[4],
                        "project_id": row[5],
                        "subphase": row[6],
                        "iteration": row[7],
                        "content_status": row[8],
                        "content": row[9],
                        "content_chars": row[10],
                        "segments": _decode_trace_segments(row[11]),
                        "created_at": row[12],
                        "updated_at": row[13],
                    }
                )
            return traces

        return await self._run_guarded(_operation, default=[])

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
        async def _operation() -> List[Dict[str, Any]]:
            where_clause = ""
            params: List[Any] = []
            if project_id:
                where_clause = "WHERE project_id = ?"
                params.append(project_id)

            query = f"""
                SELECT session_id, created_at, updated_at, status, phase, project_id, request_name
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
                    "phase": row[4],
                    "project_id": row[5],
                    "request_name": row[6],
                }
                for row in rows
            ]

        return await self._run_guarded(_operation, default=[])

    async def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a session."""
        async def _operation() -> Optional[Dict[str, Any]]:
            async with self._pool.execute(
                """
                SELECT session_id, created_at, updated_at, status, phase,
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
                "phase": row[4],
                "request_json": row[5],
                "preflight_json": row[6],
                "final_json": row[7],
                "usage_json": row[8],
                "notes": row[9],
                "project_id": row[10],
            }

        return await self._run_guarded(_operation, default=None)

    async def get_session_events(
        self,
        session_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch chronological events for display."""
        async def _operation() -> List[Dict[str, Any]]:
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

        return await self._run_guarded(_operation, default=[])

    async def purge_session(self, session_id: str) -> None:
        """Delete all data for a specific session."""
        async def _operation() -> None:
            await self._pool.execute(
                "DELETE FROM content_traces WHERE session_id = ?",
                (session_id,),
            )
            await self._pool.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await self._pool.commit()
            async with self._event_lock:
                self._event_counters.pop(session_id, None)

        await self._run_guarded(_operation, default=None)

    async def cleanup_old_sessions(
        self,
        retention_days: Optional[int] = None,
    ) -> dict:
        """
        Delete inactive debug telemetry older than ``retention_days``.

        Status is intentionally irrelevant: non-terminal rows are observations
        of a past runtime, not durable recovery or cancellation authority.

        Args:
            retention_days: Explicit inactive-day window. When omitted, the
                instance setting from ``DEBUGGER_RETENTION_DAYS`` is used; when
                that is unset or 0, nothing is ever deleted (C13.1).

        Returns:
            Dictionary with cleanup statistics
        """
        if not self.storage_available or self._pool is None:
            return {"status": "disabled", "deleted_sessions": 0, "deleted_events": 0}

        window = (
            self.retention_days
            if retention_days is None
            else _normalize_retention_days(retention_days)
        )
        if window is None:
            return {
                "status": "retention_disabled",
                "deleted_sessions": 0,
                "deleted_events": 0,
            }
        retention_days = window

        try:
            # Compute cutoff once in Python (ISO-8601 sorts correctly as text)
            cutoff_str = (datetime.utcnow() - timedelta(days=retention_days)).strftime('%Y-%m-%d %H:%M:%S')
            cleanup_params = (cutoff_str,)

            # Count before deletion for stats
            async with self._pool.execute(
                """
                SELECT COUNT(*)
                FROM sessions
                WHERE updated_at < ?
                """,
                cleanup_params,
            ) as cursor:
                row = await cursor.fetchone()
                sessions_to_delete = row[0] if row else 0

            async with self._pool.execute(
                """SELECT COUNT(*) FROM session_events
                   WHERE session_id IN (
                        SELECT session_id FROM sessions
                        WHERE updated_at < ?
                   )""",
                cleanup_params,
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
                        WHERE updated_at < ?
                   )""",
                cleanup_params,
            )

            # Content traces have no foreign key on purpose; drop them with
            # the session they belong to so retention never leaks orphan rows.
            await self._pool.execute(
                """DELETE FROM content_traces
                   WHERE session_id IN (
                        SELECT session_id FROM sessions
                        WHERE updated_at < ?
                   )""",
                cleanup_params,
            )

            # Delete old telemetry irrespective of its last observed status.
            await self._pool.execute(
                """
                DELETE FROM sessions
                WHERE updated_at < ?
                """,
                cleanup_params,
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
            if is_terminal_debugger_storage_error(e):
                await self._disable_storage(e)
                return {
                    "status": "disabled",
                    "error": str(e),
                    "deleted_sessions": 0,
                    "deleted_events": 0
                }
            logger.error("Failed to cleanup old sessions: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "deleted_sessions": 0,
                "deleted_events": 0
            }


_shared_debug_logger: Optional[DebugLogger] = None
_debug_logger_lock = asyncio.Lock()


async def initialize_debug_logger(
    enabled: bool,
    db_path: str,
    retention_days: Optional[int] = None,
) -> None:
    """Initialize shared debug logger instance."""
    logger_instance = await get_debug_logger(
        enabled=enabled,
        db_path=db_path,
        retention_days=retention_days,
    )
    await logger_instance.initialize()


async def shutdown_debug_logger() -> None:
    """Close shared debug logger."""
    global _shared_debug_logger
    if _shared_debug_logger is not None:
        await _shared_debug_logger.close()
        _shared_debug_logger = None


async def get_debug_logger(
    *,
    enabled: bool,
    db_path: str,
    retention_days: Optional[int] = None,
) -> DebugLogger:
    """Return shared debug logger (creating if needed)."""
    global _shared_debug_logger

    if _shared_debug_logger is not None:
        return _shared_debug_logger

    async with _debug_logger_lock:
        if _shared_debug_logger is None:
            _shared_debug_logger = DebugLogger(
                enabled=enabled,
                db_path=db_path,
                retention_days=retention_days,
            )
    return _shared_debug_logger
