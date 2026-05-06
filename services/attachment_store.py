"""SQLite-backed deduplicated attachment blob storage."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from config import AttachmentSettings


class AttachmentStoreError(Exception):
    """Base exception for attachment store failures."""


class AttachmentStoreConflictError(AttachmentStoreError):
    """Raised when an upload row already exists or is tombstoned."""


class AttachmentStoreIntegrityError(AttachmentStoreError):
    """Raised when an existing dedupe blob is not safe to reuse."""


@dataclass(frozen=True)
class AttachmentBlobRow:
    """Single row from attachment_blobs."""

    id: int
    sha256: str
    size_bytes: int
    kind: str
    mime_type: str
    storage_key: str
    status: str
    quarantine_reason: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AttachmentUploadRow:
    """Joined upload/blob row used by AttachmentManager."""

    id: int
    upload_id: str
    blob: AttachmentBlobRow
    user_hash: str
    hash_prefix1: str
    hash_prefix2: str
    origin: str
    intended_usage: str
    original_filename: str
    stored_filename: str
    mime_type: str
    declared_size: Optional[int]
    declared_mime: Optional[str]
    detected_mime: Optional[str]
    original_url: Optional[str]
    metadata_signature: str
    legacy_storage_path: Optional[str]
    legacy_metadata_path: Optional[str]
    migrated_from_legacy: bool
    status: str
    created_at: str
    updated_at: str


class AttachmentStore:
    """Small SQLite facade for content-addressed attachment blobs."""

    VALID_KINDS = frozenset({"text", "json", "pdf", "image"})
    ACTIVE_BLOB_STATUSES = ("pending", "ready", "gc_pending")
    CHUNK_SIZE = 1024 * 1024

    def __init__(self, *, settings: AttachmentSettings) -> None:
        self.settings = settings
        self.db_path = Path(settings.dedupe_db_path)
        self.blob_base_path = Path(settings.blob_base_path)

    def ensure_schema(self) -> None:
        """Create the database schema if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_base_path.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS attachment_blobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    kind TEXT NOT NULL CHECK (kind IN ('text', 'json', 'pdf', 'image')),
                    mime_type TEXT NOT NULL,
                    storage_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'gc_pending', 'quarantined')),
                    quarantine_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_attachment_blob_dedupe_active
                    ON attachment_blobs (sha256, size_bytes, kind)
                    WHERE status IN ('pending', 'ready', 'gc_pending');

                CREATE TABLE IF NOT EXISTS attachment_uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    upload_id TEXT NOT NULL,
                    blob_id INTEGER NOT NULL REFERENCES attachment_blobs(id) ON DELETE RESTRICT,
                    user_hash TEXT NOT NULL,
                    hash_prefix1 TEXT NOT NULL,
                    hash_prefix2 TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    intended_usage TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    declared_size INTEGER,
                    declared_mime TEXT,
                    detected_mime TEXT,
                    original_url TEXT,
                    metadata_signature TEXT NOT NULL,
                    legacy_storage_path TEXT,
                    legacy_metadata_path TEXT,
                    migrated_from_legacy INTEGER NOT NULL DEFAULT 0 CHECK (migrated_from_legacy IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'quarantined')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_hash, upload_id)
                );

                CREATE INDEX IF NOT EXISTS ix_attachment_uploads_blob_id
                    ON attachment_uploads (blob_id);

                CREATE TABLE IF NOT EXISTS attachment_upload_deletions (
                    user_hash TEXT NOT NULL,
                    upload_id TEXT NOT NULL,
                    deleted_at TEXT NOT NULL,
                    reason TEXT,
                    PRIMARY KEY (user_hash, upload_id)
                );

                CREATE TABLE IF NOT EXISTS attachment_migration_issues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    legacy_metadata_path TEXT,
                    legacy_storage_path TEXT,
                    upload_id TEXT,
                    user_hash TEXT,
                    issue_code TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("PRAGMA user_version = 1")

    def backup_to(self, output_path: Path) -> None:
        """Create a consistent SQLite backup."""
        self.ensure_schema()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(self.db_path)
        try:
            target = sqlite3.connect(output_path)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()

    def build_storage_key(
        self,
        *,
        kind: str,
        sha256: str,
        stored_filename: str,
        mime_type: str,
    ) -> str:
        """Return the canonical content-addressed storage key for a blob."""
        normalized_kind = self._normalize_kind(kind)
        digest = self._validate_sha256(sha256)
        suffix = Path(stored_filename).suffix.lower()
        if not suffix:
            suffix = mimetypes.guess_extension(mime_type) or ".bin"
        suffix = self._safe_suffix(suffix)
        return f"{normalized_kind}/sha256/{digest[:2]}/{digest[2:4]}/{digest}{suffix}"

    def blob_path(self, storage_key: str) -> Path:
        """Resolve a storage key to an absolute path under blob_base_path."""
        relative = Path(storage_key)
        if relative.is_absolute() or ".." in relative.parts:
            raise AttachmentStoreIntegrityError("Blob storage key is outside blob base path")
        path = self.blob_base_path / relative
        try:
            path.relative_to(self.blob_base_path)
        except ValueError as exc:
            raise AttachmentStoreIntegrityError("Blob path escapes blob base path") from exc
        return path

    def create_upload_from_temp(
        self,
        *,
        upload_id: str,
        user_hash: str,
        hash_prefix1: str,
        hash_prefix2: str,
        origin: str,
        intended_usage: str,
        original_filename: str,
        stored_filename: str,
        mime_type: str,
        declared_size: Optional[int],
        declared_mime: Optional[str],
        detected_mime: Optional[str],
        original_url: Optional[str],
        sha256: str,
        size_bytes: int,
        kind: str,
        temp_path: Path,
        metadata_signature: str,
        created_at: str,
        legacy_storage_path: Optional[str] = None,
        legacy_metadata_path: Optional[str] = None,
        migrated_from_legacy: bool = False,
    ) -> AttachmentUploadRow:
        """Persist or reuse a blob and create an upload row for it."""
        self.ensure_schema()
        normalized_kind = self._normalize_kind(kind)
        digest = self._validate_sha256(sha256)
        if not temp_path.exists():
            raise AttachmentStoreIntegrityError("Temporary attachment payload is missing")
        temp_hash, temp_size = self.compute_file_digest(temp_path)
        if temp_hash != digest or temp_size != size_bytes:
            temp_path.unlink(missing_ok=True)
            raise AttachmentStoreIntegrityError("Temporary attachment checksum or size mismatch")
        storage_key = self.build_storage_key(
            kind=normalized_kind,
            sha256=digest,
            stored_filename=stored_filename,
            mime_type=mime_type,
        )
        now = self._utc_now()
        final_path = self.blob_path(storage_key)
        moved_to_final = False
        committed = False

        try:
            with self._connection(immediate=True) as conn:
                if self.deletion_exists(user_hash=user_hash, upload_id=upload_id, conn=conn):
                    raise AttachmentStoreConflictError("Attachment upload_id is tombstoned")

                existing_upload = self._get_upload(
                    user_hash=user_hash,
                    upload_id=upload_id,
                    conn=conn,
                )
                if existing_upload is not None:
                    raise AttachmentStoreConflictError("Attachment upload_id already exists")

                blob = self._find_reusable_blob(
                    conn=conn,
                    sha256=digest,
                    size_bytes=size_bytes,
                    kind=normalized_kind,
                )
                if blob is None:
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    final_preexisted = final_path.exists()
                    blob_id = self._insert_pending_blob(
                        conn=conn,
                        sha256=digest,
                        size_bytes=size_bytes,
                        kind=normalized_kind,
                        mime_type=mime_type,
                        storage_key=storage_key,
                        created_at=created_at,
                        updated_at=now,
                    )
                    if final_preexisted:
                        existing_hash, existing_size = self.compute_file_digest(final_path)
                        if existing_hash != digest or existing_size != size_bytes:
                            self._quarantine_blob(
                                conn=conn,
                                blob_id=blob_id,
                                reason="Canonical blob path existed with different content",
                                now=now,
                            )
                            raise AttachmentStoreIntegrityError(
                                "Canonical blob path exists with different content"
                            )
                        temp_path.unlink(missing_ok=True)
                    else:
                        os.replace(temp_path, final_path)
                        moved_to_final = True

                    final_hash, final_size = self.compute_file_digest(final_path)
                    if final_hash != digest or final_size != size_bytes:
                        self._quarantine_blob(
                            conn=conn,
                            blob_id=blob_id,
                            reason="Final blob checksum or size mismatch",
                            now=now,
                        )
                        raise AttachmentStoreIntegrityError("Final blob verification failed")

                    conn.execute(
                        """
                        UPDATE attachment_blobs
                        SET status = 'ready', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, blob_id),
                    )
                    blob = self._get_blob_by_id(conn=conn, blob_id=blob_id)
                else:
                    temp_path.unlink(missing_ok=True)

                if blob is None:
                    raise AttachmentStoreIntegrityError("Unable to resolve attachment blob")

                conn.execute(
                    """
                    INSERT INTO attachment_uploads (
                        upload_id, blob_id, user_hash, hash_prefix1, hash_prefix2,
                        origin, intended_usage, original_filename, stored_filename, mime_type,
                        declared_size, declared_mime, detected_mime, original_url,
                        metadata_signature, legacy_storage_path, legacy_metadata_path,
                        migrated_from_legacy, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        upload_id,
                        blob.id,
                        user_hash,
                        hash_prefix1,
                        hash_prefix2,
                        origin,
                        intended_usage,
                        original_filename,
                        stored_filename,
                        mime_type,
                        declared_size,
                        declared_mime,
                        detected_mime,
                        original_url,
                        metadata_signature,
                        legacy_storage_path,
                        legacy_metadata_path,
                        1 if migrated_from_legacy else 0,
                        created_at,
                        now,
                    ),
                )
                committed = True
                return self._get_upload(
                    user_hash=user_hash,
                    upload_id=upload_id,
                    conn=conn,
                )  # type: ignore[return-value]
        except sqlite3.IntegrityError as exc:
            raise AttachmentStoreConflictError(str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
            if moved_to_final and not committed:
                final_path.unlink(missing_ok=True)

    def get_upload(self, *, user_hash: str, upload_id: str) -> Optional[AttachmentUploadRow]:
        """Fetch one upload row with its blob, regardless of status."""
        self.ensure_schema()
        with self._connection() as conn:
            return self._get_upload(user_hash=user_hash, upload_id=upload_id, conn=conn)

    def deletion_exists(
        self,
        *,
        user_hash: str,
        upload_id: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """Return true when a tombstone exists for the upload."""
        if conn is None:
            self.ensure_schema()
            with self._connection() as owned_conn:
                return self.deletion_exists(user_hash=user_hash, upload_id=upload_id, conn=owned_conn)
        row = conn.execute(
            """
            SELECT 1
            FROM attachment_upload_deletions
            WHERE user_hash = ? AND upload_id = ?
            """,
            (user_hash, upload_id),
        ).fetchone()
        return row is not None

    def delete_upload(self, *, user_hash: str, upload_id: str, reason: str) -> bool:
        """Delete an upload row and insert a tombstone."""
        self.ensure_schema()
        with self._connection(immediate=True) as conn:
            upload = self._get_upload(user_hash=user_hash, upload_id=upload_id, conn=conn)
            if upload is None:
                return False
            self._record_deletion_tombstone(
                conn=conn,
                user_hash=user_hash,
                upload_id=upload_id,
                reason=reason,
            )
            conn.execute(
                "DELETE FROM attachment_uploads WHERE user_hash = ? AND upload_id = ?",
                (user_hash, upload_id),
            )
            return True

    def record_deletion_tombstone(self, *, user_hash: str, upload_id: str, reason: str) -> None:
        """Insert a deletion tombstone without requiring an active upload row."""
        self.ensure_schema()
        with self._connection(immediate=True) as conn:
            self._record_deletion_tombstone(
                conn=conn,
                user_hash=user_hash,
                upload_id=upload_id,
                reason=reason,
            )

    def record_migration_issue(
        self,
        *,
        issue_code: str,
        detail: str,
        legacy_metadata_path: Optional[str] = None,
        legacy_storage_path: Optional[str] = None,
        upload_id: Optional[str] = None,
        user_hash: Optional[str] = None,
    ) -> None:
        """Persist a migration issue for later audit."""
        self.ensure_schema()
        with self._connection(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO attachment_migration_issues (
                    legacy_metadata_path, legacy_storage_path, upload_id, user_hash,
                    issue_code, detail, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    legacy_metadata_path,
                    legacy_storage_path,
                    upload_id,
                    user_hash,
                    issue_code,
                    detail,
                    self._utc_now(),
                ),
            )

    def iter_uploads(self) -> Iterator[AttachmentUploadRow]:
        """Yield all upload rows with blob metadata."""
        self.ensure_schema()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.id AS upload_row_id,
                    u.upload_id, u.user_hash, u.hash_prefix1, u.hash_prefix2,
                    u.origin, u.intended_usage, u.original_filename, u.stored_filename,
                    u.mime_type AS upload_mime_type,
                    u.declared_size, u.declared_mime, u.detected_mime, u.original_url,
                    u.metadata_signature, u.legacy_storage_path, u.legacy_metadata_path,
                    u.migrated_from_legacy, u.status AS upload_status,
                    u.created_at AS upload_created_at, u.updated_at AS upload_updated_at,
                    b.id AS blob_id, b.sha256, b.size_bytes, b.kind,
                    b.mime_type AS blob_mime_type, b.storage_key, b.status AS blob_status,
                    b.quarantine_reason, b.created_at AS blob_created_at,
                    b.updated_at AS blob_updated_at
                FROM attachment_uploads u
                JOIN attachment_blobs b ON b.id = u.blob_id
                ORDER BY u.created_at ASC, u.id ASC
                """
            )
            for row in rows:
                yield self._row_to_upload(row)

    def iter_deletions(self) -> Iterator[Dict[str, Optional[str]]]:
        """Yield deletion tombstones."""
        self.ensure_schema()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT user_hash, upload_id, deleted_at, reason
                FROM attachment_upload_deletions
                ORDER BY deleted_at ASC
                """
            )
            for row in rows:
                yield {
                    "user_hash": row["user_hash"],
                    "upload_id": row["upload_id"],
                    "deleted_at": row["deleted_at"],
                    "reason": row["reason"],
                }

    def count_ready_blobs(self) -> int:
        """Return number of ready blobs. Used by tests and diagnostics."""
        self.ensure_schema()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM attachment_blobs WHERE status = 'ready'"
            ).fetchone()
            return int(row["count"])

    def gc_unreferenced_blobs(self, *, dry_run: bool = True, limit: int = 100) -> Dict[str, int]:
        """Remove ready blobs that have no uploads, using gc_pending claims."""
        self.ensure_schema()
        scanned = 0
        removed = 0
        quarantined = 0
        now = self._utc_now()
        with self._connection(immediate=not dry_run) as conn:
            rows = conn.execute(
                """
                SELECT b.*
                FROM attachment_blobs b
                LEFT JOIN attachment_uploads u ON u.blob_id = b.id
                WHERE b.status = 'ready' AND u.id IS NULL
                ORDER BY b.created_at ASC
                LIMIT ?
                """,
                (max(limit, 1),),
            ).fetchall()
            for row in rows:
                scanned += 1
                blob = self._row_to_blob(row)
                if dry_run:
                    continue
                updated = conn.execute(
                    """
                    UPDATE attachment_blobs
                    SET status = 'gc_pending', updated_at = ?
                    WHERE id = ?
                      AND status = 'ready'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM attachment_uploads
                          WHERE blob_id = attachment_blobs.id
                      )
                    """,
                    (now, blob.id),
                ).rowcount
                if updated != 1:
                    continue
                path = self.blob_path(blob.storage_key)
                try:
                    path.unlink(missing_ok=True)
                    deleted = conn.execute(
                        """
                        DELETE FROM attachment_blobs
                        WHERE id = ?
                          AND status = 'gc_pending'
                          AND NOT EXISTS (
                              SELECT 1
                              FROM attachment_uploads
                              WHERE blob_id = attachment_blobs.id
                          )
                        """,
                        (blob.id,),
                    ).rowcount
                    if deleted != 1:
                        conn.execute(
                            """
                            UPDATE attachment_blobs
                            SET status = 'quarantined',
                                quarantine_reason = ?,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            ("GC delete lost unreferenced claim after unlink", now, blob.id),
                        )
                        quarantined += 1
                        continue
                    removed += 1
                except OSError as exc:
                    conn.execute(
                        """
                        UPDATE attachment_blobs
                        SET status = 'quarantined',
                            quarantine_reason = ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (f"GC failed: {exc}", now, blob.id),
                    )
                    quarantined += 1
        return {"scanned": scanned, "removed": removed, "quarantined": quarantined}

    @classmethod
    def compute_file_digest(cls, path: Path) -> Tuple[str, int]:
        """Compute SHA-256 and size for a file."""
        hasher = hashlib.sha256()
        size = 0
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(cls.CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                hasher.update(chunk)
        return hasher.hexdigest(), size

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def _connection(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _find_reusable_blob(
        self,
        *,
        conn: sqlite3.Connection,
        sha256: str,
        size_bytes: int,
        kind: str,
    ) -> Optional[AttachmentBlobRow]:
        """Find a ready reusable blob, quarantining unsafe rows on the way."""
        while True:
            row = conn.execute(
                """
                SELECT *
                FROM attachment_blobs
                WHERE sha256 = ?
                  AND size_bytes = ?
                  AND kind = ?
                  AND status IN ('pending', 'ready', 'gc_pending')
                ORDER BY id ASC
                LIMIT 1
                """,
                (sha256, size_bytes, kind),
            ).fetchone()
            if row is None:
                return None
            blob = self._row_to_blob(row)
            if blob.status == "ready":
                path = self.blob_path(blob.storage_key)
                if not path.exists():
                    self._quarantine_blob(
                        conn=conn,
                        blob_id=blob.id,
                        reason="Ready blob file is missing",
                        now=self._utc_now(),
                    )
                    continue
                existing_hash, existing_size = self.compute_file_digest(path)
                if existing_hash != sha256 or existing_size != size_bytes:
                    self._quarantine_blob(
                        conn=conn,
                        blob_id=blob.id,
                        reason="Ready blob file checksum or size mismatch",
                        now=self._utc_now(),
                    )
                    continue
                return blob
            raise AttachmentStoreIntegrityError(
                f"Blob {blob.id} is currently {blob.status}; retry later"
            )

    def _insert_pending_blob(
        self,
        *,
        conn: sqlite3.Connection,
        sha256: str,
        size_bytes: int,
        kind: str,
        mime_type: str,
        storage_key: str,
        created_at: str,
        updated_at: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO attachment_blobs (
                sha256, size_bytes, kind, mime_type, storage_key,
                status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (sha256, size_bytes, kind, mime_type, storage_key, created_at, updated_at),
        )
        return int(cursor.lastrowid)

    def _quarantine_blob(
        self,
        *,
        conn: sqlite3.Connection,
        blob_id: int,
        reason: str,
        now: str,
    ) -> None:
        blob = self._get_blob_by_id(conn=conn, blob_id=blob_id)
        if blob is None:
            return
        old_key = blob.storage_key
        new_key = self._quarantine_storage_key(blob_id, old_key)
        old_path = self.blob_path(old_key)
        new_path = self.blob_path(new_key)
        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(old_path), str(new_path))
            except OSError as exc:
                raise AttachmentStoreIntegrityError(f"Unable to quarantine blob: {exc}") from exc
        conn.execute(
            """
            UPDATE attachment_blobs
            SET status = 'quarantined',
                quarantine_reason = ?,
                storage_key = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (reason, new_key, now, blob_id),
        )

    def _get_blob_by_id(
        self,
        *,
        conn: sqlite3.Connection,
        blob_id: int,
    ) -> Optional[AttachmentBlobRow]:
        row = conn.execute("SELECT * FROM attachment_blobs WHERE id = ?", (blob_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_blob(row)

    def _get_upload(
        self,
        *,
        user_hash: str,
        upload_id: str,
        conn: sqlite3.Connection,
    ) -> Optional[AttachmentUploadRow]:
        row = conn.execute(
            """
            SELECT
                u.id AS upload_row_id,
                u.upload_id, u.user_hash, u.hash_prefix1, u.hash_prefix2,
                u.origin, u.intended_usage, u.original_filename, u.stored_filename,
                u.mime_type AS upload_mime_type,
                u.declared_size, u.declared_mime, u.detected_mime, u.original_url,
                u.metadata_signature, u.legacy_storage_path, u.legacy_metadata_path,
                u.migrated_from_legacy, u.status AS upload_status,
                u.created_at AS upload_created_at, u.updated_at AS upload_updated_at,
                b.id AS blob_id, b.sha256, b.size_bytes, b.kind,
                b.mime_type AS blob_mime_type, b.storage_key, b.status AS blob_status,
                b.quarantine_reason, b.created_at AS blob_created_at,
                b.updated_at AS blob_updated_at
            FROM attachment_uploads u
            JOIN attachment_blobs b ON b.id = u.blob_id
            WHERE u.user_hash = ? AND u.upload_id = ?
            """,
            (user_hash, upload_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_upload(row)

    def _record_deletion_tombstone(
        self,
        *,
        conn: sqlite3.Connection,
        user_hash: str,
        upload_id: str,
        reason: str,
    ) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO attachment_upload_deletions (
                user_hash, upload_id, deleted_at, reason
            )
            VALUES (?, ?, ?, ?)
            """,
            (user_hash, upload_id, self._utc_now(), reason),
        )

    def _row_to_blob(self, row: sqlite3.Row) -> AttachmentBlobRow:
        return AttachmentBlobRow(
            id=int(row["id"] if "id" in row.keys() else row["blob_id"]),
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            kind=str(row["kind"]),
            mime_type=str(row["mime_type"] if "mime_type" in row.keys() else row["blob_mime_type"]),
            storage_key=str(row["storage_key"]),
            status=str(row["status"] if "status" in row.keys() else row["blob_status"]),
            quarantine_reason=row["quarantine_reason"],
            created_at=str(row["created_at"] if "created_at" in row.keys() else row["blob_created_at"]),
            updated_at=str(row["updated_at"] if "updated_at" in row.keys() else row["blob_updated_at"]),
        )

    def _row_to_upload(self, row: sqlite3.Row) -> AttachmentUploadRow:
        blob = AttachmentBlobRow(
            id=int(row["blob_id"]),
            sha256=str(row["sha256"]),
            size_bytes=int(row["size_bytes"]),
            kind=str(row["kind"]),
            mime_type=str(row["blob_mime_type"]),
            storage_key=str(row["storage_key"]),
            status=str(row["blob_status"]),
            quarantine_reason=row["quarantine_reason"],
            created_at=str(row["blob_created_at"]),
            updated_at=str(row["blob_updated_at"]),
        )
        return AttachmentUploadRow(
            id=int(row["upload_row_id"]),
            upload_id=str(row["upload_id"]),
            blob=blob,
            user_hash=str(row["user_hash"]),
            hash_prefix1=str(row["hash_prefix1"]),
            hash_prefix2=str(row["hash_prefix2"]),
            origin=str(row["origin"]),
            intended_usage=str(row["intended_usage"]),
            original_filename=str(row["original_filename"]),
            stored_filename=str(row["stored_filename"]),
            mime_type=str(row["upload_mime_type"]),
            declared_size=row["declared_size"],
            declared_mime=row["declared_mime"],
            detected_mime=row["detected_mime"],
            original_url=row["original_url"],
            metadata_signature=str(row["metadata_signature"]),
            legacy_storage_path=row["legacy_storage_path"],
            legacy_metadata_path=row["legacy_metadata_path"],
            migrated_from_legacy=bool(row["migrated_from_legacy"]),
            status=str(row["upload_status"]),
            created_at=str(row["upload_created_at"]),
            updated_at=str(row["upload_updated_at"]),
        )

    def _normalize_kind(self, kind: str) -> str:
        normalized = (kind or "").strip().lower()
        if normalized not in self.VALID_KINDS:
            raise AttachmentStoreIntegrityError(f"Unsupported attachment blob kind: {kind}")
        return normalized

    def _validate_sha256(self, sha256: str) -> str:
        digest = (sha256 or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise AttachmentStoreIntegrityError("Invalid SHA-256 digest")
        return digest

    def _safe_suffix(self, suffix: str) -> str:
        cleaned = suffix.lower().strip()
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        allowed = []
        for ch in cleaned[:16]:
            if ch in ".abcdefghijklmnopqrstuvwxyz0123456789":
                allowed.append(ch)
        result = "".join(allowed).strip(".")
        return f".{result}" if result else ".bin"

    def _quarantine_storage_key(self, blob_id: int, storage_key: str) -> str:
        name = Path(storage_key).name or f"blob-{blob_id}"
        return f"quarantine/{blob_id}/{int(time.time())}-{name}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
