"""Migration utilities for deduplicated attachment storage."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json_utils as json  # noqa: E402

from config import config  # noqa: E402
from services.attachment_manager import (  # noqa: E402
    AttachmentManager,
    AttachmentRecord,
    AttachmentValidationError,
)
from services.attachment_store import AttachmentStoreError  # noqa: E402


class MigrationCLIError(Exception):
    """Raised when migration CLI validation fails."""


@dataclass
class MigrationReport:
    """Structured report emitted by migration commands."""

    dry_run: bool
    scanned_metadata: int = 0
    valid_metadata: int = 0
    migrated_uploads: int = 0
    skipped_existing: int = 0
    skipped_tombstoned: int = 0
    missing_binary: int = 0
    invalid_metadata: int = 0
    bytes_seen: int = 0
    bytes_migrated: int = 0
    actions: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    def issue(
        self,
        code: str,
        detail: str,
        *,
        metadata_path: Optional[Path] = None,
        storage_path: Optional[Path] = None,
        upload_id: Optional[str] = None,
        user_hash: Optional[str] = None,
    ) -> None:
        self.issues.append(
            {
                "code": code,
                "detail": detail,
                "metadata_path": str(metadata_path) if metadata_path else None,
                "storage_path": str(storage_path) if storage_path else None,
                "upload_id": upload_id,
                "user_hash": user_hash,
            }
        )

    def action(self, category: str, detail: str, **extra: Any) -> None:
        payload = {"category": category, "detail": detail}
        payload.update(extra)
        self.actions.append(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "scanned_metadata": self.scanned_metadata,
            "valid_metadata": self.valid_metadata,
            "migrated_uploads": self.migrated_uploads,
            "skipped_existing": self.skipped_existing,
            "skipped_tombstoned": self.skipped_tombstoned,
            "missing_binary": self.missing_binary,
            "invalid_metadata": self.invalid_metadata,
            "bytes_seen": self.bytes_seen,
            "bytes_migrated": self.bytes_migrated,
            "actions": self.actions,
            "issues": self.issues,
        }


def _resolve_manager() -> AttachmentManager:
    pepper = config.PEPPER
    if not pepper:
        raise MigrationCLIError("PEPPER environment variable must be configured before migration")
    return AttachmentManager(settings=config.ATTACHMENTS, pepper=pepper)


def _safe_user_child(user_root: Path, relative_path: str) -> Optional[Path]:
    candidate = user_root / Path(relative_path)
    try:
        candidate.resolve().relative_to(user_root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _legacy_metadata_files(manager: AttachmentManager) -> Iterator[Path]:
    for user_root in manager._iter_user_roots():
        uploads_dir = user_root / "uploads"
        if uploads_dir.exists():
            yield from sorted(uploads_dir.rglob("metadata.json"))


def _user_root_for_metadata(manager: AttachmentManager, metadata_path: Path) -> Optional[Tuple[Path, str, str, str]]:
    try:
        relative = metadata_path.relative_to(manager.base_path)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 5:
        return None
    prefix1, prefix2, user_hash = parts[0], parts[1], parts[2]
    return manager._user_root(prefix1, prefix2, user_hash), prefix1, prefix2, user_hash


def _load_legacy_record(
    manager: AttachmentManager,
    metadata_path: Path,
    report: MigrationReport,
) -> Optional[Tuple[AttachmentRecord, Path, Path, str]]:
    report.scanned_metadata += 1
    root_info = _user_root_for_metadata(manager, metadata_path)
    if root_info is None:
        report.invalid_metadata += 1
        report.issue("unexpected_path", "Metadata path is not under a user upload root", metadata_path=metadata_path)
        return None
    user_root, prefix1, prefix2, user_hash = root_info

    try:
        with metadata_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        report.invalid_metadata += 1
        report.issue("metadata_read_failed", str(exc), metadata_path=metadata_path)
        return None

    try:
        record = AttachmentRecord(**payload)
    except Exception as exc:
        report.invalid_metadata += 1
        report.issue("metadata_schema_failed", str(exc), metadata_path=metadata_path)
        return None

    expected_signature = manager._compute_metadata_signature(payload)
    if not record.metadata_signature or record.metadata_signature != expected_signature:
        report.invalid_metadata += 1
        report.issue(
            "metadata_signature_mismatch",
            "Legacy metadata signature does not match payload",
            metadata_path=metadata_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    expected_metadata_rel = manager._relative_path(user_root, metadata_path)
    if record.metadata_path != expected_metadata_rel:
        report.invalid_metadata += 1
        report.issue(
            "metadata_path_mismatch",
            "Record metadata_path does not match the scanned metadata file",
            metadata_path=metadata_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    if record.user_hash != user_hash or record.hash_prefix1 != prefix1 or record.hash_prefix2 != prefix2:
        report.invalid_metadata += 1
        report.issue(
            "user_hash_mismatch",
            "Record user hash fields do not match the directory layout",
            metadata_path=metadata_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    binary_path = _safe_user_child(user_root, record.storage_path)
    if binary_path is None:
        report.invalid_metadata += 1
        report.issue(
            "storage_path_unsafe",
            "Record storage_path escapes the user root",
            metadata_path=metadata_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    if not binary_path.exists():
        candidate = metadata_path.parent / record.stored_filename
        if candidate.exists():
            binary_path = candidate
        else:
            report.missing_binary += 1
            report.issue(
                "missing_binary",
                "Legacy binary is missing",
                metadata_path=metadata_path,
                storage_path=binary_path,
                upload_id=record.upload_id,
                user_hash=record.user_hash,
            )
            return None

    actual_storage_rel = manager._relative_path(user_root, binary_path)
    if actual_storage_rel is None:
        report.invalid_metadata += 1
        report.issue(
            "storage_path_unsafe",
            "Resolved binary path escapes the user root",
            metadata_path=metadata_path,
            storage_path=binary_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    try:
        checksum, size_bytes = manager._compute_file_digest(binary_path)
    except OSError as exc:
        report.invalid_metadata += 1
        report.issue(
            "binary_read_failed",
            str(exc),
            metadata_path=metadata_path,
            storage_path=binary_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    if checksum != record.sha256 or size_bytes != record.size_bytes:
        report.invalid_metadata += 1
        report.issue(
            "binary_digest_mismatch",
            "Legacy binary checksum or size does not match metadata",
            metadata_path=metadata_path,
            storage_path=binary_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return None

    report.valid_metadata += 1
    report.bytes_seen += size_bytes
    return record, user_root, binary_path, actual_storage_rel


def _copy_to_temp(manager: AttachmentManager, source_path: Path) -> Path:
    temp_dir = manager.store.blob_base_path / "_migration_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="migrate-",
        suffix=".tmp",
        dir=temp_dir,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle as destination, source_path.open("rb") as source:
            shutil.copyfileobj(source, destination, length=manager.CHUNK_SIZE)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _metadata_signature_for_storage_path(
    manager: AttachmentManager,
    record: AttachmentRecord,
    legacy_storage_path: str,
) -> str:
    payload = record.model_dump()
    payload["storage_path"] = legacy_storage_path
    return manager._compute_metadata_signature(payload)


def _migrate_one(
    manager: AttachmentManager,
    record: AttachmentRecord,
    binary_path: Path,
    legacy_storage_path: str,
    report: MigrationReport,
) -> None:
    store = manager.store
    if store.deletion_exists(user_hash=record.user_hash, upload_id=record.upload_id):
        report.skipped_tombstoned += 1
        report.action("skip_tombstone", "Upload has a deletion tombstone", upload_id=record.upload_id)
        return

    existing = store.get_upload(user_hash=record.user_hash, upload_id=record.upload_id)
    if existing is not None:
        if (
            existing.status == "active"
            and existing.blob.status == "ready"
            and existing.blob.sha256 == record.sha256
            and existing.blob.size_bytes == record.size_bytes
            and existing.origin == record.origin
            and existing.intended_usage == record.intended_usage
            and existing.original_filename == record.original_filename
            and existing.stored_filename == record.stored_filename
            and existing.mime_type == record.mime_type
            and existing.declared_size == record.declared_size
            and existing.declared_mime == record.declared_mime
            and existing.detected_mime == record.detected_mime
            and existing.original_url == record.original_url
            and existing.legacy_storage_path == legacy_storage_path
            and existing.legacy_metadata_path == record.metadata_path
            and existing.metadata_signature == _metadata_signature_for_storage_path(
                manager,
                record,
                legacy_storage_path,
            )
        ):
            report.skipped_existing += 1
            report.action("skip_existing", "Upload already exists in dedupe store", upload_id=record.upload_id)
            return
        report.issue(
            "existing_upload_mismatch",
            "Existing DB upload row does not match legacy metadata",
            storage_path=binary_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return

    kind = manager._attachment_kind(final_mime=record.mime_type, stored_filename=record.stored_filename)
    metadata_signature = _metadata_signature_for_storage_path(manager, record, legacy_storage_path)
    report.action(
        "migrate",
        "Would migrate legacy upload" if report.dry_run else "Migrating legacy upload",
        upload_id=record.upload_id,
        storage_path=str(binary_path),
    )
    if report.dry_run:
        return

    temp_path = _copy_to_temp(manager, binary_path)
    try:
        store.create_upload_from_temp(
            upload_id=record.upload_id,
            user_hash=record.user_hash,
            hash_prefix1=record.hash_prefix1,
            hash_prefix2=record.hash_prefix2,
            origin=record.origin,
            intended_usage=record.intended_usage,
            original_filename=record.original_filename,
            stored_filename=record.stored_filename,
            mime_type=record.mime_type,
            declared_size=record.declared_size,
            declared_mime=record.declared_mime,
            detected_mime=record.detected_mime,
            original_url=record.original_url,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            kind=kind,
            temp_path=temp_path,
            metadata_signature=metadata_signature,
            created_at=record.created_at,
            legacy_storage_path=legacy_storage_path,
            legacy_metadata_path=record.metadata_path,
            migrated_from_legacy=True,
        )
    except AttachmentStoreError as exc:
        report.issue(
            "store_insert_failed",
            str(exc),
            storage_path=binary_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        store.record_migration_issue(
            issue_code="store_insert_failed",
            detail=str(exc),
            legacy_metadata_path=record.metadata_path,
            legacy_storage_path=legacy_storage_path,
            upload_id=record.upload_id,
            user_hash=record.user_hash,
        )
        return

    report.migrated_uploads += 1
    report.bytes_migrated += record.size_bytes


def _command_init_schema(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    manager.store.ensure_schema()
    print(json.dumps({"ok": True, "db_path": str(manager.store.db_path)}, ensure_ascii=False, indent=2))
    return 0


def _command_inventory(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    report = MigrationReport(dry_run=True)
    for metadata_path in _legacy_metadata_files(manager):
        _load_legacy_record(manager, metadata_path, report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.issues and args.fail_on_issues else 0


def _command_backup(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    output_path = Path(args.output)
    manager.store.backup_to(output_path)
    print(json.dumps({"ok": True, "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


def _command_migrate(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.commit:
        raise MigrationCLIError("Use --dry-run or --commit")
    if args.dry_run and args.commit:
        raise MigrationCLIError("Use only one of --dry-run or --commit")

    manager = _resolve_manager()
    manager.store.ensure_schema()
    report = MigrationReport(dry_run=not args.commit)
    for metadata_path in _legacy_metadata_files(manager):
        loaded = _load_legacy_record(manager, metadata_path, report)
        if loaded is None:
            continue
        record, _user_root, binary_path, legacy_storage_path = loaded
        _migrate_one(manager, record, binary_path, legacy_storage_path, report)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.issues and args.fail_on_issues else 0


def _command_verify(args: argparse.Namespace) -> int:
    manager = _resolve_manager()
    report = MigrationReport(dry_run=True)
    for row in manager.store.iter_uploads():
        report.scanned_metadata += 1
        if row.status != "active":
            report.issue(
                "upload_not_active",
                "Upload row is not active",
                upload_id=row.upload_id,
                user_hash=row.user_hash,
            )
            continue
        if row.blob.status != "ready":
            report.issue(
                "blob_not_ready",
                "Upload points to a non-ready blob",
                upload_id=row.upload_id,
                user_hash=row.user_hash,
            )
            continue
        blob_path = manager.store.blob_path(row.blob.storage_key)
        if not blob_path.exists():
            report.issue(
                "blob_missing",
                "Ready blob is missing from disk",
                storage_path=blob_path,
                upload_id=row.upload_id,
                user_hash=row.user_hash,
            )
            continue
        checksum, size_bytes = manager.store.compute_file_digest(blob_path)
        if checksum != row.blob.sha256 or size_bytes != row.blob.size_bytes:
            report.issue(
                "blob_digest_mismatch",
                "Ready blob checksum or size does not match DB row",
                storage_path=blob_path,
                upload_id=row.upload_id,
                user_hash=row.user_hash,
            )
            continue
        report.valid_metadata += 1
        report.bytes_seen += size_bytes
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.issues else 0


def _safe_remove(path: Path, report: MigrationReport, *, category: str, dry_run: bool) -> None:
    if not path.exists():
        return
    report.action(category, "Would remove file" if dry_run else "Removing file", path=str(path))
    if dry_run:
        return
    path.unlink()


def _command_rehydrate_legacy(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.commit:
        raise MigrationCLIError("Use --dry-run or --commit")
    if args.dry_run and args.commit:
        raise MigrationCLIError("Use only one of --dry-run or --commit")

    dry_run = not args.commit
    manager = _resolve_manager()
    report = MigrationReport(dry_run=dry_run)
    for row in manager.store.iter_uploads():
        if row.status != "active" or row.blob.status != "ready":
            continue
        record = manager._record_from_store_row(row)
        user_root = manager._user_root(row.hash_prefix1, row.hash_prefix2, row.user_hash)
        legacy_binary = _safe_user_child(user_root, record.storage_path)
        legacy_metadata = _safe_user_child(user_root, record.metadata_path)
        if legacy_binary is None or legacy_metadata is None:
            report.issue(
                "legacy_path_unsafe",
                "Upload has unsafe legacy paths",
                upload_id=row.upload_id,
                user_hash=row.user_hash,
            )
            continue
        blob_path = manager.store.blob_path(row.blob.storage_key)
        if not blob_path.exists():
            report.issue("blob_missing", "DB blob is missing", storage_path=blob_path, upload_id=row.upload_id)
            continue
        report.action("rehydrate", "Would rehydrate legacy files" if dry_run else "Rehydrating legacy files", upload_id=row.upload_id)
        if dry_run:
            continue
        legacy_binary.parent.mkdir(parents=True, exist_ok=True)
        if legacy_binary.exists():
            legacy_checksum, legacy_size = manager.store.compute_file_digest(legacy_binary)
            if legacy_checksum != row.blob.sha256 or legacy_size != row.blob.size_bytes:
                report.issue(
                    "legacy_binary_mismatch",
                    "Existing legacy binary does not match DB blob; refusing to write rollback metadata",
                    storage_path=legacy_binary,
                    upload_id=row.upload_id,
                    user_hash=row.user_hash,
                )
                continue
        else:
            shutil.copy2(blob_path, legacy_binary)
        manager._write_metadata(legacy_metadata, record)
        manager._update_index(user_root / "uploads" / "index.json", record)
        report.migrated_uploads += 1
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.issues else 0


def _command_cleanup_legacy(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.commit:
        raise MigrationCLIError("Use --dry-run or --commit")
    if args.dry_run and args.commit:
        raise MigrationCLIError("Use only one of --dry-run or --commit")

    dry_run = not args.commit
    manager = _resolve_manager()
    report = MigrationReport(dry_run=dry_run)

    for row in manager.store.iter_uploads():
        if not row.migrated_from_legacy or row.status != "active" or row.blob.status != "ready":
            continue
        record = manager._record_from_store_row(row)
        user_root = manager._user_root(row.hash_prefix1, row.hash_prefix2, row.user_hash)
        legacy_binary = _safe_user_child(user_root, record.storage_path)
        legacy_metadata = _safe_user_child(user_root, record.metadata_path)
        blob_path = manager.store.blob_path(row.blob.storage_key)
        if legacy_binary is None or legacy_metadata is None:
            report.issue("legacy_path_unsafe", "Migrated upload has unsafe legacy paths", upload_id=row.upload_id)
            continue
        if not blob_path.exists():
            report.issue("blob_missing", "Cannot cleanup legacy copy while DB blob is missing", upload_id=row.upload_id)
            continue
        checksum, size_bytes = manager.store.compute_file_digest(blob_path)
        if checksum != row.blob.sha256 or size_bytes != row.blob.size_bytes:
            report.issue("blob_digest_mismatch", "Cannot cleanup legacy copy while DB blob is invalid", upload_id=row.upload_id)
            continue
        if legacy_binary.exists():
            legacy_checksum, legacy_size = manager.store.compute_file_digest(legacy_binary)
            if legacy_checksum != row.blob.sha256 or legacy_size != row.blob.size_bytes:
                report.issue(
                    "legacy_binary_mismatch",
                    "Cannot cleanup legacy copy because it no longer matches the DB blob",
                    storage_path=legacy_binary,
                    upload_id=row.upload_id,
                    user_hash=row.user_hash,
                )
                continue
        _safe_remove(legacy_binary, report, category="cleanup_legacy_binary", dry_run=dry_run)
        _safe_remove(legacy_metadata, report, category="cleanup_legacy_metadata", dry_run=dry_run)
        if not dry_run:
            manager._remove_index_entry(user_root / "uploads" / "index.json", row.upload_id)

    tombstones = {
        (str(item["user_hash"]), str(item["upload_id"]))
        for item in manager.store.iter_deletions()
        if item.get("user_hash") and item.get("upload_id")
    }
    if tombstones:
        for metadata_path in _legacy_metadata_files(manager):
            loaded = _load_legacy_record(manager, metadata_path, report)
            if loaded is None:
                continue
            record, user_root, binary_path, _legacy_storage_path = loaded
            if (record.user_hash, record.upload_id) not in tombstones:
                continue
            _safe_remove(binary_path, report, category="cleanup_tombstoned_binary", dry_run=dry_run)
            _safe_remove(metadata_path, report, category="cleanup_tombstoned_metadata", dry_run=dry_run)
            if not dry_run:
                manager._remove_index_entry(user_root / "uploads" / "index.json", record.upload_id)

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 1 if report.issues else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicated attachment storage migration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-schema", help="Create dedupe DB schema").set_defaults(func=_command_init_schema)

    inventory = subparsers.add_parser("inventory", help="Scan legacy attachment metadata")
    inventory.add_argument("--fail-on-issues", action="store_true", help="Exit non-zero when issues are found")
    inventory.set_defaults(func=_command_inventory)

    backup = subparsers.add_parser("backup", help="Create a consistent SQLite backup")
    backup.add_argument("--output", required=True, help="Backup SQLite path")
    backup.set_defaults(func=_command_backup)

    migrate = subparsers.add_parser("migrate", help="Backfill dedupe DB from legacy files")
    migrate.add_argument("--dry-run", action="store_true", help="Validate only")
    migrate.add_argument("--commit", action="store_true", help="Write DB rows and blobs")
    migrate.add_argument("--fail-on-issues", action="store_true", help="Exit non-zero when issues are found")
    migrate.set_defaults(func=_command_migrate)

    verify = subparsers.add_parser("verify", help="Verify DB rows and physical blobs")
    verify.set_defaults(func=_command_verify)

    rehydrate = subparsers.add_parser("rehydrate-legacy", help="Restore legacy files from DB blobs")
    rehydrate.add_argument("--dry-run", action="store_true", help="Show intended writes only")
    rehydrate.add_argument("--commit", action="store_true", help="Write legacy files")
    rehydrate.set_defaults(func=_command_rehydrate_legacy)

    cleanup = subparsers.add_parser("cleanup-legacy", help="Remove migrated/tombstoned legacy files")
    cleanup.add_argument("--dry-run", action="store_true", help="Show intended removals only")
    cleanup.add_argument("--commit", action="store_true", help="Remove legacy files")
    cleanup.set_defaults(func=_command_cleanup_legacy)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (AttachmentValidationError, MigrationCLIError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(main())
