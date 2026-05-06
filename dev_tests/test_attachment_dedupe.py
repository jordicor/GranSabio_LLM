import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from config import AttachmentSettings
from services.attachment_manager import (
    AttachmentError,
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentValidationError,
)
from services.attachment_store import AttachmentStoreIntegrityError
from tools.attachments_dedupe_migration import (
    MigrationReport,
    _command_cleanup_legacy,
    _load_legacy_record,
    _migrate_one,
)
from tools.attachments_cli import _command_delete


def _settings(tmp_path, **overrides):
    values = {
        "base_path": str(tmp_path / "users"),
        "dedupe_db_path": str(tmp_path / "attachments" / "attachments.sqlite3"),
        "blob_base_path": str(tmp_path / "attachment_blobs"),
        "dedupe_read_enabled": True,
        "dedupe_write_enabled": True,
        "legacy_write_index_enabled": False,
    }
    values.update(overrides)
    return AttachmentSettings(**values)


@pytest.mark.asyncio
async def test_dedupe_write_reuses_blob_for_distinct_uploads(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    data = b"same attachment bytes"

    first = await manager.store_bytes(
        username="alice",
        data=data,
        filename="first.txt",
        mime_type="text/plain",
    )
    second = await manager.store_bytes(
        username="alice",
        data=data,
        filename="second.txt",
        mime_type="text/plain",
    )

    assert first.upload_id != second.upload_id
    assert first.sha256 == second.sha256
    assert manager.store.count_ready_blobs() == 1

    first_resolved = manager.resolve_attachment(username="alice", upload_id=first.upload_id)
    second_resolved = manager.resolve_attachment(username="alice", upload_id=second.upload_id)
    assert first_resolved.binary_path == second_resolved.binary_path
    assert first_resolved.binary_path.read_bytes() == data


@pytest.mark.asyncio
async def test_metadata_json_filename_is_reserved_for_mirror(tmp_path):
    manager = AttachmentManager(
        settings=_settings(tmp_path, legacy_write_index_enabled=True),
        pepper="test-pepper",
    )
    record = await manager.store_bytes(
        username="alice",
        data=b'{"payload": true}',
        filename="metadata.json",
        mime_type="application/json",
    )

    assert record.original_filename == "metadata.json"
    assert record.stored_filename != "metadata.json"
    assert Path(record.storage_path).name == record.stored_filename
    assert Path(record.metadata_path).name == "metadata.json"
    assert record.storage_path != record.metadata_path

    resolved = manager.resolve_attachment(username="alice", upload_id=record.upload_id)
    assert resolved.binary_path.read_bytes() == b'{"payload": true}'
    result = manager.delete_attachment(
        username="alice",
        upload_id=record.upload_id,
        reason="test delete",
    )
    assert result["db_backed"] is True


@pytest.mark.asyncio
async def test_concurrent_identical_uploads_reuse_blob_without_conflict(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    data = b"same concurrent attachment bytes"

    async def store_one(index):
        return await manager.store_bytes(
            username="alice",
            data=data,
            filename=f"file-{index}.txt",
            mime_type="text/plain",
        )

    records = await asyncio.gather(*(store_one(index) for index in range(4)))

    assert len({record.upload_id for record in records}) == 4
    assert len({record.sha256 for record in records}) == 1
    assert manager.store.count_ready_blobs() == 1


@pytest.mark.asyncio
async def test_store_verifies_temp_bytes_before_reusing_existing_blob(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b"canonical bytes",
        filename="canonical.txt",
        mime_type="text/plain",
    )
    prefix1, prefix2, user_hash = manager.generate_user_hash("alice")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "bad-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"different bytes")

    with pytest.raises(AttachmentStoreIntegrityError):
        manager.store.create_upload_from_temp(
            upload_id="manual-mismatch",
            user_hash=user_hash,
            hash_prefix1=prefix1,
            hash_prefix2=prefix2,
            origin="upload",
            intended_usage="context",
            original_filename="bad.txt",
            stored_filename="bad.txt",
            mime_type="text/plain",
            declared_size=None,
            declared_mime="text/plain",
            detected_mime="text/plain",
            original_url=None,
            sha256=record.sha256,
            size_bytes=record.size_bytes,
            kind="text",
            temp_path=temp_path,
            metadata_signature=record.metadata_signature,
            created_at=record.created_at,
        )

    assert not temp_path.exists()


@pytest.mark.asyncio
async def test_dedupe_read_falls_back_to_legacy_when_row_missing(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"legacy bytes",
        filename="legacy.txt",
        mime_type="text/plain",
    )

    read_manager = AttachmentManager(
        settings=_settings(
            tmp_path,
            dedupe_read_enabled=True,
            dedupe_write_enabled=False,
            legacy_read_fallback_enabled=True,
        ),
        pepper="test-pepper",
    )

    record = read_manager.get_metadata(username="alice", upload_id=legacy_record.upload_id)
    resolved = read_manager.resolve_attachment(username="alice", upload_id=legacy_record.upload_id)

    assert record.upload_id == legacy_record.upload_id
    assert resolved.binary_path.exists()
    assert resolved.binary_path.read_bytes() == b"legacy bytes"


@pytest.mark.asyncio
async def test_dedupe_tombstone_blocks_legacy_fallback(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"legacy bytes",
        filename="legacy.txt",
        mime_type="text/plain",
    )

    read_manager = AttachmentManager(
        settings=_settings(
            tmp_path,
            dedupe_read_enabled=True,
            dedupe_write_enabled=False,
            legacy_read_fallback_enabled=True,
        ),
        pepper="test-pepper",
    )
    _prefix1, _prefix2, user_hash = read_manager.generate_user_hash("alice")
    read_manager.store.record_deletion_tombstone(
        user_hash=user_hash,
        upload_id=legacy_record.upload_id,
        reason="test tombstone",
    )

    with pytest.raises(AttachmentNotFoundError):
        read_manager.get_metadata(username="alice", upload_id=legacy_record.upload_id)

    with pytest.raises(AttachmentNotFoundError):
        read_manager.resolve_attachment(username="alice", upload_id=legacy_record.upload_id)


@pytest.mark.asyncio
async def test_db_backed_delete_removes_migrated_legacy_binary(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"legacy bytes",
        filename="legacy.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    legacy_binary = user_root / legacy_record.storage_path
    legacy_metadata = user_root / legacy_record.metadata_path

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "legacy-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(legacy_binary.read_bytes())
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind=manager._attachment_kind(
            final_mime=legacy_record.mime_type,
            stored_filename=legacy_record.stored_filename,
        ),
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )

    result = manager.delete_attachment(
        username="alice",
        upload_id=legacy_record.upload_id,
        reason="test delete",
    )

    assert result["db_backed"] is True
    assert result["blob_deleted"] is False
    assert not legacy_binary.exists()
    assert not legacy_metadata.exists()
    with pytest.raises(AttachmentNotFoundError):
        manager.get_metadata(username="alice", upload_id=legacy_record.upload_id)


@pytest.mark.asyncio
async def test_migration_preserves_actual_fallback_binary_path(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"fallback bytes",
        filename="fallback.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    metadata_path = user_root / legacy_record.metadata_path
    actual_storage_path = legacy_record.storage_path
    legacy_record.storage_path = "uploads/2099/01/missing/fallback.txt"
    legacy_record.metadata_signature = legacy_manager._compute_metadata_signature(
        legacy_record.model_dump()
    )
    legacy_manager._write_metadata(metadata_path, legacy_record)

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    report = MigrationReport(dry_run=False)
    loaded = _load_legacy_record(manager, metadata_path, report)
    assert loaded is not None
    record, _user_root, binary_path, legacy_storage_path = loaded
    _migrate_one(manager, record, binary_path, legacy_storage_path, report)

    row = manager.store.get_upload(
        user_hash=legacy_record.user_hash,
        upload_id=legacy_record.upload_id,
    )
    assert row is not None
    assert row.legacy_storage_path == actual_storage_path
    returned = manager.get_metadata(username="alice", upload_id=legacy_record.upload_id)
    assert returned.storage_path == actual_storage_path
    assert returned.metadata_signature == manager._compute_metadata_signature(returned.model_dump())


@pytest.mark.asyncio
async def test_migration_rerun_reports_existing_row_with_bad_legacy_path(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"fallback bytes",
        filename="fallback.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    metadata_path = user_root / legacy_record.metadata_path
    actual_storage_path = legacy_record.storage_path
    legacy_record.storage_path = "uploads/2099/01/missing/fallback.txt"
    legacy_record.metadata_signature = legacy_manager._compute_metadata_signature(
        legacy_record.model_dump()
    )
    legacy_manager._write_metadata(metadata_path, legacy_record)

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "bad-existing.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes((user_root / actual_storage_path).read_bytes())
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )

    report = MigrationReport(dry_run=False)
    loaded = _load_legacy_record(manager, metadata_path, report)
    assert loaded is not None
    record, _user_root, binary_path, legacy_storage_path = loaded
    _migrate_one(manager, record, binary_path, legacy_storage_path, report)

    assert any(issue["code"] == "existing_upload_mismatch" for issue in report.issues)


@pytest.mark.asyncio
async def test_cleanup_legacy_skips_mismatched_legacy_binary(tmp_path, monkeypatch):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"cleanup bytes",
        filename="cleanup.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    legacy_binary = user_root / legacy_record.storage_path

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "cleanup-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"cleanup bytes")
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )
    legacy_binary.write_bytes(b"changed after migration")
    monkeypatch.setattr("tools.attachments_dedupe_migration._resolve_manager", lambda: manager)

    exit_code = _command_cleanup_legacy(SimpleNamespace(dry_run=False, commit=True))

    assert exit_code == 1
    assert legacy_binary.exists()
    assert legacy_binary.read_bytes() == b"changed after migration"


@pytest.mark.asyncio
async def test_db_backed_delete_refuses_mismatched_legacy_binary_before_tombstone(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"delete bytes",
        filename="delete.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    legacy_binary = user_root / legacy_record.storage_path

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "delete-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"delete bytes")
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )
    legacy_binary.write_bytes(b"changed before delete")

    with pytest.raises(AttachmentError, match="no longer matches"):
        manager.delete_attachment(
            username="alice",
            upload_id=legacy_record.upload_id,
            reason="test delete",
        )

    assert manager.store.get_upload(user_hash=legacy_record.user_hash, upload_id=legacy_record.upload_id) is not None
    assert not manager.store.deletion_exists(user_hash=legacy_record.user_hash, upload_id=legacy_record.upload_id)
    assert legacy_binary.exists()


@pytest.mark.asyncio
async def test_db_backed_delete_can_retry_mirror_cleanup_after_tombstone(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"retry bytes",
        filename="retry.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    legacy_binary = user_root / legacy_record.storage_path

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "retry-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"retry bytes")
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )

    with patch.object(manager.store, "delete_upload", side_effect=RuntimeError("db unavailable")):
        with pytest.raises(RuntimeError, match="db unavailable"):
            manager.delete_attachment(
                username="alice",
                upload_id=legacy_record.upload_id,
                reason="test delete",
            )

    assert not legacy_binary.exists()
    manager.store.delete_upload(
        user_hash=legacy_record.user_hash,
        upload_id=legacy_record.upload_id,
        reason="manual tombstone",
    )

    result = manager.delete_attachment(
        username="alice",
        upload_id=legacy_record.upload_id,
        reason="retry cleanup",
    )

    assert result["db_backed"] is True
    assert result["removed_files"] >= 1
    assert not (user_root / legacy_record.metadata_path).exists()


@pytest.mark.asyncio
async def test_cli_commit_delete_retries_tombstoned_cleanup(tmp_path, monkeypatch):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    legacy_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"cli retry bytes",
        filename="cli-retry.txt",
        mime_type="text/plain",
    )
    user_root = legacy_manager._user_root(
        legacy_record.hash_prefix1,
        legacy_record.hash_prefix2,
        legacy_record.user_hash,
    )
    legacy_binary = user_root / legacy_record.storage_path

    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    temp_path = manager.store.blob_base_path / "_test_tmp" / "cli-retry-copy.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(b"cli retry bytes")
    manager.store.create_upload_from_temp(
        upload_id=legacy_record.upload_id,
        user_hash=legacy_record.user_hash,
        hash_prefix1=legacy_record.hash_prefix1,
        hash_prefix2=legacy_record.hash_prefix2,
        origin=legacy_record.origin,
        intended_usage=legacy_record.intended_usage,
        original_filename=legacy_record.original_filename,
        stored_filename=legacy_record.stored_filename,
        mime_type=legacy_record.mime_type,
        declared_size=legacy_record.declared_size,
        declared_mime=legacy_record.declared_mime,
        detected_mime=legacy_record.detected_mime,
        original_url=legacy_record.original_url,
        sha256=legacy_record.sha256,
        size_bytes=legacy_record.size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature=legacy_record.metadata_signature,
        created_at=legacy_record.created_at,
        legacy_storage_path=legacy_record.storage_path,
        legacy_metadata_path=legacy_record.metadata_path,
        migrated_from_legacy=True,
    )
    manager.store.delete_upload(
        user_hash=legacy_record.user_hash,
        upload_id=legacy_record.upload_id,
        reason="pre-existing tombstone",
    )
    monkeypatch.setattr("tools.attachments_cli._resolve_manager", lambda: manager)

    exit_code = _command_delete(
        SimpleNamespace(username="alice", upload_id=legacy_record.upload_id, commit=True)
    )

    assert exit_code == 0
    assert not legacy_binary.exists()
    assert not (user_root / legacy_record.metadata_path).exists()


@pytest.mark.asyncio
async def test_url_cache_disabled_when_dedupe_read_enabled(tmp_path):
    legacy_settings = _settings(
        tmp_path,
        dedupe_read_enabled=False,
        dedupe_write_enabled=False,
        legacy_write_index_enabled=True,
    )
    legacy_manager = AttachmentManager(settings=legacy_settings, pepper="test-pepper")
    cached_record = await legacy_manager.store_bytes(
        username="alice",
        data=b"cached url bytes",
        filename="cached.txt",
        mime_type="text/plain",
    )

    manager = AttachmentManager(
        settings=_settings(
            tmp_path,
            dedupe_read_enabled=True,
            dedupe_write_enabled=False,
            legacy_read_fallback_enabled=True,
        ),
        pepper="test-pepper",
    )
    url = "https://example.com/cached.txt"
    manager._recent_url_cache[("alice", url)] = (
        cached_record.upload_id,
        cached_record.sha256,
        time.time() + 60,
    )
    manager._send_pinned_request = AsyncMock(
        side_effect=AttachmentValidationError("network attempted")
    )

    with pytest.raises(AttachmentValidationError, match="network attempted"):
        await manager.store_from_url(username="alice", url=url)
