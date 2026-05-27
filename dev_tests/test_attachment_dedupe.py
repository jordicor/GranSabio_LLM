import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from config import AttachmentSettings
from services.attachment_manager import (
    AttachmentManager,
    AttachmentNotFoundError,
    AttachmentValidationError,
)
from services.attachment_store import AttachmentStoreIntegrityError
from tools.attachments_cli import _command_delete, _command_list


def _settings(tmp_path, **overrides):
    values = {
        "dedupe_db_path": str(tmp_path / "attachments" / "attachments.sqlite3"),
        "blob_base_path": str(tmp_path / "attachment_blobs"),
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
    assert first_resolved.metadata_path == manager.store.db_path


@pytest.mark.asyncio
async def test_metadata_json_filename_is_safe_in_db_storage(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b'{"payload": true}',
        filename="metadata.json",
        mime_type="application/json",
    )

    assert record.original_filename == "metadata.json"
    assert record.stored_filename != "metadata.json"
    assert record.storage_path == f"dedupe/{record.upload_id}/{record.stored_filename}"
    assert record.metadata_path == f"dedupe/{record.upload_id}/metadata.json"

    resolved = manager.resolve_attachment(username="alice", upload_id=record.upload_id)
    assert resolved.binary_path.read_bytes() == b'{"payload": true}'
    result = manager.delete_attachment(
        username="alice",
        upload_id=record.upload_id,
        reason="test delete",
    )
    assert result == {"db_backed": True, "removed_files": 0, "blob_deleted": False}


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


def test_attachment_settings_no_longer_expose_rollout_or_legacy_flags(tmp_path):
    settings = _settings(tmp_path)

    for removed_name in (
        "base_path",
        "index_history_limit",
        "dedupe_read_enabled",
        "dedupe_write_enabled",
        "legacy_read_fallback_enabled",
        "legacy_write_index_enabled",
    ):
        assert not hasattr(settings, removed_name)


@pytest.mark.asyncio
async def test_missing_db_row_does_not_fall_back_to_user_index_files(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")

    with pytest.raises(AttachmentNotFoundError):
        manager.get_metadata(username="alice", upload_id="missing-upload")

    with pytest.raises(AttachmentNotFoundError):
        manager.resolve_attachment(username="alice", upload_id="missing-upload")


@pytest.mark.asyncio
async def test_delete_attachment_tombstones_db_row_and_leaves_shared_blob_for_gc(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b"delete me",
        filename="delete.txt",
        mime_type="text/plain",
    )

    resolved = manager.resolve_attachment(username="alice", upload_id=record.upload_id)
    result = manager.delete_attachment(username="alice", upload_id=record.upload_id)

    assert result == {"db_backed": True, "removed_files": 0, "blob_deleted": False}
    assert resolved.binary_path.exists()
    assert manager.store.count_ready_blobs() == 1
    with pytest.raises(AttachmentNotFoundError):
        manager.get_metadata(username="alice", upload_id=record.upload_id)


@pytest.mark.asyncio
async def test_cleanup_expires_old_uploads_and_collects_unreferenced_blobs(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path, blob_gc_enabled=True), pepper="test-pepper")
    prefix1, prefix2, user_hash = manager.generate_user_hash("alice")
    data = b"old upload bytes"
    temp_path = manager.store.blob_base_path / "_test_tmp" / "old.tmp"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(data)
    sha256, size_bytes = manager.store.compute_file_digest(temp_path)

    manager.store.create_upload_from_temp(
        upload_id="old-upload",
        user_hash=user_hash,
        hash_prefix1=prefix1,
        hash_prefix2=prefix2,
        origin="upload",
        intended_usage="context",
        original_filename="old.txt",
        stored_filename="old.txt",
        mime_type="text/plain",
        declared_size=None,
        declared_mime="text/plain",
        detected_mime="text/plain",
        original_url=None,
        sha256=sha256,
        size_bytes=size_bytes,
        kind="text",
        temp_path=temp_path,
        metadata_signature="test-signature",
        created_at="2020-01-01T00:00:00Z",
    )

    report = manager.run_cleanup(dry_run=False, username="alice", retention_days=1)

    assert report.retention_expired == 1
    assert report.attachments_removed == 1
    assert manager.store.count_ready_blobs() == 0
    with pytest.raises(AttachmentNotFoundError):
        manager.get_metadata(username="alice", upload_id="old-upload")


@pytest.mark.asyncio
async def test_cleanup_keeps_unreferenced_blobs_when_gc_is_disabled(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b"gc disabled",
        filename="gc.txt",
        mime_type="text/plain",
    )
    manager.delete_attachment(username="alice", upload_id=record.upload_id)

    report = manager.run_cleanup(dry_run=False, username="alice")

    assert manager.store.count_ready_blobs() == 1
    assert any("garbage collection is disabled" in issue for issue in report.issues)


@pytest.mark.asyncio
async def test_url_cache_reuses_current_db_record(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    cached_record = await manager.store_bytes(
        username="alice",
        data=b"cached url bytes",
        filename="cached.txt",
        mime_type="text/plain",
    )
    url = "https://example.com/cached.txt"
    manager._recent_url_cache[("alice", url)] = (
        cached_record.upload_id,
        cached_record.sha256,
        time.time() + 60,
    )
    manager.url_fetcher.send_pinned_request = AsyncMock(
        side_effect=AttachmentValidationError("network attempted")
    )

    record = await manager.store_from_url(username="alice", url=url)

    assert record.upload_id == cached_record.upload_id
    manager.url_fetcher.send_pinned_request.assert_not_called()


def test_decode_base64_payload_rejects_estimated_oversize():
    with pytest.raises(AttachmentValidationError, match="Payload too large"):
        AttachmentManager.decode_base64_payload("A" * 16, max_decoded_size=4)


@pytest.mark.asyncio
async def test_url_fetcher_closes_response_after_stream_consumed(tmp_path):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    request = httpx.Request("GET", "https://93.184.216.34/file.txt")
    response = httpx.Response(
        200,
        headers={"Content-Type": "text/plain", "Content-Length": "4"},
        content=b"test",
        request=request,
    )
    manager.url_fetcher.send_pinned_request = AsyncMock(return_value=response)

    async with manager.url_fetcher.fetch("https://example.com/file.txt") as fetched:
        chunks = []
        async for chunk in fetched.data_stream:
            chunks.append(chunk)

    assert b"".join(chunks) == b"test"
    assert response.is_closed


@pytest.mark.asyncio
async def test_cli_list_reads_db_uploads(tmp_path, monkeypatch, capsys):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b"cli list",
        filename="cli.txt",
        mime_type="text/plain",
    )
    other = await manager.store_bytes(
        username="bob",
        data=b"bob cli list",
        filename="bob.txt",
        mime_type="text/plain",
    )
    monkeypatch.setattr(
        manager.store,
        "iter_uploads",
        lambda: pytest.fail("CLI list must query by user in SQL"),
    )
    monkeypatch.setattr("tools.attachments_cli._resolve_manager", lambda: manager)

    exit_code = _command_list(SimpleNamespace(username="alice", user_hash=None, limit=None, as_json=True))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert record.upload_id in output
    assert other.upload_id not in output
    assert "cli.txt" in output


@pytest.mark.asyncio
async def test_cli_commit_delete_tombstones_db_upload(tmp_path, monkeypatch):
    manager = AttachmentManager(settings=_settings(tmp_path), pepper="test-pepper")
    record = await manager.store_bytes(
        username="alice",
        data=b"cli delete",
        filename="delete.txt",
        mime_type="text/plain",
    )
    monkeypatch.setattr("tools.attachments_cli._resolve_manager", lambda: manager)

    exit_code = _command_delete(
        SimpleNamespace(username="alice", upload_id=record.upload_id, commit=True)
    )

    assert exit_code == 0
    with pytest.raises(AttachmentNotFoundError):
        manager.get_metadata(username="alice", upload_id=record.upload_id)
